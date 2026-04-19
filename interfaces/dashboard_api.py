"""Dashboard API — FastAPI router for the web frontend.

GETs are read-only and unauth (internal network). Mutations (POST/PATCH) write
through the same tables the bot uses, and emit an AuditLog row tagged
tool_name="dashboard:<op>" so the agent's recall + Pipeline Watcher see when
the user touched something in the UI vs. via chat.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy import func, select, desc

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

_sm = None  # set by mount_dashboard_api()

DASH_SESSION = "dashboard"  # session_id stamped on dashboard-originated writes


def mount_dashboard_api(app, session_maker):
    global _sm
    _sm = session_maker
    app.include_router(router)


# ---- Write helpers -------------------------------------------------


async def _audit(op: str, args: dict, status: str = "ok", summary: str = ""):
    """Best-effort audit log so dashboard writes show up alongside bot tool calls.

    Failure here must never block the user-visible mutation, hence the broad
    except.
    """
    from ..db.models import AuditLog
    try:
        async with _sm() as s:
            s.add(AuditLog(
                tool_name=f"dashboard:{op}",
                args_summary=json.dumps(args, default=str)[:500],
                result_status=status,
                result_summary=summary[:500],
                session_id=DASH_SESSION,
                safety="auto",
            ))
            await s.commit()
    except Exception:
        pass


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# ---- Pipeline (deals by stage) ------------------------------------


@router.get("/pipeline")
async def pipeline():
    """All active deals grouped by stage for the kanban board."""
    from ..db.models import Deal, Company
    async with _sm() as s:
        deals = (await s.execute(
            select(Deal).where(Deal.stage.notin_(["closed_won", "closed_lost"]))
            .order_by(Deal.updated_at.desc().nullslast())
        )).scalars().all()

        company_ids = {d.company_id for d in deals if d.company_id}
        companies = {}
        if company_ids:
            rows = (await s.execute(select(Company).where(Company.id.in_(company_ids)))).scalars().all()
            companies = {c.id: c.name for c in rows}

    stages = {}
    for d in deals:
        stage = d.stage or "prospect"
        if stage not in stages:
            stages[stage] = []
        stages[stage].append({
            "id": d.id, "name": d.name, "stage": stage,
            "value_usd": d.value_usd or 0,
            "company": companies.get(d.company_id, ""),
            "next_step": (d.next_step or "")[:100],
            "close_date": str(d.close_date) if d.close_date else None,
            "updated_at": str(d.updated_at),
        })

    stage_order = ["prospect", "qualified", "proposal", "negotiation"]
    ordered = {s: stages.get(s, []) for s in stage_order}
    for s in stages:
        if s not in ordered:
            ordered[s] = stages[s]
    return {"stages": ordered, "total_deals": len(deals)}


# ---- Deal detail --------------------------------------------------


@router.get("/deals/{deal_id}")
async def deal_detail(deal_id: str):
    """Full deal context — MEDDIC, stakeholders, meetings, actions, bids."""
    from ..db.models import (
        ActionItem, Bid, Contact, Deal, DealStakeholder, Meeting,
    )
    async with _sm() as s:
        d = await s.get(Deal, deal_id)
        if not d:
            return {"error": "not found"}

        stakeholders = (await s.execute(
            select(DealStakeholder).where(DealStakeholder.deal_id == deal_id)
        )).scalars().all()
        contact_ids = {st.contact_id for st in stakeholders}
        contacts = {}
        if contact_ids:
            rows = (await s.execute(select(Contact).where(Contact.id.in_(contact_ids)))).scalars().all()
            contacts = {c.id: {"name": c.name, "title": c.title, "personal_notes": c.personal_notes} for c in rows}

        meetings = (await s.execute(
            select(Meeting).where(Meeting.deal_id == deal_id).order_by(Meeting.date.desc()).limit(10)
        )).scalars().all()

        actions = (await s.execute(
            select(ActionItem).where(ActionItem.deal_id == deal_id).order_by(ActionItem.created_at.desc()).limit(15)
        )).scalars().all()

        bids = (await s.execute(
            select(Bid).where(Bid.deal_id == deal_id).order_by(Bid.submission_deadline.asc().nullslast())
        )).scalars().all()

        # MEDDIC gaps
        meddic = {
            "metrics": d.metrics or "",
            "economic_buyer": contacts.get(d.economic_buyer_id, {}).get("name", "") if d.economic_buyer_id else "",
            "champion": contacts.get(d.champion_id, {}).get("name", "") if d.champion_id else "",
            "decision_criteria": d.decision_criteria or "",
            "decision_process": d.decision_process or "",
            "paper_process": d.paper_process or "",
            "pain": d.pain or "",
        }
        gaps = [k for k, v in meddic.items() if not v]

    # Resolve plant + company for header context
    from ..db.models import Plant
    plant_name = ""
    company_name = ""
    if d.plant_id:
        async with _sm() as s2:
            pl = await s2.get(Plant, d.plant_id)
            if pl:
                plant_name = pl.name
    if d.company_id:
        from ..db.models import Company
        async with _sm() as s2:
            co = await s2.get(Company, d.company_id)
            if co:
                company_name = co.name

    return {
        "deal": {
            "id": d.id, "name": d.name, "stage": d.stage,
            "value_usd": d.value_usd, "close_date": str(d.close_date) if d.close_date else None,
            "next_step": d.next_step, "notes": d.notes, "competitors": d.competitors,
            "company_id": d.company_id, "company": company_name,
            "plant_id": d.plant_id, "plant": plant_name,
        },
        "meddic": meddic,
        "meddic_gaps": gaps,
        "stakeholders": [
            {
                "id": st.id,
                "contact_id": st.contact_id, "role": st.role,
                "sentiment": st.sentiment, "influence": st.influence,
                "name": contacts.get(st.contact_id, {}).get("name", ""),
                "title": contacts.get(st.contact_id, {}).get("title", ""),
            }
            for st in stakeholders
        ],
        "meetings": [
            {"id": m.id, "date": str(m.date), "summary": m.summary,
             "attendees": m.attendees, "decisions": m.decisions,
             "meeting_type": m.meeting_type or "other",
             "sentiment": m.sentiment or "unknown",
             "audio_processing_status": m.audio_processing_status or "idle",
             "competitors_mentioned": m.competitors_mentioned or "",
             "pricing_mentioned": m.pricing_mentioned or "",
             "has_transcript": bool(m.transcript)}
            for m in meetings
        ],
        "action_items": [
            {"id": a.id, "description": a.description, "status": a.status,
             "due_date": str(a.due_date) if a.due_date else None, "source": a.source}
            for a in actions
        ],
        "bids": [
            {"id": b.id, "name": b.name, "stage": b.stage,
             "value_usd": b.value_usd,
             "submission_deadline": str(b.submission_deadline) if b.submission_deadline else None}
            for b in bids
        ],
    }


# ---- Contacts directory -------------------------------------------


@router.get("/contacts")
async def contacts_list(q: str = "", limit: int = 50):
    from ..db.models import Contact, Company
    async with _sm() as s:
        query = select(Contact).order_by(Contact.updated_at.desc().nullslast()).limit(limit)
        if q:
            query = select(Contact).where(
                Contact.name.ilike(f"%{q}%") | Contact.email.ilike(f"%{q}%")
            ).limit(limit)
        rows = (await s.execute(query)).scalars().all()
        company_ids = {r.company_id for r in rows if r.company_id}
        companies = {}
        if company_ids:
            co = (await s.execute(select(Company).where(Company.id.in_(company_ids)))).scalars().all()
            companies = {c.id: c.name for c in co}

    return {
        "contacts": [
            {
                "id": r.id, "name": r.name, "title": r.title,
                "email": r.email, "phone": r.phone,
                "company": companies.get(r.company_id, ""),
                "personal_notes": (r.personal_notes or "")[:200],
                "last_touch": str(r.last_touch) if r.last_touch else None,
            }
            for r in rows
        ],
    }


# ---- Activity feed ------------------------------------------------


@router.get("/activity")
async def activity_feed(hours: int = 72, limit: int = 30):
    """Recent events across all entities — meetings, actions, reminders."""
    from ..db.models import ActionItem, Meeting, Reminder
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with _sm() as s:
        meetings = (await s.execute(
            select(Meeting).where(Meeting.created_at >= cutoff)
            .order_by(Meeting.created_at.desc()).limit(limit)
        )).scalars().all()

        actions = (await s.execute(
            select(ActionItem).where(ActionItem.created_at >= cutoff)
            .order_by(ActionItem.created_at.desc()).limit(limit)
        )).scalars().all()

        reminders = (await s.execute(
            select(Reminder).where(
                Reminder.created_at >= cutoff,
                Reminder.status.in_(["sent", "pending"]),
            ).order_by(Reminder.created_at.desc()).limit(limit)
        )).scalars().all()

    events = []
    for m in meetings:
        events.append({
            "type": "meeting", "id": m.id, "timestamp": str(m.created_at),
            "title": f"Meeting logged", "detail": (m.summary or "")[:150],
            "deal_id": m.deal_id,
        })
    for a in actions:
        events.append({
            "type": "action_item", "id": a.id, "timestamp": str(a.created_at),
            "title": f"Action: {a.description[:80]}", "detail": f"Status: {a.status}",
            "deal_id": a.deal_id,
        })
    for r in reminders:
        events.append({
            "type": "reminder", "id": r.id, "timestamp": str(r.created_at),
            "title": f"Reminder: {r.message[:80]}", "detail": f"Status: {r.status}",
            "deal_id": r.related_deal_id,
        })

    events.sort(key=lambda e: e["timestamp"], reverse=True)
    return {"events": events[:limit]}


# ---- Analytics summary -------------------------------------------


@router.get("/analytics")
async def analytics():
    """Pipeline summary stats for the dashboard header cards."""
    from ..db.models import ActionItem, Deal
    async with _sm() as s:
        deals = (await s.execute(select(Deal))).scalars().all()
        open_actions = (await s.execute(
            select(func.count()).select_from(ActionItem).where(ActionItem.status == "open")
        )).scalar()

    total_value = sum(d.value_usd or 0 for d in deals)
    by_stage = {}
    for d in deals:
        stage = d.stage or "unknown"
        by_stage.setdefault(stage, {"count": 0, "value": 0})
        by_stage[stage]["count"] += 1
        by_stage[stage]["value"] += d.value_usd or 0

    active = [d for d in deals if d.stage not in ("closed_won", "closed_lost")]
    won = [d for d in deals if d.stage == "closed_won"]

    return {
        "total_deals": len(deals),
        "active_deals": len(active),
        "total_pipeline_value": total_value,
        "active_pipeline_value": sum(d.value_usd or 0 for d in active),
        "won_value": sum(d.value_usd or 0 for d in won),
        "win_rate": round(len(won) / len(deals), 2) if deals else 0,
        "open_actions": open_actions or 0,
        "by_stage": by_stage,
    }


# =====================================================================
# WRITE endpoints — dashboard mutations. Round-trip into the same tables
# the bot reads, so a stage moved on the kanban shows up next time the
# agent asks "what stage is X in?". Each writes an AuditLog row tagged
# dashboard:<op> for downstream visibility.
# =====================================================================


_DEAL_STAGES = {"prospect", "qualified", "proposal", "negotiation", "closed_won", "closed_lost"}
_ACTION_STATUSES = {"open", "done", "snoozed"}
# Industrial buying-committee taxonomy. The original SaaS-style 6 (champion/EB/TB/
# blocker/coach/user) misses ~half of who shows up in a Bosch/Honeywell-class deal
# review — OT-cyber gates, parent-co standards committees, and operations/maint
# routinely kill deals that look healthy on a SaaS-shaped scorecard.
_STAKE_ROLES = {
    "champion", "economic_buyer", "technical_buyer", "blocker", "coach", "user",
    "ot_cyber", "it_cyber", "operations", "maintenance",
    "procurement", "legal", "finance", "parent_company_standards",
}
_STAKE_SENTIMENTS = {"supportive", "neutral", "opposed", "unknown"}
_STAKE_INFLUENCES = {"low", "medium", "high"}
_MEDDIC_FIELDS = {"metrics", "decision_criteria", "decision_process", "paper_process", "pain"}
_BID_STAGES = {"evaluating", "in_progress", "submitted", "won", "lost", "withdrawn"}
# Reminder statuses — extended beyond the bot-side default (pending|sent|cancelled|failed)
# with `done` to capture user-acknowledged-from-the-dashboard. Snooze is a state
# transition that bumps trigger_at + sets status back to pending so the
# ReminderService picks it up again at the new time.
_REMINDER_STATUSES = {"pending", "sent", "cancelled", "failed", "done"}
_PLANT_SITE_TYPES = {
    "refinery", "chemical", "power_gen", "water_wastewater", "manufacturing",
    "data_center", "pharma", "food_bev", "mining", "utility_substation", "other",
}
_SPEC_FAMILIES = {
    "hazardous_area", "functional_safety", "cybersecurity", "electrical",
    "export_control", "quality", "environmental", "other",
}
_COMPLIANCE_STATUSES = {"compliant", "partial", "exception", "not_applicable", "unanswered"}
_ASSET_TYPES = {
    "dcs", "plc", "hmi", "scada", "safety_system", "drive", "motor",
    "instrument", "network", "analyzer", "other",
}
_ASSET_VENDORS = {"us", "competitor", "partner"}
_COSELLER_ROLES = {"oem_rep", "channel_partner", "si_partner", "distributor", "consultant", "reseller"}
_COSELLER_STATUSES = {"active", "introduced", "dormant", "replaced"}
_CONTRACT_TYPES = {
    "pm_quarterly", "pm_annual", "24x7_support", "on_call",
    "parts_only", "training", "calibration", "other",
}
_CONTRACT_STATUSES = {"active", "expiring", "expired", "cancelled", "renewed"}


# ---- Deals --------------------------------------------------------


class DealCreate(BaseModel):
    name: str
    company_id: str | None = None
    plant_id: str | None = None
    stage: str = "prospect"
    value_usd: float = 0.0
    close_date: str | None = None
    next_step: str = ""
    notes: str = ""
    competitors: str = ""


@router.post("/deals")
async def create_deal(body: DealCreate):
    from ..db.models import Deal
    if body.stage not in _DEAL_STAGES:
        raise HTTPException(400, f"invalid stage: {body.stage}")
    async with _sm() as s:
        d = Deal(
            name=body.name,
            company_id=body.company_id or None,
            plant_id=body.plant_id or None,
            stage=body.stage,
            value_usd=body.value_usd,
            close_date=_parse_date(body.close_date),
            next_step=body.next_step,
            notes=body.notes,
            competitors=body.competitors,
        )
        s.add(d)
        await s.commit()
        await s.refresh(d)
    await _audit("deal.create", body.model_dump(), summary=f"{d.id} {d.name}")
    return {"id": d.id, "name": d.name, "stage": d.stage}


class DealPatch(BaseModel):
    name: str | None = None
    stage: str | None = None
    value_usd: float | None = None
    close_date: str | None = None
    next_step: str | None = None
    notes: str | None = None              # replaces notes (use append endpoint to add)
    notes_append: str | None = None       # append a paragraph to existing notes
    competitors: str | None = None
    company_id: str | None = None
    plant_id: str | None = None
    economic_buyer_id: str | None = None
    champion_id: str | None = None
    metrics: str | None = None
    decision_criteria: str | None = None
    decision_process: str | None = None
    paper_process: str | None = None
    pain: str | None = None


@router.patch("/deals/{deal_id}")
async def patch_deal(deal_id: str, body: DealPatch):
    from ..db.models import Deal
    async with _sm() as s:
        d = await s.get(Deal, deal_id)
        if not d:
            raise HTTPException(404, "deal not found")
        if body.stage is not None:
            if body.stage not in _DEAL_STAGES:
                raise HTTPException(400, f"invalid stage: {body.stage}")
            d.stage = body.stage
        if body.name is not None:
            d.name = body.name
        if body.value_usd is not None:
            d.value_usd = body.value_usd
        if body.close_date is not None:
            d.close_date = _parse_date(body.close_date)
        if body.next_step is not None:
            d.next_step = body.next_step
        if body.notes is not None:
            d.notes = body.notes
        if body.notes_append:
            d.notes = (d.notes + "\n\n" if d.notes else "") + body.notes_append
        if body.competitors is not None:
            d.competitors = body.competitors
        if body.company_id is not None:
            d.company_id = body.company_id or None
        if body.plant_id is not None:
            d.plant_id = body.plant_id or None
        if body.economic_buyer_id is not None:
            d.economic_buyer_id = body.economic_buyer_id or None
        if body.champion_id is not None:
            d.champion_id = body.champion_id or None
        for f in _MEDDIC_FIELDS:
            v = getattr(body, f)
            if v is not None:
                setattr(d, f, v)
        await s.commit()
    await _audit("deal.patch", {"id": deal_id, **body.model_dump(exclude_none=True)},
                 summary=f"{deal_id}")
    return {"id": deal_id, "updated": True}


# ---- Action items -------------------------------------------------


class ActionCreate(BaseModel):
    description: str
    due_date: str | None = None
    contact_id: str | None = None
    source: str = "dashboard"


@router.post("/deals/{deal_id}/actions")
async def create_action(deal_id: str, body: ActionCreate):
    from ..db.models import ActionItem, Deal
    async with _sm() as s:
        d = await s.get(Deal, deal_id)
        if not d:
            raise HTTPException(404, "deal not found")
        a = ActionItem(
            deal_id=deal_id,
            contact_id=body.contact_id or None,
            description=body.description,
            due_date=_parse_date(body.due_date),
            source=body.source,
        )
        s.add(a)
        await s.commit()
        await s.refresh(a)
    await _audit("action.create", {"deal_id": deal_id, **body.model_dump()},
                 summary=f"{a.id} {body.description[:80]}")
    return {
        "id": a.id, "description": a.description, "status": a.status,
        "due_date": str(a.due_date) if a.due_date else None, "source": a.source,
    }


class ActionPatch(BaseModel):
    description: str | None = None
    status: str | None = None
    due_date: str | None = None


@router.patch("/actions/{action_id}")
async def patch_action(action_id: str, body: ActionPatch):
    from ..db.models import ActionItem
    from datetime import datetime as _dt, timezone as _tz
    async with _sm() as s:
        a = await s.get(ActionItem, action_id)
        if not a:
            raise HTTPException(404, "action not found")
        if body.status is not None:
            if body.status not in _ACTION_STATUSES:
                raise HTTPException(400, f"invalid status: {body.status}")
            a.status = body.status
            if body.status == "done" and not a.completed_at:
                a.completed_at = _dt.now(_tz.utc)
            if body.status == "open":
                a.completed_at = None
        if body.description is not None:
            a.description = body.description
        if body.due_date is not None:
            a.due_date = _parse_date(body.due_date)
        await s.commit()
    await _audit("action.patch", {"id": action_id, **body.model_dump(exclude_none=True)},
                 summary=f"{action_id}")
    return {"id": action_id, "updated": True}


# ---- Stakeholders -------------------------------------------------


class StakeholderCreate(BaseModel):
    contact_id: str
    role: str
    sentiment: str = "unknown"
    influence: str = "medium"
    notes: str = ""


@router.post("/deals/{deal_id}/stakeholders")
async def create_stakeholder(deal_id: str, body: StakeholderCreate):
    from ..db.models import Deal, DealStakeholder
    if body.role not in _STAKE_ROLES:
        raise HTTPException(400, f"invalid role: {body.role}")
    if body.sentiment not in _STAKE_SENTIMENTS:
        raise HTTPException(400, f"invalid sentiment: {body.sentiment}")
    if body.influence not in _STAKE_INFLUENCES:
        raise HTTPException(400, f"invalid influence: {body.influence}")
    async with _sm() as s:
        d = await s.get(Deal, deal_id)
        if not d:
            raise HTTPException(404, "deal not found")
        existing = (await s.execute(
            select(DealStakeholder).where(
                DealStakeholder.deal_id == deal_id,
                DealStakeholder.contact_id == body.contact_id,
                DealStakeholder.role == body.role,
            )
        )).scalar_one_or_none()
        if existing:
            existing.sentiment = body.sentiment
            existing.influence = body.influence
            if body.notes:
                existing.notes = body.notes
            sh_id = existing.id
        else:
            st = DealStakeholder(
                deal_id=deal_id, contact_id=body.contact_id, role=body.role,
                sentiment=body.sentiment, influence=body.influence, notes=body.notes,
            )
            s.add(st)
            await s.commit()
            await s.refresh(st)
            sh_id = st.id
            await s.commit()
        await s.commit()
    await _audit("stakeholder.create", {"deal_id": deal_id, **body.model_dump()},
                 summary=f"{sh_id}")
    return {"id": sh_id, "deal_id": deal_id}


class StakeholderPatch(BaseModel):
    sentiment: str | None = None
    influence: str | None = None
    role: str | None = None
    notes: str | None = None


@router.patch("/stakeholders/{stakeholder_id}")
async def patch_stakeholder(stakeholder_id: str, body: StakeholderPatch):
    from ..db.models import DealStakeholder
    async with _sm() as s:
        st = await s.get(DealStakeholder, stakeholder_id)
        if not st:
            raise HTTPException(404, "stakeholder not found")
        if body.sentiment is not None:
            if body.sentiment not in _STAKE_SENTIMENTS:
                raise HTTPException(400, f"invalid sentiment: {body.sentiment}")
            st.sentiment = body.sentiment
        if body.influence is not None:
            if body.influence not in _STAKE_INFLUENCES:
                raise HTTPException(400, f"invalid influence: {body.influence}")
            st.influence = body.influence
        if body.role is not None:
            if body.role not in _STAKE_ROLES:
                raise HTTPException(400, f"invalid role: {body.role}")
            st.role = body.role
        if body.notes is not None:
            st.notes = body.notes
        await s.commit()
    await _audit("stakeholder.patch",
                 {"id": stakeholder_id, **body.model_dump(exclude_none=True)},
                 summary=f"{stakeholder_id}")
    return {"id": stakeholder_id, "updated": True}


@router.delete("/stakeholders/{stakeholder_id}")
async def delete_stakeholder(stakeholder_id: str):
    from ..db.models import DealStakeholder
    async with _sm() as s:
        st = await s.get(DealStakeholder, stakeholder_id)
        if not st:
            raise HTTPException(404, "stakeholder not found")
        await s.delete(st)
        await s.commit()
    await _audit("stakeholder.delete", {"id": stakeholder_id}, summary=stakeholder_id)
    return {"id": stakeholder_id, "deleted": True}


# ---- Contacts -----------------------------------------------------


class ContactCreate(BaseModel):
    name: str
    company_id: str | None = None
    title: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    personal_notes: str = ""


@router.post("/contacts")
async def create_contact(body: ContactCreate):
    from ..db.models import Contact
    async with _sm() as s:
        c = Contact(
            name=body.name, company_id=body.company_id or None,
            title=body.title, email=body.email, phone=body.phone,
            linkedin=body.linkedin, personal_notes=body.personal_notes,
        )
        s.add(c)
        await s.commit()
        await s.refresh(c)
    await _audit("contact.create", body.model_dump(), summary=f"{c.id} {c.name}")
    return {"id": c.id, "name": c.name}


class ContactPatch(BaseModel):
    name: str | None = None
    title: str | None = None
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    personal_notes: str | None = None
    company_id: str | None = None


@router.patch("/contacts/{contact_id}")
async def patch_contact(contact_id: str, body: ContactPatch):
    from ..db.models import Contact
    async with _sm() as s:
        c = await s.get(Contact, contact_id)
        if not c:
            raise HTTPException(404, "contact not found")
        for field in ("name", "title", "email", "phone", "linkedin", "personal_notes"):
            v = getattr(body, field)
            if v is not None:
                setattr(c, field, v)
        if body.company_id is not None:
            c.company_id = body.company_id or None
        await s.commit()
    await _audit("contact.patch", {"id": contact_id, **body.model_dump(exclude_none=True)},
                 summary=contact_id)
    return {"id": contact_id, "updated": True}


# ---- Companies (used by contact/deal create dropdowns) ------------


@router.get("/companies")
async def list_companies(q: str = "", limit: int = 100):
    from ..db.models import Company
    async with _sm() as s:
        query = select(Company).order_by(Company.name).limit(limit)
        if q:
            query = select(Company).where(Company.name.ilike(f"%{q}%")).limit(limit)
        rows = (await s.execute(query)).scalars().all()
    return {"companies": [{"id": r.id, "name": r.name, "industry": r.industry} for r in rows]}


class CompanyCreate(BaseModel):
    name: str
    industry: str = ""
    website: str = ""


@router.post("/companies")
async def create_company(body: CompanyCreate):
    from ..db.models import Company
    async with _sm() as s:
        c = Company(name=body.name, industry=body.industry, website=body.website)
        s.add(c)
        await s.commit()
        await s.refresh(c)
    await _audit("company.create", body.model_dump(), summary=c.id)
    return {"id": c.id, "name": c.name}


class CompanyPatch(BaseModel):
    name: str | None = None
    industry: str | None = None
    website: str | None = None
    notes: str | None = None


@router.patch("/companies/{company_id}")
async def patch_company(company_id: str, body: CompanyPatch):
    from ..db.models import Company
    async with _sm() as s:
        c = await s.get(Company, company_id)
        if not c:
            raise HTTPException(404, "company not found")
        for f in ("name", "industry", "website", "notes"):
            v = getattr(body, f)
            if v is not None:
                setattr(c, f, v)
        await s.commit()
    await _audit("company.patch", {"id": company_id, **body.model_dump(exclude_none=True)},
                 summary=company_id)
    return {"id": company_id, "updated": True}


@router.get("/companies/{company_id}")
async def company_detail(company_id: str):
    """Account-centric rollup — every deal, contact, bid, plant, and recent
    activity for this company. Powers /companies/[id]."""
    from ..db.models import ActionItem, Bid, Company, Contact, Deal, Meeting, Plant
    async with _sm() as s:
        c = await s.get(Company, company_id)
        if not c:
            raise HTTPException(404, "company not found")
        deals = (await s.execute(
            select(Deal).where(Deal.company_id == company_id).order_by(Deal.updated_at.desc())
        )).scalars().all()
        contacts = (await s.execute(
            select(Contact).where(Contact.company_id == company_id).order_by(Contact.name)
        )).scalars().all()
        bids = (await s.execute(
            select(Bid).where(Bid.company_id == company_id)
            .order_by(Bid.submission_deadline.asc().nullslast())
        )).scalars().all()
        plants = (await s.execute(
            select(Plant).where(Plant.company_id == company_id).order_by(Plant.name)
        )).scalars().all()
        deal_ids = [d.id for d in deals]
        recent_meetings = []
        recent_actions = []
        if deal_ids:
            recent_meetings = (await s.execute(
                select(Meeting).where(Meeting.deal_id.in_(deal_ids))
                .order_by(Meeting.date.desc()).limit(10)
            )).scalars().all()
            recent_actions = (await s.execute(
                select(ActionItem).where(ActionItem.deal_id.in_(deal_ids))
                .order_by(ActionItem.created_at.desc()).limit(15)
            )).scalars().all()

    total_pipeline = sum(d.value_usd or 0 for d in deals if d.stage not in ("closed_won", "closed_lost"))
    won_value = sum(d.value_usd or 0 for d in deals if d.stage == "closed_won")

    return {
        "company": {
            "id": c.id, "name": c.name, "industry": c.industry,
            "website": c.website, "notes": c.notes,
        },
        "stats": {
            "deal_count": len(deals),
            "active_pipeline_value": total_pipeline,
            "won_value": won_value,
            "contact_count": len(contacts),
            "open_bid_count": sum(1 for b in bids if b.stage in ("evaluating", "in_progress")),
            "plant_count": len(plants),
        },
        "plants": [
            {"id": p.id, "name": p.name,
             "site_address": p.site_address or "",
             "site_type": p.site_type or "other"}
            for p in plants
        ],
        "deals": [
            {"id": d.id, "name": d.name, "stage": d.stage,
             "value_usd": d.value_usd or 0,
             "close_date": str(d.close_date) if d.close_date else None,
             "next_step": d.next_step or ""}
            for d in deals
        ],
        "contacts": [
            {"id": ct.id, "name": ct.name, "title": ct.title,
             "email": ct.email, "phone": ct.phone,
             "personal_notes": (ct.personal_notes or "")[:200]}
            for ct in contacts
        ],
        "bids": [
            {"id": b.id, "name": b.name, "stage": b.stage,
             "value_usd": b.value_usd or 0,
             "submission_deadline": b.submission_deadline.isoformat() if b.submission_deadline else None,
             "deal_id": b.deal_id}
            for b in bids
        ],
        "recent_meetings": [
            {"id": m.id, "date": str(m.date), "summary": (m.summary or "")[:200], "deal_id": m.deal_id}
            for m in recent_meetings
        ],
        "recent_actions": [
            {"id": a.id, "description": a.description, "status": a.status,
             "due_date": str(a.due_date) if a.due_date else None, "deal_id": a.deal_id}
            for a in recent_actions
        ],
    }


# ---- Bids (RFPs) — top-level industrial-sales surface --------------


@router.get("/bids")
async def list_bids(stage: str = "", limit: int = 100):
    """All bids, ordered by submission deadline (urgent first). Optional
    stage filter."""
    from ..db.models import Bid, Company, Deal
    async with _sm() as s:
        q = select(Bid).order_by(Bid.submission_deadline.asc().nullslast()).limit(limit)
        if stage:
            if stage not in _BID_STAGES:
                raise HTTPException(400, f"invalid stage: {stage}")
            q = select(Bid).where(Bid.stage == stage).order_by(
                Bid.submission_deadline.asc().nullslast()
            ).limit(limit)
        rows = (await s.execute(q)).scalars().all()
        company_ids = {r.company_id for r in rows if r.company_id}
        deal_ids = {r.deal_id for r in rows if r.deal_id}
        companies = {}
        if company_ids:
            cs = (await s.execute(select(Company).where(Company.id.in_(company_ids)))).scalars().all()
            companies = {c.id: c.name for c in cs}
        deals = {}
        if deal_ids:
            ds = (await s.execute(select(Deal).where(Deal.id.in_(deal_ids)))).scalars().all()
            deals = {d.id: d.name for d in ds}
    return {
        "bids": [
            {
                "id": r.id, "name": r.name, "stage": r.stage,
                "value_usd": r.value_usd or 0,
                "submission_deadline": r.submission_deadline.isoformat() if r.submission_deadline else None,
                "qa_deadline": r.qa_deadline.isoformat() if r.qa_deadline else None,
                "company_id": r.company_id, "company": companies.get(r.company_id, ""),
                "deal_id": r.deal_id, "deal": deals.get(r.deal_id, ""),
                "rfp_url": r.rfp_url or "",
            }
            for r in rows
        ],
    }


@router.get("/bids/{bid_id}")
async def bid_detail(bid_id: str):
    from ..db.models import Bid, Company, Deal
    async with _sm() as s:
        b = await s.get(Bid, bid_id)
        if not b:
            raise HTTPException(404, "bid not found")
        company_name = ""
        deal_name = ""
        if b.company_id:
            c = await s.get(Company, b.company_id)
            if c:
                company_name = c.name
        if b.deal_id:
            d = await s.get(Deal, b.deal_id)
            if d:
                deal_name = d.name
    plant_name = ""
    if b.plant_id:
        from ..db.models import Plant
        async with _sm() as s:
            pl = await s.get(Plant, b.plant_id)
            if pl:
                plant_name = pl.name
    return {
        "bid": {
            "id": b.id, "name": b.name, "stage": b.stage,
            "value_usd": b.value_usd or 0,
            "submission_deadline": b.submission_deadline.isoformat() if b.submission_deadline else None,
            "qa_deadline": b.qa_deadline.isoformat() if b.qa_deadline else None,
            "rfp_url": b.rfp_url or "",
            "deliverables": b.deliverables or "",
            "notes": b.notes or "",
            "company_id": b.company_id, "company": company_name,
            "deal_id": b.deal_id, "deal": deal_name,
            "plant_id": b.plant_id, "plant": plant_name,
        },
    }


def _parse_dt(s: str | None):
    """Parse an ISO datetime or YYYY-MM-DD into UTC-aware datetime."""
    if not s:
        return None
    from datetime import datetime as _dt, timezone as _tz
    try:
        # ISO with timezone
        d = _dt.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        # plain date
        d = _dt.strptime(s, "%Y-%m-%d")
    if d.tzinfo is None:
        d = d.replace(tzinfo=_tz.utc)
    return d


class BidCreate(BaseModel):
    name: str
    company_id: str | None = None
    deal_id: str | None = None
    plant_id: str | None = None
    stage: str = "evaluating"
    value_usd: float = 0.0
    submission_deadline: str | None = None
    qa_deadline: str | None = None
    rfp_url: str = ""
    deliverables: str = ""
    notes: str = ""


@router.post("/bids")
async def create_bid(body: BidCreate):
    from ..db.models import Bid
    if body.stage not in _BID_STAGES:
        raise HTTPException(400, f"invalid stage: {body.stage}")
    async with _sm() as s:
        b = Bid(
            name=body.name,
            company_id=body.company_id or None,
            deal_id=body.deal_id or None,
            plant_id=body.plant_id or None,
            stage=body.stage,
            value_usd=body.value_usd,
            submission_deadline=_parse_dt(body.submission_deadline),
            qa_deadline=_parse_dt(body.qa_deadline),
            rfp_url=body.rfp_url,
            deliverables=body.deliverables,
            notes=body.notes,
        )
        s.add(b)
        await s.commit()
        await s.refresh(b)
    await _audit("bid.create", body.model_dump(), summary=f"{b.id} {b.name}")
    return {"id": b.id, "name": b.name, "stage": b.stage}


class BidPatch(BaseModel):
    name: str | None = None
    stage: str | None = None
    value_usd: float | None = None
    submission_deadline: str | None = None
    qa_deadline: str | None = None
    rfp_url: str | None = None
    deliverables: str | None = None
    notes: str | None = None
    company_id: str | None = None
    deal_id: str | None = None
    plant_id: str | None = None


@router.patch("/bids/{bid_id}")
async def patch_bid(bid_id: str, body: BidPatch):
    from ..db.models import Bid
    async with _sm() as s:
        b = await s.get(Bid, bid_id)
        if not b:
            raise HTTPException(404, "bid not found")
        if body.stage is not None:
            if body.stage not in _BID_STAGES:
                raise HTTPException(400, f"invalid stage: {body.stage}")
            b.stage = body.stage
        for f in ("name", "value_usd", "rfp_url", "deliverables", "notes"):
            v = getattr(body, f)
            if v is not None:
                setattr(b, f, v)
        if body.submission_deadline is not None:
            b.submission_deadline = _parse_dt(body.submission_deadline)
        if body.qa_deadline is not None:
            b.qa_deadline = _parse_dt(body.qa_deadline)
        if body.company_id is not None:
            b.company_id = body.company_id or None
        if body.deal_id is not None:
            b.deal_id = body.deal_id or None
        if body.plant_id is not None:
            b.plant_id = body.plant_id or None
        await s.commit()
    await _audit("bid.patch", {"id": bid_id, **body.model_dump(exclude_none=True)}, summary=bid_id)
    return {"id": bid_id, "updated": True}


# ---- Meetings (CRUD parity — previously read-only) ---------------


class MeetingCreate(BaseModel):
    deal_id: str | None = None
    date: str | None = None              # ISO datetime; defaults to now
    attendees: str = ""
    summary: str = ""
    decisions: str = ""
    transcript: str = ""


@router.post("/deals/{deal_id}/meetings")
async def create_meeting(deal_id: str, body: MeetingCreate):
    from ..db.models import Deal, Meeting
    async with _sm() as s:
        d = await s.get(Deal, deal_id)
        if not d:
            raise HTTPException(404, "deal not found")
        kwargs: dict = {
            "deal_id": deal_id,
            "attendees": body.attendees,
            "summary": body.summary,
            "decisions": body.decisions,
            "transcript": body.transcript,
        }
        parsed_date = _parse_dt(body.date)
        if parsed_date is not None:
            kwargs["date"] = parsed_date
        m = Meeting(**kwargs)
        s.add(m)
        await s.commit()
        await s.refresh(m)
    await _audit("meeting.create", {"deal_id": deal_id, **body.model_dump()},
                 summary=f"{m.id}")
    return {"id": m.id, "date": str(m.date)}


class MeetingPatch(BaseModel):
    date: str | None = None
    attendees: str | None = None
    summary: str | None = None
    decisions: str | None = None
    transcript: str | None = None


@router.patch("/meetings/{meeting_id}")
async def patch_meeting(meeting_id: str, body: MeetingPatch):
    from ..db.models import Meeting
    async with _sm() as s:
        m = await s.get(Meeting, meeting_id)
        if not m:
            raise HTTPException(404, "meeting not found")
        if body.date is not None:
            m.date = _parse_dt(body.date)
        for f in ("attendees", "summary", "decisions", "transcript"):
            v = getattr(body, f)
            if v is not None:
                setattr(m, f, v)
        await s.commit()
    await _audit("meeting.patch", {"id": meeting_id, **body.model_dump(exclude_none=True)},
                 summary=meeting_id)
    return {"id": meeting_id, "updated": True}


# ---- DELETE — universal CRUD parity ---------------------------------


async def _delete_by_id(model_cls, entity_id: str, op: str, label: str):
    async with _sm() as s:
        obj = await s.get(model_cls, entity_id)
        if not obj:
            raise HTTPException(404, f"{label} not found")
        await s.delete(obj)
        await s.commit()
    await _audit(op, {"id": entity_id}, summary=entity_id)
    return {"id": entity_id, "deleted": True}


@router.delete("/deals/{deal_id}")
async def delete_deal(deal_id: str):
    from ..db.models import Deal
    return await _delete_by_id(Deal, deal_id, "deal.delete", "deal")


@router.delete("/contacts/{contact_id}")
async def delete_contact(contact_id: str):
    from ..db.models import Contact
    return await _delete_by_id(Contact, contact_id, "contact.delete", "contact")


@router.delete("/companies/{company_id}")
async def delete_company(company_id: str):
    from ..db.models import Company
    return await _delete_by_id(Company, company_id, "company.delete", "company")


@router.delete("/actions/{action_id}")
async def delete_action(action_id: str):
    from ..db.models import ActionItem
    return await _delete_by_id(ActionItem, action_id, "action.delete", "action")


@router.delete("/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str):
    from ..db.models import Meeting
    return await _delete_by_id(Meeting, meeting_id, "meeting.delete", "meeting")


@router.delete("/bids/{bid_id}")
async def delete_bid(bid_id: str):
    from ..db.models import Bid
    return await _delete_by_id(Bid, bid_id, "bid.delete", "bid")


# =====================================================================
# Reminders + Inbox — turn the activity feed from a passive log into
# an actionable surface. Bot-created reminders fire to Telegram, but
# the user also needs to dismiss/resolve/snooze them from the UI.
# =====================================================================


class ReminderPatch(BaseModel):
    status: str | None = None
    message: str | None = None
    trigger_at: str | None = None


@router.patch("/reminders/{reminder_id}")
async def patch_reminder(reminder_id: str, body: ReminderPatch):
    from ..db.models import Reminder
    async with _sm() as s:
        r = await s.get(Reminder, reminder_id)
        if not r:
            raise HTTPException(404, "reminder not found")
        if body.status is not None:
            if body.status not in _REMINDER_STATUSES:
                raise HTTPException(400, f"invalid status: {body.status}")
            r.status = body.status
        if body.message is not None:
            r.message = body.message
        if body.trigger_at is not None:
            r.trigger_at = _parse_dt(body.trigger_at)
        await s.commit()
    await _audit("reminder.patch", {"id": reminder_id, **body.model_dump(exclude_none=True)},
                 summary=reminder_id)
    return {"id": reminder_id, "updated": True}


class ReminderSnooze(BaseModel):
    hours: float = 24.0


@router.post("/reminders/{reminder_id}/snooze")
async def snooze_reminder(reminder_id: str, body: ReminderSnooze):
    """Snooze pushes trigger_at forward and resets status to pending so the
    ReminderService re-fires it at the new time. Cleaner than a separate
    'snoozed' status because it reuses the existing pending pickup loop."""
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    from ..db.models import Reminder
    async with _sm() as s:
        r = await s.get(Reminder, reminder_id)
        if not r:
            raise HTTPException(404, "reminder not found")
        r.trigger_at = _dt.now(_tz.utc) + _td(hours=body.hours)
        r.status = "pending"
        r.sent_at = None
        await s.commit()
    await _audit("reminder.snooze", {"id": reminder_id, "hours": body.hours},
                 summary=f"{reminder_id} +{body.hours}h")
    return {"id": reminder_id, "trigger_at": str(r.trigger_at)}


@router.delete("/reminders/{reminder_id}")
async def delete_reminder(reminder_id: str):
    from ..db.models import Reminder
    return await _delete_by_id(Reminder, reminder_id, "reminder.delete", "reminder")


# ---- Inbox — actionable items, deduped --------------------------------


def _dedup_key(deal_id: str | None, message: str) -> str:
    """Group reminders by deal + message-shape so the watcher firing the
    same alert twice in the same window collapses to one inbox row."""
    import hashlib
    base = f"{deal_id or ''}|{(message or '').strip().lower()[:100]}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]


@router.get("/inbox")
async def inbox(limit: int = 50):
    """One stream of things that need the user's attention right now:

    - Reminders the bot has *sent* but the user hasn't yet acknowledged
    - Reminders pending in the next 7 days (so user can plan)
    - Open pending_actions (action-gate approval queue)
    - Open action_items due within 7 days

    Deduped by deal_id + message-hash within a 24h window.
    """
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    from ..db.models import ActionItem, Deal, PendingAction, Reminder

    now = _dt.now(_tz.utc)
    horizon = now + _td(days=7)

    async with _sm() as s:
        # Reminders: sent (awaiting ack) OR pending (upcoming) OR snoozed-back
        reminders = (await s.execute(
            select(Reminder).where(
                Reminder.status.in_(("pending", "sent")),
                Reminder.trigger_at <= horizon,
            ).order_by(Reminder.trigger_at.asc())
        )).scalars().all()

        # Pending agent-action approvals
        pending_acts = (await s.execute(
            select(PendingAction).where(PendingAction.status == "pending")
            .order_by(PendingAction.created_at.desc())
        )).scalars().all()

        # Action items due in the next 7 days
        actions_due = (await s.execute(
            select(ActionItem).where(
                ActionItem.status == "open",
                ActionItem.due_date != None,  # noqa: E711
                ActionItem.due_date <= horizon.date(),
            ).order_by(ActionItem.due_date.asc())
        )).scalars().all()

        # Resolve deal names for context
        deal_ids = (
            {r.related_deal_id for r in reminders if r.related_deal_id}
            | {p.related_deal_id for p in pending_acts if p.related_deal_id}
            | {a.deal_id for a in actions_due if a.deal_id}
        )
        deals = {}
        if deal_ids:
            ds = (await s.execute(select(Deal).where(Deal.id.in_(deal_ids)))).scalars().all()
            deals = {d.id: d.name for d in ds}

    items: list[dict] = []
    seen_keys: dict[str, dict] = {}

    for r in reminders:
        # Dedup: collapse near-identical reminders for the same deal — the
        # Pipeline Watcher firing the same alert twice in 24h shouldn't
        # mean two inbox rows the user has to dismiss separately.
        key = _dedup_key(r.related_deal_id, r.message)
        existing = seen_keys.get(key)
        if existing:
            existing["dup_count"] = existing.get("dup_count", 1) + 1
            continue
        item = {
            "kind": "reminder",
            "id": r.id,
            "title": r.message[:200],
            "status": r.status,
            "trigger_at": str(r.trigger_at),
            "is_overdue": r.status == "sent",       # bot fired, user hasn't acked
            "deal_id": r.related_deal_id,
            "deal_name": deals.get(r.related_deal_id) if r.related_deal_id else None,
            "kind_detail": r.kind,
            "dup_count": 1,
        }
        items.append(item)
        seen_keys[key] = item

    for p in pending_acts:
        items.append({
            "kind": "pending_action",
            "id": p.id,
            "title": p.summary or f"Approve {p.tool_name}",
            "tool_name": p.tool_name,
            "status": p.status,
            "created_at": str(p.created_at),
            "expires_at": str(p.expires_at) if p.expires_at else None,
            "deal_id": p.related_deal_id,
            "deal_name": deals.get(p.related_deal_id) if p.related_deal_id else None,
        })

    for a in actions_due:
        items.append({
            "kind": "action_item",
            "id": a.id,
            "title": a.description[:200],
            "status": a.status,
            "due_date": str(a.due_date) if a.due_date else None,
            "source": a.source,
            "deal_id": a.deal_id,
            "deal_name": deals.get(a.deal_id) if a.deal_id else None,
        })

    # Sort: overdue/sent reminders first, then pending actions, then due-soon
    # action items, then upcoming reminders. Inside each group, soonest first.
    def _priority(it: dict) -> tuple[int, str]:
        kind = it["kind"]
        if kind == "reminder" and it.get("is_overdue"):
            return (0, it.get("trigger_at", ""))
        if kind == "pending_action":
            return (1, it.get("created_at", ""))
        if kind == "action_item":
            return (2, it.get("due_date", "9999"))
        return (3, it.get("trigger_at", ""))
    items.sort(key=_priority)

    return {"items": items[:limit], "counts": {
        "reminders_overdue": sum(1 for i in items if i["kind"] == "reminder" and i.get("is_overdue")),
        "pending_actions": sum(1 for i in items if i["kind"] == "pending_action"),
        "actions_due": sum(1 for i in items if i["kind"] == "action_item"),
        "reminders_upcoming": sum(1 for i in items if i["kind"] == "reminder" and not i.get("is_overdue")),
    }}


# =====================================================================
# PR3 — industrial data model: Plants, Specs, Compliance Matrix
# =====================================================================


# ---- Plants -------------------------------------------------------


@router.get("/plants")
async def list_plants(company_id: str = "", q: str = "", limit: int = 100):
    from ..db.models import Company, Plant
    async with _sm() as s:
        q_stmt = select(Plant).order_by(Plant.name).limit(limit)
        if company_id:
            q_stmt = select(Plant).where(Plant.company_id == company_id).order_by(Plant.name)
        elif q:
            q_stmt = select(Plant).where(Plant.name.ilike(f"%{q}%")).limit(limit)
        rows = (await s.execute(q_stmt)).scalars().all()
        company_ids = {r.company_id for r in rows if r.company_id}
        companies = {}
        if company_ids:
            cs = (await s.execute(select(Company).where(Company.id.in_(company_ids)))).scalars().all()
            companies = {c.id: c.name for c in cs}
    return {
        "plants": [
            {
                "id": r.id, "name": r.name,
                "company_id": r.company_id, "company": companies.get(r.company_id, ""),
                "site_address": r.site_address or "",
                "site_type": r.site_type or "other",
            }
            for r in rows
        ],
    }


@router.get("/plants/{plant_id}")
async def plant_detail(plant_id: str):
    from ..db.models import Asset, Bid, Company, Contact, Deal, Plant, ServiceContract
    async with _sm() as s:
        p = await s.get(Plant, plant_id)
        if not p:
            raise HTTPException(404, "plant not found")
        company = None
        if p.company_id:
            c = await s.get(Company, p.company_id)
            if c:
                company = {"id": c.id, "name": c.name}
        deals = (await s.execute(
            select(Deal).where(Deal.plant_id == plant_id).order_by(Deal.updated_at.desc())
        )).scalars().all()
        bids = (await s.execute(
            select(Bid).where(Bid.plant_id == plant_id)
            .order_by(Bid.submission_deadline.asc().nullslast())
        )).scalars().all()
        assets = (await s.execute(
            select(Asset).where(Asset.plant_id == plant_id)
            .order_by(Asset.manufacturer, Asset.name)
        )).scalars().all()
        contracts = (await s.execute(
            select(ServiceContract).where(ServiceContract.plant_id == plant_id)
            .order_by(ServiceContract.renewal_date.asc().nullslast())
        )).scalars().all()
        manager = None
        if p.plant_manager_contact_id:
            mc = await s.get(Contact, p.plant_manager_contact_id)
            if mc:
                manager = {"id": mc.id, "name": mc.name, "title": mc.title, "email": mc.email}

    return {
        "plant": {
            "id": p.id, "name": p.name,
            "site_address": p.site_address or "",
            "site_type": p.site_type or "other",
            "standards_notes": p.standards_notes or "",
            "notes": p.notes or "",
            "company": company,
            "plant_manager": manager,
        },
        "deals": [
            {"id": d.id, "name": d.name, "stage": d.stage,
             "value_usd": d.value_usd or 0,
             "next_step": d.next_step or "",
             "close_date": str(d.close_date) if d.close_date else None}
            for d in deals
        ],
        "bids": [
            {"id": b.id, "name": b.name, "stage": b.stage,
             "value_usd": b.value_usd or 0,
             "submission_deadline": b.submission_deadline.isoformat() if b.submission_deadline else None}
            for b in bids
        ],
        "assets": [
            {"id": a.id, "name": a.name,
             "manufacturer": a.manufacturer, "model": a.model,
             "asset_type": a.asset_type, "vendor": a.vendor,
             "quantity": a.quantity,
             "end_of_life_date": str(a.end_of_life_date) if a.end_of_life_date else None}
            for a in assets
        ],
        "contracts": [
            {"id": c.id, "name": c.name,
             "contract_type": c.contract_type,
             "value_usd_annual": c.value_usd_annual or 0,
             "renewal_date": str(c.renewal_date) if c.renewal_date else None,
             "status": c.status}
            for c in contracts
        ],
    }


class PlantCreate(BaseModel):
    name: str
    company_id: str
    site_address: str = ""
    site_type: str = "other"
    plant_manager_contact_id: str | None = None
    standards_notes: str = ""
    notes: str = ""


@router.post("/plants")
async def create_plant(body: PlantCreate):
    from ..db.models import Company, Plant
    if body.site_type not in _PLANT_SITE_TYPES:
        raise HTTPException(400, f"invalid site_type: {body.site_type}")
    async with _sm() as s:
        if not await s.get(Company, body.company_id):
            raise HTTPException(404, "company not found")
        p = Plant(
            name=body.name,
            company_id=body.company_id,
            site_address=body.site_address,
            site_type=body.site_type,
            plant_manager_contact_id=body.plant_manager_contact_id or None,
            standards_notes=body.standards_notes,
            notes=body.notes,
        )
        s.add(p)
        await s.commit()
        await s.refresh(p)
    await _audit("plant.create", body.model_dump(), summary=f"{p.id} {p.name}")
    return {"id": p.id, "name": p.name}


class PlantPatch(BaseModel):
    name: str | None = None
    site_address: str | None = None
    site_type: str | None = None
    plant_manager_contact_id: str | None = None
    standards_notes: str | None = None
    notes: str | None = None
    company_id: str | None = None


@router.patch("/plants/{plant_id}")
async def patch_plant(plant_id: str, body: PlantPatch):
    from ..db.models import Plant
    async with _sm() as s:
        p = await s.get(Plant, plant_id)
        if not p:
            raise HTTPException(404, "plant not found")
        if body.site_type is not None:
            if body.site_type not in _PLANT_SITE_TYPES:
                raise HTTPException(400, f"invalid site_type: {body.site_type}")
            p.site_type = body.site_type
        for f in ("name", "site_address", "standards_notes", "notes"):
            v = getattr(body, f)
            if v is not None:
                setattr(p, f, v)
        if body.plant_manager_contact_id is not None:
            p.plant_manager_contact_id = body.plant_manager_contact_id or None
        if body.company_id is not None:
            p.company_id = body.company_id
        await s.commit()
    await _audit("plant.patch", {"id": plant_id, **body.model_dump(exclude_none=True)},
                 summary=plant_id)
    return {"id": plant_id, "updated": True}


@router.delete("/plants/{plant_id}")
async def delete_plant(plant_id: str):
    from ..db.models import Plant
    return await _delete_by_id(Plant, plant_id, "plant.delete", "plant")


# ---- Specs (standards / certifications library) -------------------


@router.get("/specs")
async def list_specs(family: str = "", q: str = "", limit: int = 200):
    from ..db.models import Spec
    async with _sm() as s:
        q_stmt = select(Spec).order_by(Spec.family, Spec.code).limit(limit)
        if family:
            q_stmt = select(Spec).where(Spec.family == family).order_by(Spec.code)
        elif q:
            q_stmt = select(Spec).where(
                Spec.code.ilike(f"%{q}%") | Spec.name.ilike(f"%{q}%")
            ).order_by(Spec.code).limit(limit)
        rows = (await s.execute(q_stmt)).scalars().all()
    return {
        "specs": [
            {"id": r.id, "code": r.code, "name": r.name,
             "family": r.family, "scope": r.scope or "",
             "evidence_required": r.evidence_required or ""}
            for r in rows
        ],
    }


class SpecCreate(BaseModel):
    code: str
    name: str
    family: str = "other"
    scope: str = ""
    evidence_required: str = ""


@router.post("/specs")
async def create_spec(body: SpecCreate):
    from sqlalchemy.exc import IntegrityError
    from ..db.models import Spec
    if body.family not in _SPEC_FAMILIES:
        raise HTTPException(400, f"invalid family: {body.family}")
    async with _sm() as s:
        sp = Spec(
            code=body.code, name=body.name, family=body.family,
            scope=body.scope, evidence_required=body.evidence_required,
        )
        s.add(sp)
        try:
            await s.commit()
        except IntegrityError:
            await s.rollback()
            raise HTTPException(409, f"spec code already exists: {body.code}")
        await s.refresh(sp)
    await _audit("spec.create", body.model_dump(), summary=sp.id)
    return {"id": sp.id, "code": sp.code}


class SpecPatch(BaseModel):
    name: str | None = None
    family: str | None = None
    scope: str | None = None
    evidence_required: str | None = None


@router.patch("/specs/{spec_id}")
async def patch_spec(spec_id: str, body: SpecPatch):
    from ..db.models import Spec
    async with _sm() as s:
        sp = await s.get(Spec, spec_id)
        if not sp:
            raise HTTPException(404, "spec not found")
        if body.family is not None:
            if body.family not in _SPEC_FAMILIES:
                raise HTTPException(400, f"invalid family: {body.family}")
            sp.family = body.family
        for f in ("name", "scope", "evidence_required"):
            v = getattr(body, f)
            if v is not None:
                setattr(sp, f, v)
        await s.commit()
    await _audit("spec.patch", {"id": spec_id, **body.model_dump(exclude_none=True)},
                 summary=spec_id)
    return {"id": spec_id, "updated": True}


@router.delete("/specs/{spec_id}")
async def delete_spec(spec_id: str):
    from ..db.models import Spec
    return await _delete_by_id(Spec, spec_id, "spec.delete", "spec")


# ---- Compliance Matrix --------------------------------------------


@router.get("/bids/{bid_id}/compliance")
async def list_compliance(bid_id: str):
    from ..db.models import Bid, ComplianceMatrixItem
    async with _sm() as s:
        if not await s.get(Bid, bid_id):
            raise HTTPException(404, "bid not found")
        rows = (await s.execute(
            select(ComplianceMatrixItem).where(ComplianceMatrixItem.bid_id == bid_id)
            .order_by(ComplianceMatrixItem.sort_order, ComplianceMatrixItem.created_at)
        )).scalars().all()

    summary = {s_: 0 for s_ in _COMPLIANCE_STATUSES}
    for r in rows:
        summary[r.status or "unanswered"] = summary.get(r.status or "unanswered", 0) + 1

    return {
        "items": [
            {"id": r.id, "clause_section": r.clause_section or "",
             "clause_text": r.clause_text, "our_response": r.our_response or "",
             "status": r.status or "unanswered",
             "spec_ids": [sid for sid in (r.spec_ids or "").split(",") if sid],
             "notes": r.notes or "", "sort_order": r.sort_order or 0}
            for r in rows
        ],
        "summary": summary,
        "total": len(rows),
    }


class ComplianceCreate(BaseModel):
    clause_section: str = ""
    clause_text: str
    our_response: str = ""
    status: str = "unanswered"
    spec_ids: list[str] = []
    notes: str = ""
    sort_order: int = 0


@router.post("/bids/{bid_id}/compliance")
async def create_compliance(bid_id: str, body: ComplianceCreate):
    from ..db.models import Bid, ComplianceMatrixItem
    if body.status not in _COMPLIANCE_STATUSES:
        raise HTTPException(400, f"invalid status: {body.status}")
    async with _sm() as s:
        if not await s.get(Bid, bid_id):
            raise HTTPException(404, "bid not found")
        item = ComplianceMatrixItem(
            bid_id=bid_id,
            clause_section=body.clause_section,
            clause_text=body.clause_text,
            our_response=body.our_response,
            status=body.status,
            spec_ids=",".join(body.spec_ids),
            notes=body.notes,
            sort_order=body.sort_order,
        )
        s.add(item)
        await s.commit()
        await s.refresh(item)
    await _audit("compliance.create", {"bid_id": bid_id, **body.model_dump()}, summary=item.id)
    return {"id": item.id}


class ComplianceBulk(BaseModel):
    """Bulk-paste mode: one clause per line. Lines starting with a section
    marker like "4.2.1 " split into clause_section + clause_text."""
    text: str


@router.post("/bids/{bid_id}/compliance/bulk")
async def bulk_compliance(bid_id: str, body: ComplianceBulk):
    import re
    from ..db.models import Bid, ComplianceMatrixItem
    async with _sm() as s:
        if not await s.get(Bid, bid_id):
            raise HTTPException(404, "bid not found")
        lines = [ln.strip() for ln in body.text.splitlines() if ln.strip()]
        # Match leading "1.2.3" or "Section 4.2.1" style markers
        section_re = re.compile(r"^(?:Section\s+)?([\w\-\.]+)\s+(.*)$")
        created = 0
        for i, raw in enumerate(lines):
            m = section_re.match(raw)
            if m and any(ch.isdigit() for ch in m.group(1)):
                section = m.group(1)
                text = m.group(2)
            else:
                section, text = "", raw
            s.add(ComplianceMatrixItem(
                bid_id=bid_id,
                clause_section=section,
                clause_text=text,
                status="unanswered",
                sort_order=i,
            ))
            created += 1
        await s.commit()
    await _audit("compliance.bulk", {"bid_id": bid_id, "count": created}, summary=f"{created} clauses")
    return {"created": created}


class CompliancePatch(BaseModel):
    clause_section: str | None = None
    clause_text: str | None = None
    our_response: str | None = None
    status: str | None = None
    spec_ids: list[str] | None = None
    notes: str | None = None
    sort_order: int | None = None


@router.patch("/compliance/{item_id}")
async def patch_compliance(item_id: str, body: CompliancePatch):
    from ..db.models import ComplianceMatrixItem
    async with _sm() as s:
        item = await s.get(ComplianceMatrixItem, item_id)
        if not item:
            raise HTTPException(404, "compliance item not found")
        if body.status is not None:
            if body.status not in _COMPLIANCE_STATUSES:
                raise HTTPException(400, f"invalid status: {body.status}")
            item.status = body.status
        for f in ("clause_section", "clause_text", "our_response", "notes"):
            v = getattr(body, f)
            if v is not None:
                setattr(item, f, v)
        if body.spec_ids is not None:
            item.spec_ids = ",".join(body.spec_ids)
        if body.sort_order is not None:
            item.sort_order = body.sort_order
        await s.commit()
    await _audit("compliance.patch", {"id": item_id, **body.model_dump(exclude_none=True)},
                 summary=item_id)
    return {"id": item_id, "updated": True}


@router.delete("/compliance/{item_id}")
async def delete_compliance(item_id: str):
    from ..db.models import ComplianceMatrixItem
    return await _delete_by_id(ComplianceMatrixItem, item_id, "compliance.delete", "compliance item")


# =====================================================================
# PR4 — Chat history for the web chat panel.
# The POST /chat endpoint already lives in main.py; this exposes recent
# conversation rows so the dashboard can render the same back-and-forth
# the user would see in Telegram for a given session.
# =====================================================================


# =====================================================================
# PR5 — Forecast + champion strength + AI MEDDIC suggester
# Deterministic, explainable scoring so reps trust the bucket placement.
# Reps mistrust black-box AI forecasts — they want to see the rationale.
# =====================================================================


def _recency_decay(last_touch_dt) -> float:
    """0.0 (cold) to 1.0 (today). Industrial cycles run long, so we
    don't punish a deal that's been quiet for 2 weeks the way a SaaS
    forecast would. Scale: today=1.0, 7d=1.0, 30d=0.7, 60d=0.4, 90d+=0.1."""
    if not last_touch_dt:
        return 0.2  # we don't know — treat as warmish, don't punish hard
    from datetime import datetime as _dt, timezone as _tz
    now = _dt.now(_tz.utc)
    if last_touch_dt.tzinfo is None:
        last_touch_dt = last_touch_dt.replace(tzinfo=_tz.utc)
    days = max(0, (now - last_touch_dt).days)
    if days <= 7: return 1.0
    if days <= 30: return 0.7
    if days <= 60: return 0.4
    if days <= 90: return 0.2
    return 0.1


def _sentiment_w(s: str) -> float:
    return {"supportive": 1.0, "neutral": 0.5, "opposed": 0.0, "unknown": 0.3}.get(s, 0.3)


def _influence_w(i: str) -> float:
    return {"high": 1.0, "medium": 0.6, "low": 0.3}.get(i, 0.5)


def _champion_strength(champion_stake, champion_contact) -> tuple[int, str]:
    """0-100 + a one-line explanation. The single most predictive
    variable for industrial deals — research is clear that 9/10 deals
    with a 3/10 champion are 3/10 deals."""
    if not champion_stake:
        return 0, "no champion mapped"
    s = _sentiment_w(champion_stake.sentiment)
    i = _influence_w(champion_stake.influence)
    last_touch = champion_contact.last_touch if champion_contact else None
    r = _recency_decay(last_touch)
    score = round(s * i * r * 100)
    bits = []
    bits.append(champion_stake.sentiment)
    bits.append(f"{champion_stake.influence} influence")
    if last_touch:
        from datetime import datetime as _dt, timezone as _tz
        now = _dt.now(_tz.utc)
        lt = last_touch if last_touch.tzinfo else last_touch.replace(tzinfo=_tz.utc)
        days = max(0, (now - lt).days)
        bits.append(f"last touch {days}d ago")
    else:
        bits.append("no last-touch on file")
    return score, ", ".join(bits)


def _meddic_fill(deal) -> tuple[int, list[str]]:
    """Returns (percent_filled, missing_field_labels)."""
    fields = [
        ("metrics", deal.metrics),
        ("economic buyer", deal.economic_buyer_id),
        ("champion", deal.champion_id),
        ("decision criteria", deal.decision_criteria),
        ("decision process", deal.decision_process),
        ("paper process", deal.paper_process),
        ("pain", deal.pain),
    ]
    filled = sum(1 for _, v in fields if v)
    missing = [name for name, v in fields if not v]
    return round(100 * filled / len(fields)), missing


_STAGE_BASE_SLIP = {
    "prospect": 90, "qualified": 70, "proposal": 40, "negotiation": 15,
    "closed_won": 0, "closed_lost": 0,
}


def _slip_risk(deal, meddic_pct: int, champion: int) -> int:
    """Probability deal won't close THIS QUARTER. Distinct from loss
    risk — industrial deals slip a fiscal quarter routinely without
    being lost."""
    base = _STAGE_BASE_SLIP.get(deal.stage or "prospect", 90)
    if meddic_pct < 50:
        base += 20
    if champion < 40:
        base += 15
    if not deal.close_date:
        base += 10
    else:
        from datetime import date as _date
        days_to_close = (deal.close_date - _date.today()).days
        if 0 < days_to_close <= 30 and (deal.stage or "") in ("prospect", "qualified"):
            base += 25
    return max(0, min(100, base))


def _bucket(deal, meddic_pct: int, champion: int) -> tuple[str, list[str]]:
    """Commit / Best Case / Pipeline + top-3 reasons."""
    reasons: list[str] = []
    stage = deal.stage or "prospect"
    if stage in ("negotiation",) and meddic_pct >= 70 and champion >= 60:
        reasons.append(f"in negotiation with strong champion ({champion}/100)")
        if deal.paper_process:
            reasons.append("paper process documented")
        return "commit", reasons[:3]
    if stage in ("proposal", "negotiation") and meddic_pct >= 50 and champion >= 40:
        reasons.append(f"stage: {stage}")
        reasons.append(f"MEDDIC {meddic_pct}%")
        if champion >= 60:
            reasons.append(f"champion {champion}/100")
        return "best_case", reasons[:3]
    # Pipeline default
    if meddic_pct < 50:
        reasons.append(f"MEDDIC only {meddic_pct}% complete")
    if champion < 40:
        reasons.append(f"champion weak ({champion}/100)" if champion else "no champion mapped")
    if stage in ("prospect", "qualified"):
        reasons.append(f"stage: {stage}")
    return "pipeline", reasons[:3]


@router.get("/forecast")
async def forecast():
    """Active deals bucketed Commit / Best Case / Pipeline with rationale.

    Two distinct probabilities per deal — slip risk (won't close this
    quarter) and champion strength (predictor of win-eventually). Reps
    trust forecasts they can see the math behind."""
    from ..db.models import Contact, Deal, DealStakeholder, Meeting

    async with _sm() as s:
        deals = (await s.execute(
            select(Deal).where(Deal.stage.notin_(("closed_won", "closed_lost")))
            .order_by(Deal.value_usd.desc().nullslast())
        )).scalars().all()

        # Bulk-load related data
        deal_ids = [d.id for d in deals]
        stakeholders_by_deal: dict[str, list] = {}
        if deal_ids:
            stakeholder_rows = (await s.execute(
                select(DealStakeholder).where(DealStakeholder.deal_id.in_(deal_ids))
            )).scalars().all()
            for st in stakeholder_rows:
                stakeholders_by_deal.setdefault(st.deal_id, []).append(st)

        contact_ids = {d.champion_id for d in deals if d.champion_id}
        contacts = {}
        if contact_ids:
            crows = (await s.execute(
                select(Contact).where(Contact.id.in_(contact_ids))
            )).scalars().all()
            contacts = {c.id: c for c in crows}

    buckets = {"commit": [], "best_case": [], "pipeline": []}
    for d in deals:
        meddic_pct, missing = _meddic_fill(d)
        # Find the champion stakeholder for this deal
        champ_stake = next(
            (st for st in stakeholders_by_deal.get(d.id, []) if st.role == "champion"),
            None,
        )
        champ_contact = contacts.get(d.champion_id) if d.champion_id else None
        champion_score, champion_detail = _champion_strength(champ_stake, champ_contact)
        bucket, reasons = _bucket(d, meddic_pct, champion_score)
        slip = _slip_risk(d, meddic_pct, champion_score)

        buckets[bucket].append({
            "id": d.id, "name": d.name, "stage": d.stage,
            "value_usd": d.value_usd or 0,
            "close_date": str(d.close_date) if d.close_date else None,
            "meddic_pct": meddic_pct,
            "meddic_missing": missing[:4],
            "champion_score": champion_score,
            "champion_detail": champion_detail,
            "slip_risk": slip,
            "reasons": reasons,
        })

    totals = {
        b: {
            "count": len(items),
            "value": sum(it["value_usd"] for it in items),
        }
        for b, items in buckets.items()
    }
    return {"buckets": buckets, "totals": totals}


# ---- Champion strength on a single deal (used by deal detail) -----


@router.get("/deals/{deal_id}/health")
async def deal_health(deal_id: str):
    """Per-deal scorecard: MEDDIC fill, champion strength, slip risk,
    forecast bucket, and rationale. Powers the deal-page header chip."""
    from ..db.models import Contact, Deal, DealStakeholder

    async with _sm() as s:
        d = await s.get(Deal, deal_id)
        if not d:
            raise HTTPException(404, "deal not found")
        sts = (await s.execute(
            select(DealStakeholder).where(DealStakeholder.deal_id == deal_id)
        )).scalars().all()
        champ_stake = next((st for st in sts if st.role == "champion"), None)
        champ_contact = None
        if d.champion_id:
            champ_contact = await s.get(Contact, d.champion_id)

    meddic_pct, missing = _meddic_fill(d)
    champion_score, champion_detail = _champion_strength(champ_stake, champ_contact)
    bucket, reasons = _bucket(d, meddic_pct, champion_score)
    slip = _slip_risk(d, meddic_pct, champion_score)
    return {
        "deal_id": deal_id,
        "meddic_pct": meddic_pct,
        "meddic_missing": missing,
        "champion_score": champion_score,
        "champion_detail": champion_detail,
        "slip_risk": slip,
        "forecast_bucket": bucket,
        "reasons": reasons,
    }


# ---- AI MEDDIC suggester from a meeting ---------------------------


_MEDDIC_PROMPT = """You are extracting MEDDIC sales-qualification field updates from a meeting note.

Given the existing deal context and the new meeting summary, suggest updates ONLY for fields where the meeting provides concrete new information. Do NOT invent. Do NOT propose updates without textual evidence.

Output valid JSON in this exact shape:
{{
  "metrics": "..." | null,
  "decision_criteria": "..." | null,
  "decision_process": "..." | null,
  "paper_process": "..." | null,
  "pain": "..." | null,
  "rationale": "one sentence per non-null field explaining what in the meeting backed it"
}}

Use null for fields you have no new info on. Keep each field under 200 chars. Be concrete and quotable.

DEAL: {deal_name} (stage: {stage})
EXISTING MEDDIC:
- metrics: {metrics}
- decision_criteria: {dc}
- decision_process: {dp}
- paper_process: {pp}
- pain: {pain}

MEETING ({date}):
attendees: {attendees}
summary: {summary}
decisions: {decisions}
{transcript_section}

Return JSON only, no prose."""


@router.post("/meetings/{meeting_id}/suggest-meddic")
async def suggest_meddic(meeting_id: str):
    """LLM extracts proposed MEDDIC field deltas from this meeting +
    deal context. Returns suggestions only — user reviews and applies."""
    from ..db.models import Deal, Meeting

    async with _sm() as s:
        m = await s.get(Meeting, meeting_id)
        if not m:
            raise HTTPException(404, "meeting not found")
        if not m.deal_id:
            raise HTTPException(400, "meeting has no associated deal")
        d = await s.get(Deal, m.deal_id)
        if not d:
            raise HTTPException(404, "deal not found")

    # Pull the live agent's LLM client — same auth path as the rest of
    # the system, no separate credentials needed.
    from .. import main as app_module
    agent = getattr(app_module, "agent", None)
    client = getattr(agent, "client", None) if agent else None
    if client is None:
        raise HTTPException(503, "LLM not configured")

    transcript_section = (
        f"transcript excerpt: {(m.transcript or '')[:2000]}"
        if m.transcript else ""
    )
    prompt = _MEDDIC_PROMPT.format(
        deal_name=d.name, stage=d.stage,
        metrics=d.metrics or "(empty)",
        dc=d.decision_criteria or "(empty)",
        dp=d.decision_process or "(empty)",
        pp=d.paper_process or "(empty)",
        pain=d.pain or "(empty)",
        date=str(m.date), attendees=m.attendees or "(none)",
        summary=m.summary or "(none)",
        decisions=m.decisions or "(none)",
        transcript_section=transcript_section,
    )

    settings = getattr(app_module, "settings", None)
    fast_model = getattr(settings, "fast_model", "claude-haiku-4-5") if settings else "claude-haiku-4-5"

    try:
        resp = await client.messages.create(
            model=fast_model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise HTTPException(500, f"LLM call failed: {e}")

    text = ""
    for block in resp.content:
        if hasattr(block, "text"):
            text += block.text
    text = text.strip()
    # Strip ```json fences if the model wrapped them
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        suggestions = json.loads(text)
    except json.JSONDecodeError:
        raise HTTPException(500, f"LLM returned non-JSON: {text[:200]}")

    # Filter to non-null fields and known keys
    deltas = {}
    for k in ("metrics", "decision_criteria", "decision_process", "paper_process", "pain"):
        v = suggestions.get(k)
        if v and isinstance(v, str) and v.strip() and v.lower() != "null":
            deltas[k] = v.strip()

    await _audit("meeting.suggest_meddic", {"meeting_id": meeting_id, "deal_id": d.id},
                 summary=f"{len(deltas)} suggestions")
    return {
        "deal_id": d.id,
        "meeting_id": meeting_id,
        "suggestions": deltas,
        "rationale": suggestions.get("rationale", ""),
    }


# =====================================================================
# PR7 — Meeting prep brief + per-deal audit log
# =====================================================================


_BRIEF_PROMPT = """You are preparing a sales engineer for an upcoming meeting on this deal.

Generate a TIGHT one-pager (≤400 words) in markdown with these sections:

## Where we are
One paragraph: deal stage, value, what changed in the last meeting, current next step. Be specific with numbers and names — no fluff.

## Who's in the room (or should be)
Bullet list of stakeholders: name (role) — sentiment/influence — last touch. Flag missing critical roles (especially: champion, economic buyer, technical buyer).

## What to probe
3–5 bullets prioritized by MEDDIC gap. For each: what to ask, why it matters. Reference the rep's wedge: industrial sales engineer, Bosch/Honeywell-class buyer.

## Open commitments
Action items the rep owes back, oldest first. Include due dates.

## Risk + competitor context
1–2 lines on what's at risk this quarter (slip risk reasoning) and any competitor noise.

## Suggested opener
ONE punchy sentence the rep can lead with.

Be concrete, opinionated, quotable. Skip empty sections.

DEAL: {name} (stage: {stage}, value: ${value:,.0f}, close: {close_date})
COMPANY: {company}{plant_section}
NEXT STEP: {next_step}
NOTES: {notes}
COMPETITORS: {competitors}

MEDDIC:
- metrics: {metrics}
- economic buyer: {eb}
- champion: {champion}
- decision criteria: {dc}
- decision process: {dp}
- paper process: {pp}
- pain: {pain}

STAKEHOLDERS ({n_stake}):
{stakeholders}

RECENT MEETINGS ({n_meet}):
{meetings}

OPEN ACTION ITEMS ({n_act}):
{actions}

ACTIVE BIDS:
{bids}

Generate the brief now."""


@router.post("/deals/{deal_id}/brief")
async def generate_brief(deal_id: str):
    """Generate a meeting-prep one-pager for this deal. Returns markdown.

    The persona's wedge — research said this is the single highest-impact
    'this saves me daily' feature. Pre-meeting brief beats post-meeting
    recap because it surfaces knowledge AT the moment of need."""
    from ..db.models import (
        ActionItem, Bid, Company, Contact, Deal, DealStakeholder, Meeting, Plant,
    )

    async with _sm() as s:
        d = await s.get(Deal, deal_id)
        if not d:
            raise HTTPException(404, "deal not found")
        company = await s.get(Company, d.company_id) if d.company_id else None
        plant = await s.get(Plant, d.plant_id) if d.plant_id else None
        sts = (await s.execute(
            select(DealStakeholder).where(DealStakeholder.deal_id == deal_id)
        )).scalars().all()
        contact_ids = {st.contact_id for st in sts}
        contacts = {}
        if contact_ids:
            crows = (await s.execute(
                select(Contact).where(Contact.id.in_(contact_ids))
            )).scalars().all()
            contacts = {c.id: c for c in crows}
        meetings = (await s.execute(
            select(Meeting).where(Meeting.deal_id == deal_id)
            .order_by(Meeting.date.desc()).limit(5)
        )).scalars().all()
        actions = (await s.execute(
            select(ActionItem).where(
                ActionItem.deal_id == deal_id, ActionItem.status == "open"
            ).order_by(ActionItem.created_at.asc())
        )).scalars().all()
        bids = (await s.execute(
            select(Bid).where(Bid.deal_id == deal_id)
            .order_by(Bid.submission_deadline.asc().nullslast())
        )).scalars().all()
        eb_contact = await s.get(Contact, d.economic_buyer_id) if d.economic_buyer_id else None
        ch_contact = await s.get(Contact, d.champion_id) if d.champion_id else None

    # Render compact context for the prompt
    def _stake_line(st):
        c = contacts.get(st.contact_id)
        name = c.name if c else "(unknown contact)"
        title = c.title if c else ""
        last_touch = ""
        if c and c.last_touch:
            from datetime import datetime as _dt, timezone as _tz
            now = _dt.now(_tz.utc)
            lt = c.last_touch if c.last_touch.tzinfo else c.last_touch.replace(tzinfo=_tz.utc)
            days = max(0, (now - lt).days)
            last_touch = f", last touch {days}d ago"
        return f"- {name} ({st.role}, {st.sentiment}/{st.influence} influence{last_touch}){' — ' + title if title else ''}"

    stake_block = "\n".join(_stake_line(st) for st in sts) if sts else "(none mapped — significant risk)"
    meet_block = "\n".join(
        f"- {str(m.date)[:10]} ({m.attendees or 'no attendees'}): "
        f"{(m.summary or '')[:200]}"
        + (f" | decisions: {(m.decisions or '')[:120]}" if m.decisions else "")
        for m in meetings
    ) or "(no meetings logged)"
    act_block = "\n".join(
        f"- {a.description[:200]}"
        + (f" (due {str(a.due_date)})" if a.due_date else " (no due date)")
        for a in actions[:10]
    ) or "(none)"
    bid_block = "\n".join(
        f"- {b.name} [{b.stage}] ${(b.value_usd or 0):,.0f}"
        + (f" deadline {b.submission_deadline.isoformat()[:10]}" if b.submission_deadline else "")
        for b in bids if b.stage in ("evaluating", "in_progress", "submitted")
    ) or "(none active)"

    plant_section = f" — Plant: {plant.name}" if plant else ""

    prompt = _BRIEF_PROMPT.format(
        name=d.name, stage=d.stage, value=d.value_usd or 0,
        close_date=str(d.close_date) if d.close_date else "no date",
        company=company.name if company else "(no company)",
        plant_section=plant_section,
        next_step=d.next_step or "(none)",
        notes=(d.notes or "(none)")[:600],
        competitors=d.competitors or "(none on file)",
        metrics=d.metrics or "(empty)",
        eb=eb_contact.name if eb_contact else "(unmapped)",
        champion=ch_contact.name if ch_contact else "(unmapped)",
        dc=d.decision_criteria or "(empty)",
        dp=d.decision_process or "(empty)",
        pp=d.paper_process or "(empty)",
        pain=d.pain or "(empty)",
        n_stake=len(sts),
        stakeholders=stake_block,
        n_meet=len(meetings),
        meetings=meet_block,
        n_act=len(actions),
        actions=act_block,
        bids=bid_block,
    )

    from .. import main as app_module
    agent = getattr(app_module, "agent", None)
    client = getattr(agent, "client", None) if agent else None
    if client is None:
        raise HTTPException(503, "LLM not configured")

    settings = getattr(app_module, "settings", None)
    # Use a stronger model for the brief — this is the rep's actual prep doc,
    # not a quick extraction. fast_model is fine, but allow override.
    model = getattr(settings, "fast_model", "claude-haiku-4-5") if settings else "claude-haiku-4-5"

    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise HTTPException(500, f"LLM call failed: {e}")

    text = ""
    for block in resp.content:
        if hasattr(block, "text"):
            text += block.text

    await _audit("deal.brief", {"deal_id": deal_id}, summary=f"{len(text)} chars")
    return {"deal_id": deal_id, "brief_md": text.strip()}


# ---- Per-deal audit log -------------------------------------------


@router.get("/deals/{deal_id}/audit")
async def deal_audit(deal_id: str, limit: int = 100):
    """Chronological tool-call + dashboard-mutation history for this deal.

    Filtering is done with a LIKE on args_summary (which contains the
    JSON-encoded tool args). Not perfect — false positives possible if
    a tool call mentions this deal_id incidentally — but good enough
    for a debugging/trust surface."""
    from ..db.models import AuditLog

    async with _sm() as s:
        rows = (await s.execute(
            select(AuditLog)
            .where(
                AuditLog.tool_name != "_turn",  # exclude per-turn token rows
                # Search both args (where the deal_id usually lives in tool
                # input) and result_summary (where dashboard create-style
                # ops put the new entity id).
                AuditLog.args_summary.contains(deal_id) | AuditLog.result_summary.contains(deal_id),
            )
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )).scalars().all()

    return {
        "items": [
            {
                "id": r.id,
                "timestamp": str(r.timestamp),
                "tool_name": r.tool_name,
                "args_summary": r.args_summary[:300],
                "result_status": r.result_status,
                "result_summary": (r.result_summary or "")[:200],
                "session_id": r.session_id,
                "duration_ms": r.duration_ms,
                "safety": r.safety,
                "source": "dashboard" if r.tool_name.startswith("dashboard:") else (
                    "bot" if r.session_id != "dashboard" else "system"
                ),
            }
            for r in rows
        ],
    }


# =====================================================================
# PR6 — unified search for the Cmd-K palette. One round-trip across
# every searchable entity instead of fanning out 5 requests.
# =====================================================================


@router.get("/search")
async def search(q: str, limit: int = 8):
    """Search deals/contacts/companies/bids/plants by name in a single
    pass. Returns up to `limit` per entity type. Empty `q` returns
    nothing (palette is intent-driven; pre-loading everything would be
    noise)."""
    from ..db.models import Bid, Company, Contact, Deal, Plant

    q_str = (q or "").strip()
    if not q_str:
        return {"results": []}
    pat = f"%{q_str}%"

    async with _sm() as s:
        deals = (await s.execute(
            select(Deal).where(Deal.name.ilike(pat))
            .order_by(Deal.updated_at.desc().nullslast()).limit(limit)
        )).scalars().all()
        contacts = (await s.execute(
            select(Contact).where(
                Contact.name.ilike(pat) | Contact.email.ilike(pat)
            ).limit(limit)
        )).scalars().all()
        companies = (await s.execute(
            select(Company).where(Company.name.ilike(pat))
            .order_by(Company.name).limit(limit)
        )).scalars().all()
        bids = (await s.execute(
            select(Bid).where(Bid.name.ilike(pat))
            .order_by(Bid.submission_deadline.asc().nullslast()).limit(limit)
        )).scalars().all()
        plants = (await s.execute(
            select(Plant).where(Plant.name.ilike(pat))
            .order_by(Plant.name).limit(limit)
        )).scalars().all()

    results: list[dict] = []
    for d in deals:
        results.append({
            "kind": "deal", "id": d.id, "title": d.name,
            "subtitle": f"{d.stage} · ${(d.value_usd or 0):,.0f}",
            "href": f"/deals/{d.id}",
        })
    for c in contacts:
        results.append({
            "kind": "contact", "id": c.id, "title": c.name,
            "subtitle": c.title or c.email or "",
            "href": "/contacts",
        })
    for co in companies:
        results.append({
            "kind": "company", "id": co.id, "title": co.name,
            "subtitle": co.industry or "",
            "href": f"/companies/{co.id}",
        })
    for b in bids:
        results.append({
            "kind": "bid", "id": b.id, "title": b.name,
            "subtitle": f"{b.stage} · ${(b.value_usd or 0):,.0f}",
            "href": f"/bids/{b.id}",
        })
    for p in plants:
        results.append({
            "kind": "plant", "id": p.id, "title": p.name,
            "subtitle": p.site_address or p.site_type,
            "href": f"/plants/{p.id}",
        })
    return {"results": results}


@router.get("/chat/{session_id}")
async def chat_history(session_id: str, limit: int = 50):
    """Recent conversation turns for a session, oldest first.

    Excludes rows that have been compacted into a SessionBrief (those
    have already been summarized into the prompt and the raw text is
    kept for audit, not re-display)."""
    from ..db.models import Conversation
    async with _sm() as s:
        rows = (await s.execute(
            select(Conversation)
            .where(
                Conversation.session_id == session_id,
                Conversation.compacted_into.is_(None),
            )
            .order_by(Conversation.timestamp.desc())
            .limit(limit)
        )).scalars().all()
    return {
        "messages": [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "interface": r.interface,
                "timestamp": str(r.timestamp),
            }
            for r in reversed(rows)
        ],
    }


# =====================================================================
# PR8 — Assets (installed base) + CoSellers + ServiceContracts
# =====================================================================


# ---- Assets -------------------------------------------------------


@router.get("/assets")
async def list_assets(
    plant_id: str = "", manufacturer: str = "", vendor: str = "",
    q: str = "", limit: int = 200,
):
    from ..db.models import Asset, Plant
    async with _sm() as s:
        stmt = select(Asset).order_by(Asset.manufacturer, Asset.name).limit(limit)
        clauses = []
        if plant_id:
            clauses.append(Asset.plant_id == plant_id)
        if manufacturer:
            clauses.append(Asset.manufacturer.ilike(f"%{manufacturer}%"))
        if vendor:
            clauses.append(Asset.vendor == vendor)
        if q:
            clauses.append(Asset.name.ilike(f"%{q}%") | Asset.model.ilike(f"%{q}%"))
        if clauses:
            stmt = select(Asset).where(*clauses).order_by(Asset.manufacturer, Asset.name).limit(limit)
        rows = (await s.execute(stmt)).scalars().all()
        plant_ids = {r.plant_id for r in rows if r.plant_id}
        plants: dict[str, Plant] = {}
        if plant_ids:
            prows = (await s.execute(select(Plant).where(Plant.id.in_(plant_ids)))).scalars().all()
            plants = {p.id: p for p in prows}
    return {
        "assets": [
            {
                "id": r.id, "name": r.name,
                "manufacturer": r.manufacturer, "model": r.model,
                "asset_type": r.asset_type, "quantity": r.quantity,
                "vendor": r.vendor,
                "serial_number": r.serial_number,
                "installed_date": str(r.installed_date) if r.installed_date else None,
                "end_of_life_date": str(r.end_of_life_date) if r.end_of_life_date else None,
                "plant_id": r.plant_id,
                "plant": plants[r.plant_id].name if r.plant_id in plants else "",
                "company_id": plants[r.plant_id].company_id if r.plant_id in plants else None,
                "notes": (r.notes or "")[:200],
            }
            for r in rows
        ],
    }


class AssetCreate(BaseModel):
    plant_id: str
    name: str
    manufacturer: str = ""
    model: str = ""
    asset_type: str = "other"
    serial_number: str = ""
    quantity: int = 1
    installed_date: str | None = None
    end_of_life_date: str | None = None
    vendor: str = "competitor"
    notes: str = ""


@router.post("/assets")
async def create_asset(body: AssetCreate):
    from ..db.models import Asset, Plant
    if body.asset_type not in _ASSET_TYPES:
        raise HTTPException(400, f"invalid asset_type: {body.asset_type}")
    if body.vendor not in _ASSET_VENDORS:
        raise HTTPException(400, f"invalid vendor: {body.vendor}")
    async with _sm() as s:
        if not await s.get(Plant, body.plant_id):
            raise HTTPException(404, "plant not found")
        a = Asset(
            plant_id=body.plant_id, name=body.name,
            manufacturer=body.manufacturer, model=body.model,
            asset_type=body.asset_type, serial_number=body.serial_number,
            quantity=body.quantity,
            installed_date=_parse_date(body.installed_date),
            end_of_life_date=_parse_date(body.end_of_life_date),
            vendor=body.vendor, notes=body.notes,
        )
        s.add(a)
        await s.commit()
        await s.refresh(a)
    await _audit("asset.create", body.model_dump(), summary=f"{a.id} {a.name}")
    return {"id": a.id, "name": a.name}


class AssetPatch(BaseModel):
    name: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    asset_type: str | None = None
    serial_number: str | None = None
    quantity: int | None = None
    installed_date: str | None = None
    end_of_life_date: str | None = None
    vendor: str | None = None
    notes: str | None = None
    plant_id: str | None = None


@router.patch("/assets/{asset_id}")
async def patch_asset(asset_id: str, body: AssetPatch):
    from ..db.models import Asset
    async with _sm() as s:
        a = await s.get(Asset, asset_id)
        if not a:
            raise HTTPException(404, "asset not found")
        if body.asset_type is not None:
            if body.asset_type not in _ASSET_TYPES:
                raise HTTPException(400, f"invalid asset_type: {body.asset_type}")
            a.asset_type = body.asset_type
        if body.vendor is not None:
            if body.vendor not in _ASSET_VENDORS:
                raise HTTPException(400, f"invalid vendor: {body.vendor}")
            a.vendor = body.vendor
        for f in ("name", "manufacturer", "model", "serial_number", "notes"):
            v = getattr(body, f)
            if v is not None:
                setattr(a, f, v)
        if body.quantity is not None:
            a.quantity = body.quantity
        if body.installed_date is not None:
            a.installed_date = _parse_date(body.installed_date)
        if body.end_of_life_date is not None:
            a.end_of_life_date = _parse_date(body.end_of_life_date)
        if body.plant_id is not None:
            a.plant_id = body.plant_id
        await s.commit()
    await _audit("asset.patch", {"id": asset_id, **body.model_dump(exclude_none=True)}, summary=asset_id)
    return {"id": asset_id, "updated": True}


@router.delete("/assets/{asset_id}")
async def delete_asset(asset_id: str):
    from ..db.models import Asset
    return await _delete_by_id(Asset, asset_id, "asset.delete", "asset")


# ---- Co-sellers (deal-scoped) ------------------------------------


@router.get("/deals/{deal_id}/co-sellers")
async def list_co_sellers(deal_id: str):
    from ..db.models import Contact, CoSeller
    async with _sm() as s:
        rows = (await s.execute(
            select(CoSeller).where(CoSeller.deal_id == deal_id)
            .order_by(CoSeller.created_at.asc())
        )).scalars().all()
        contact_ids = {r.contact_id for r in rows if r.contact_id}
        contacts = {}
        if contact_ids:
            crows = (await s.execute(select(Contact).where(Contact.id.in_(contact_ids)))).scalars().all()
            contacts = {c.id: c for c in crows}
    return {
        "co_sellers": [
            {
                "id": r.id, "deal_id": r.deal_id,
                "org_name": r.org_name, "role": r.role,
                "commission_pct": r.commission_pct or 0,
                "status": r.status,
                "contact_id": r.contact_id,
                "contact_name": contacts[r.contact_id].name if r.contact_id in contacts else "",
                "contact_title": contacts[r.contact_id].title if r.contact_id in contacts else "",
                "notes": r.notes or "",
            }
            for r in rows
        ],
    }


class CoSellerCreate(BaseModel):
    org_name: str
    role: str = "oem_rep"
    contact_id: str | None = None
    commission_pct: float = 0.0
    status: str = "active"
    notes: str = ""


@router.post("/deals/{deal_id}/co-sellers")
async def create_co_seller(deal_id: str, body: CoSellerCreate):
    from ..db.models import CoSeller, Deal
    if body.role not in _COSELLER_ROLES:
        raise HTTPException(400, f"invalid role: {body.role}")
    if body.status not in _COSELLER_STATUSES:
        raise HTTPException(400, f"invalid status: {body.status}")
    async with _sm() as s:
        if not await s.get(Deal, deal_id):
            raise HTTPException(404, "deal not found")
        cs = CoSeller(
            deal_id=deal_id, org_name=body.org_name, role=body.role,
            contact_id=body.contact_id or None,
            commission_pct=body.commission_pct, status=body.status,
            notes=body.notes,
        )
        s.add(cs)
        await s.commit()
        await s.refresh(cs)
    await _audit("co_seller.create", {"deal_id": deal_id, **body.model_dump()}, summary=cs.id)
    return {"id": cs.id, "org_name": cs.org_name}


class CoSellerPatch(BaseModel):
    org_name: str | None = None
    role: str | None = None
    contact_id: str | None = None
    commission_pct: float | None = None
    status: str | None = None
    notes: str | None = None


@router.patch("/co-sellers/{co_seller_id}")
async def patch_co_seller(co_seller_id: str, body: CoSellerPatch):
    from ..db.models import CoSeller
    async with _sm() as s:
        cs = await s.get(CoSeller, co_seller_id)
        if not cs:
            raise HTTPException(404, "co-seller not found")
        if body.role is not None:
            if body.role not in _COSELLER_ROLES:
                raise HTTPException(400, f"invalid role: {body.role}")
            cs.role = body.role
        if body.status is not None:
            if body.status not in _COSELLER_STATUSES:
                raise HTTPException(400, f"invalid status: {body.status}")
            cs.status = body.status
        for f in ("org_name", "notes"):
            v = getattr(body, f)
            if v is not None:
                setattr(cs, f, v)
        if body.commission_pct is not None:
            cs.commission_pct = body.commission_pct
        if body.contact_id is not None:
            cs.contact_id = body.contact_id or None
        await s.commit()
    await _audit("co_seller.patch", {"id": co_seller_id, **body.model_dump(exclude_none=True)}, summary=co_seller_id)
    return {"id": co_seller_id, "updated": True}


@router.delete("/co-sellers/{co_seller_id}")
async def delete_co_seller(co_seller_id: str):
    from ..db.models import CoSeller
    return await _delete_by_id(CoSeller, co_seller_id, "co_seller.delete", "co-seller")


# ---- Service Contracts ---------------------------------------------


@router.get("/contracts")
async def list_contracts(
    company_id: str = "", plant_id: str = "", status: str = "",
    limit: int = 200,
):
    """Service contracts ordered by renewal date (most urgent first)."""
    from ..db.models import Company, Plant, ServiceContract
    async with _sm() as s:
        clauses = []
        if company_id:
            clauses.append(ServiceContract.company_id == company_id)
        if plant_id:
            clauses.append(ServiceContract.plant_id == plant_id)
        if status:
            if status not in _CONTRACT_STATUSES:
                raise HTTPException(400, f"invalid status: {status}")
            clauses.append(ServiceContract.status == status)
        stmt = select(ServiceContract).order_by(
            ServiceContract.renewal_date.asc().nullslast()
        ).limit(limit)
        if clauses:
            stmt = select(ServiceContract).where(*clauses).order_by(
                ServiceContract.renewal_date.asc().nullslast()
            ).limit(limit)
        rows = (await s.execute(stmt)).scalars().all()
        co_ids = {r.company_id for r in rows if r.company_id}
        plant_ids = {r.plant_id for r in rows if r.plant_id}
        cos = {}
        plants = {}
        if co_ids:
            crows = (await s.execute(select(Company).where(Company.id.in_(co_ids)))).scalars().all()
            cos = {c.id: c.name for c in crows}
        if plant_ids:
            prows = (await s.execute(select(Plant).where(Plant.id.in_(plant_ids)))).scalars().all()
            plants = {p.id: p.name for p in prows}
    return {
        "contracts": [
            {
                "id": r.id, "name": r.name,
                "contract_type": r.contract_type,
                "value_usd_annual": r.value_usd_annual or 0,
                "start_date": str(r.start_date) if r.start_date else None,
                "end_date": str(r.end_date) if r.end_date else None,
                "renewal_date": str(r.renewal_date) if r.renewal_date else None,
                "status": r.status,
                "company_id": r.company_id, "company": cos.get(r.company_id, ""),
                "plant_id": r.plant_id, "plant": plants.get(r.plant_id, ""),
                "notes": (r.notes or "")[:200],
            }
            for r in rows
        ],
    }


class ContractCreate(BaseModel):
    company_id: str
    name: str
    plant_id: str | None = None
    contract_type: str = "pm_annual"
    value_usd_annual: float = 0.0
    start_date: str | None = None
    end_date: str | None = None
    renewal_date: str | None = None
    status: str = "active"
    contact_id: str | None = None
    notes: str = ""


@router.post("/contracts")
async def create_contract(body: ContractCreate):
    from ..db.models import Company, ServiceContract
    if body.contract_type not in _CONTRACT_TYPES:
        raise HTTPException(400, f"invalid contract_type: {body.contract_type}")
    if body.status not in _CONTRACT_STATUSES:
        raise HTTPException(400, f"invalid status: {body.status}")
    async with _sm() as s:
        if not await s.get(Company, body.company_id):
            raise HTTPException(404, "company not found")
        c = ServiceContract(
            company_id=body.company_id, name=body.name,
            plant_id=body.plant_id or None,
            contract_type=body.contract_type,
            value_usd_annual=body.value_usd_annual,
            start_date=_parse_date(body.start_date),
            end_date=_parse_date(body.end_date),
            renewal_date=_parse_date(body.renewal_date),
            status=body.status,
            contact_id=body.contact_id or None,
            notes=body.notes,
        )
        s.add(c)
        await s.commit()
        await s.refresh(c)
    await _audit("contract.create", body.model_dump(), summary=f"{c.id} {c.name}")
    return {"id": c.id, "name": c.name}


class ContractPatch(BaseModel):
    name: str | None = None
    contract_type: str | None = None
    value_usd_annual: float | None = None
    start_date: str | None = None
    end_date: str | None = None
    renewal_date: str | None = None
    status: str | None = None
    contact_id: str | None = None
    notes: str | None = None
    plant_id: str | None = None
    company_id: str | None = None


@router.patch("/contracts/{contract_id}")
async def patch_contract(contract_id: str, body: ContractPatch):
    from ..db.models import ServiceContract
    async with _sm() as s:
        c = await s.get(ServiceContract, contract_id)
        if not c:
            raise HTTPException(404, "contract not found")
        if body.contract_type is not None:
            if body.contract_type not in _CONTRACT_TYPES:
                raise HTTPException(400, f"invalid contract_type: {body.contract_type}")
            c.contract_type = body.contract_type
        if body.status is not None:
            if body.status not in _CONTRACT_STATUSES:
                raise HTTPException(400, f"invalid status: {body.status}")
            c.status = body.status
        for f in ("name", "notes"):
            v = getattr(body, f)
            if v is not None:
                setattr(c, f, v)
        if body.value_usd_annual is not None:
            c.value_usd_annual = body.value_usd_annual
        if body.start_date is not None:
            c.start_date = _parse_date(body.start_date)
        if body.end_date is not None:
            c.end_date = _parse_date(body.end_date)
        if body.renewal_date is not None:
            c.renewal_date = _parse_date(body.renewal_date)
        if body.contact_id is not None:
            c.contact_id = body.contact_id or None
        if body.plant_id is not None:
            c.plant_id = body.plant_id or None
        if body.company_id is not None:
            c.company_id = body.company_id
        await s.commit()
    await _audit("contract.patch", {"id": contract_id, **body.model_dump(exclude_none=True)}, summary=contract_id)
    return {"id": contract_id, "updated": True}


@router.delete("/contracts/{contract_id}")
async def delete_contract(contract_id: str):
    from ..db.models import ServiceContract
    return await _delete_by_id(ServiceContract, contract_id, "contract.delete", "contract")


# =====================================================================
# PR9 — Jobs (post-won execution) + DailyLog + ChangeOrder + Punchlist
# =====================================================================

_JOB_STAGES = {"scheduled", "in_progress", "punch", "inspected", "closed_out", "warranty"}
_CO_STATUSES = {"draft", "pm_review", "submitted", "approved", "rejected", "invoiced"}
_PUNCH_STATUSES = {"open", "in_progress", "done", "waived"}


@router.get("/jobs")
async def list_jobs(stage: str = "", company_id: str = "", limit: int = 100):
    from ..db.models import Company, Job
    async with _sm() as s:
        stmt = select(Job).order_by(Job.scheduled_start.asc().nullslast()).limit(limit)
        clauses = []
        if stage:
            if stage not in _JOB_STAGES:
                raise HTTPException(400, f"invalid stage: {stage}")
            clauses.append(Job.stage == stage)
        if company_id:
            clauses.append(Job.company_id == company_id)
        if clauses:
            stmt = select(Job).where(*clauses).order_by(Job.scheduled_start.asc().nullslast()).limit(limit)
        rows = (await s.execute(stmt)).scalars().all()
        co_ids = {r.company_id for r in rows if r.company_id}
        cos = {}
        if co_ids:
            crows = (await s.execute(select(Company).where(Company.id.in_(co_ids)))).scalars().all()
            cos = {c.id: c.name for c in crows}
    return {
        "jobs": [
            {
                "id": r.id, "name": r.name, "job_number": r.job_number or "",
                "stage": r.stage, "contract_value_usd": r.contract_value_usd or 0,
                "scheduled_start": str(r.scheduled_start) if r.scheduled_start else None,
                "scheduled_end": str(r.scheduled_end) if r.scheduled_end else None,
                "company_id": r.company_id, "company": cos.get(r.company_id, ""),
                "site_address": r.site_address or "",
            }
            for r in rows
        ],
    }


@router.get("/jobs/{job_id}")
async def job_detail(job_id: str):
    from ..db.models import (
        Bid, ChangeOrder, Company, DailyLog, Deal, Job, PunchlistItem, User,
    )
    async with _sm() as s:
        j = await s.get(Job, job_id)
        if not j:
            raise HTTPException(404, "job not found")
        co_name = ""
        if j.company_id:
            c = await s.get(Company, j.company_id)
            if c: co_name = c.name
        deal_name = ""
        if j.deal_id:
            d = await s.get(Deal, j.deal_id)
            if d: deal_name = d.name
        bid_name = ""
        if j.bid_id:
            b = await s.get(Bid, j.bid_id)
            if b: bid_name = b.name
        pm_name = ""
        if j.project_manager_id:
            pm = await s.get(User, j.project_manager_id)
            if pm: pm_name = pm.name
        foreman_name = ""
        if j.foreman_id:
            fm = await s.get(User, j.foreman_id)
            if fm: foreman_name = fm.name
        logs = (await s.execute(
            select(DailyLog).where(DailyLog.job_id == job_id)
            .order_by(DailyLog.log_date.desc()).limit(50)
        )).scalars().all()
        cos = (await s.execute(
            select(ChangeOrder).where(ChangeOrder.job_id == job_id)
            .order_by(ChangeOrder.created_at.desc())
        )).scalars().all()
        punch = (await s.execute(
            select(PunchlistItem).where(PunchlistItem.job_id == job_id)
            .order_by(PunchlistItem.status, PunchlistItem.created_at.desc())
        )).scalars().all()
    return {
        "job": {
            "id": j.id, "name": j.name, "job_number": j.job_number or "",
            "stage": j.stage,
            "site_address": j.site_address or "", "gc_name": j.gc_name or "",
            "scope": j.scope or "",
            "contract_value_usd": j.contract_value_usd or 0,
            "labor_budget_hours": j.labor_budget_hours or 0,
            "material_budget_usd": j.material_budget_usd or 0,
            "scheduled_start": str(j.scheduled_start) if j.scheduled_start else None,
            "scheduled_end": str(j.scheduled_end) if j.scheduled_end else None,
            "actual_start": str(j.actual_start) if j.actual_start else None,
            "actual_end": str(j.actual_end) if j.actual_end else None,
            "notes": j.notes or "",
            "company_id": j.company_id, "company": co_name,
            "deal_id": j.deal_id, "deal": deal_name,
            "bid_id": j.bid_id, "bid": bid_name,
            "project_manager_id": j.project_manager_id, "project_manager": pm_name,
            "foreman_id": j.foreman_id, "foreman": foreman_name,
        },
        "daily_logs": [
            {"id": l.id, "log_date": str(l.log_date),
             "summary": l.summary or "", "work_performed": l.work_performed or "",
             "issues": l.issues or "", "hours_total": l.hours_total or 0,
             "next_day_plan": l.next_day_plan or ""}
            for l in logs
        ],
        "change_orders": [
            {"id": c.id, "co_number": c.co_number or "", "status": c.status,
             "description": c.description or "", "price_usd": c.price_usd or 0,
             "labor_hours": c.labor_hours or 0,
             "approved_at": str(c.approved_at) if c.approved_at else None,
             "approver": c.approver or ""}
            for c in cos
        ],
        "punchlist": [
            {"id": p.id, "description": p.description, "location": p.location or "",
             "status": p.status,
             "completed_at": str(p.completed_at) if p.completed_at else None}
            for p in punch
        ],
    }


class JobCreate(BaseModel):
    name: str
    company_id: str | None = None
    deal_id: str | None = None
    bid_id: str | None = None
    job_number: str = ""
    stage: str = "scheduled"
    site_address: str = ""
    gc_name: str = ""
    scope: str = ""
    contract_value_usd: float = 0.0
    labor_budget_hours: float = 0.0
    material_budget_usd: float = 0.0
    scheduled_start: str | None = None
    scheduled_end: str | None = None
    notes: str = ""


@router.post("/jobs")
async def create_job(body: JobCreate):
    from ..db.models import Job
    if body.stage not in _JOB_STAGES:
        raise HTTPException(400, f"invalid stage: {body.stage}")
    async with _sm() as s:
        j = Job(
            name=body.name, job_number=body.job_number,
            company_id=body.company_id or None, deal_id=body.deal_id or None, bid_id=body.bid_id or None,
            stage=body.stage, site_address=body.site_address, gc_name=body.gc_name,
            scope=body.scope, contract_value_usd=body.contract_value_usd,
            labor_budget_hours=body.labor_budget_hours,
            material_budget_usd=body.material_budget_usd,
            scheduled_start=_parse_date(body.scheduled_start),
            scheduled_end=_parse_date(body.scheduled_end),
            notes=body.notes,
        )
        s.add(j); await s.commit(); await s.refresh(j)
    await _audit("job.create", body.model_dump(), summary=f"{j.id} {j.name}")
    return {"id": j.id, "name": j.name}


class JobPatch(BaseModel):
    name: str | None = None
    job_number: str | None = None
    stage: str | None = None
    site_address: str | None = None
    gc_name: str | None = None
    scope: str | None = None
    contract_value_usd: float | None = None
    labor_budget_hours: float | None = None
    material_budget_usd: float | None = None
    scheduled_start: str | None = None
    scheduled_end: str | None = None
    actual_start: str | None = None
    actual_end: str | None = None
    notes: str | None = None
    company_id: str | None = None
    deal_id: str | None = None
    bid_id: str | None = None


@router.patch("/jobs/{job_id}")
async def patch_job(job_id: str, body: JobPatch):
    from ..db.models import Job
    async with _sm() as s:
        j = await s.get(Job, job_id)
        if not j:
            raise HTTPException(404, "job not found")
        if body.stage is not None:
            if body.stage not in _JOB_STAGES:
                raise HTTPException(400, f"invalid stage: {body.stage}")
            j.stage = body.stage
        for f in ("name", "job_number", "site_address", "gc_name", "scope", "notes"):
            v = getattr(body, f)
            if v is not None: setattr(j, f, v)
        for f in ("contract_value_usd", "labor_budget_hours", "material_budget_usd"):
            v = getattr(body, f)
            if v is not None: setattr(j, f, v)
        for f in ("scheduled_start", "scheduled_end", "actual_start", "actual_end"):
            v = getattr(body, f)
            if v is not None: setattr(j, f, _parse_date(v))
        for f in ("company_id", "deal_id", "bid_id"):
            v = getattr(body, f)
            if v is not None: setattr(j, f, v or None)
        await s.commit()
    await _audit("job.patch", {"id": job_id, **body.model_dump(exclude_none=True)}, summary=job_id)
    return {"id": job_id, "updated": True}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    from ..db.models import Job
    return await _delete_by_id(Job, job_id, "job.delete", "job")


# ---- Daily Logs ---------------------------------------------------


class DailyLogCreate(BaseModel):
    log_date: str | None = None
    summary: str = ""
    work_performed: str = ""
    issues: str = ""
    hours_total: float = 0.0
    next_day_plan: str = ""
    transcript: str = ""


@router.post("/jobs/{job_id}/daily-logs")
async def create_daily_log(job_id: str, body: DailyLogCreate):
    from ..db.models import DailyLog, Job
    async with _sm() as s:
        if not await s.get(Job, job_id):
            raise HTTPException(404, "job not found")
        kwargs: dict = {
            "job_id": job_id,
            "summary": body.summary,
            "work_performed": body.work_performed,
            "issues": body.issues,
            "hours_total": body.hours_total,
            "next_day_plan": body.next_day_plan,
            "transcript": body.transcript,
        }
        d = _parse_date(body.log_date)
        if d is not None:
            kwargs["log_date"] = d
        log = DailyLog(**kwargs)
        s.add(log); await s.commit(); await s.refresh(log)
    await _audit("daily_log.create", {"job_id": job_id, **body.model_dump()}, summary=log.id)
    return {"id": log.id}


class DailyLogPatch(BaseModel):
    log_date: str | None = None
    summary: str | None = None
    work_performed: str | None = None
    issues: str | None = None
    hours_total: float | None = None
    next_day_plan: str | None = None


@router.patch("/daily-logs/{log_id}")
async def patch_daily_log(log_id: str, body: DailyLogPatch):
    from ..db.models import DailyLog
    async with _sm() as s:
        log = await s.get(DailyLog, log_id)
        if not log:
            raise HTTPException(404, "log not found")
        for f in ("summary", "work_performed", "issues", "next_day_plan"):
            v = getattr(body, f)
            if v is not None: setattr(log, f, v)
        if body.log_date is not None:
            log.log_date = _parse_date(body.log_date)
        if body.hours_total is not None:
            log.hours_total = body.hours_total
        await s.commit()
    await _audit("daily_log.patch", {"id": log_id, **body.model_dump(exclude_none=True)}, summary=log_id)
    return {"id": log_id, "updated": True}


@router.delete("/daily-logs/{log_id}")
async def delete_daily_log(log_id: str):
    from ..db.models import DailyLog
    return await _delete_by_id(DailyLog, log_id, "daily_log.delete", "daily log")


# ---- Change Orders ------------------------------------------------


class ChangeOrderCreate(BaseModel):
    description: str
    co_number: str = ""
    requested_by: str = ""
    labor_hours: float = 0.0
    material_cost_usd: float = 0.0
    price_usd: float = 0.0
    status: str = "draft"
    notes: str = ""


@router.post("/jobs/{job_id}/change-orders")
async def create_change_order(job_id: str, body: ChangeOrderCreate):
    from ..db.models import ChangeOrder, Job
    if body.status not in _CO_STATUSES:
        raise HTTPException(400, f"invalid status: {body.status}")
    async with _sm() as s:
        if not await s.get(Job, job_id):
            raise HTTPException(404, "job not found")
        co = ChangeOrder(
            job_id=job_id, description=body.description,
            co_number=body.co_number, requested_by=body.requested_by,
            labor_hours=body.labor_hours, material_cost_usd=body.material_cost_usd,
            price_usd=body.price_usd, status=body.status, notes=body.notes,
        )
        s.add(co); await s.commit(); await s.refresh(co)
    await _audit("change_order.create", {"job_id": job_id, **body.model_dump()}, summary=co.id)
    return {"id": co.id, "co_number": co.co_number}


class ChangeOrderPatch(BaseModel):
    description: str | None = None
    co_number: str | None = None
    status: str | None = None
    requested_by: str | None = None
    labor_hours: float | None = None
    material_cost_usd: float | None = None
    price_usd: float | None = None
    approver: str | None = None
    notes: str | None = None


@router.patch("/change-orders/{co_id}")
async def patch_change_order(co_id: str, body: ChangeOrderPatch):
    from datetime import datetime as _dt, timezone as _tz
    from ..db.models import ChangeOrder
    async with _sm() as s:
        co = await s.get(ChangeOrder, co_id)
        if not co:
            raise HTTPException(404, "change order not found")
        if body.status is not None:
            if body.status not in _CO_STATUSES:
                raise HTTPException(400, f"invalid status: {body.status}")
            co.status = body.status
            if body.status == "approved" and not co.approved_at:
                co.approved_at = _dt.now(_tz.utc)
        for f in ("description", "co_number", "requested_by", "approver", "notes"):
            v = getattr(body, f)
            if v is not None: setattr(co, f, v)
        for f in ("labor_hours", "material_cost_usd", "price_usd"):
            v = getattr(body, f)
            if v is not None: setattr(co, f, v)
        await s.commit()
    await _audit("change_order.patch", {"id": co_id, **body.model_dump(exclude_none=True)}, summary=co_id)
    return {"id": co_id, "updated": True}


@router.delete("/change-orders/{co_id}")
async def delete_change_order(co_id: str):
    from ..db.models import ChangeOrder
    return await _delete_by_id(ChangeOrder, co_id, "change_order.delete", "change order")


# ---- Punchlist ----------------------------------------------------


class PunchCreate(BaseModel):
    description: str
    location: str = ""
    status: str = "open"


@router.post("/jobs/{job_id}/punchlist")
async def create_punch(job_id: str, body: PunchCreate):
    from ..db.models import Job, PunchlistItem
    if body.status not in _PUNCH_STATUSES:
        raise HTTPException(400, f"invalid status: {body.status}")
    async with _sm() as s:
        if not await s.get(Job, job_id):
            raise HTTPException(404, "job not found")
        p = PunchlistItem(
            job_id=job_id, description=body.description,
            location=body.location, status=body.status,
        )
        s.add(p); await s.commit(); await s.refresh(p)
    await _audit("punch.create", {"job_id": job_id, **body.model_dump()}, summary=p.id)
    return {"id": p.id}


class PunchPatch(BaseModel):
    description: str | None = None
    location: str | None = None
    status: str | None = None


@router.patch("/punchlist/{punch_id}")
async def patch_punch(punch_id: str, body: PunchPatch):
    from datetime import datetime as _dt, timezone as _tz
    from ..db.models import PunchlistItem
    async with _sm() as s:
        p = await s.get(PunchlistItem, punch_id)
        if not p:
            raise HTTPException(404, "punch item not found")
        if body.status is not None:
            if body.status not in _PUNCH_STATUSES:
                raise HTTPException(400, f"invalid status: {body.status}")
            p.status = body.status
            if body.status == "done" and not p.completed_at:
                p.completed_at = _dt.now(_tz.utc)
        for f in ("description", "location"):
            v = getattr(body, f)
            if v is not None: setattr(p, f, v)
        await s.commit()
    await _audit("punch.patch", {"id": punch_id, **body.model_dump(exclude_none=True)}, summary=punch_id)
    return {"id": punch_id, "updated": True}


@router.delete("/punchlist/{punch_id}")
async def delete_punch(punch_id: str):
    from ..db.models import PunchlistItem
    return await _delete_by_id(PunchlistItem, punch_id, "punch.delete", "punch item")


# =====================================================================
# PR10 — Competitors + Battle Cards + Proposal Library + Win/Loss
# =====================================================================


@router.get("/competitors")
async def list_competitors():
    from ..db.models import BattleCard, Competitor
    async with _sm() as s:
        rows = (await s.execute(select(Competitor).order_by(Competitor.name))).scalars().all()
        # Count battle cards per competitor for the list view
        bc_rows = (await s.execute(select(BattleCard))).scalars().all()
        bc_count: dict[str, int] = {}
        for bc in bc_rows:
            if bc.competitor_id:
                bc_count[bc.competitor_id] = bc_count.get(bc.competitor_id, 0) + 1
    return {
        "competitors": [
            {"id": r.id, "name": r.name, "aliases": r.aliases or "",
             "strengths": r.strengths or "", "weaknesses": r.weaknesses or "",
             "pricing_notes": r.pricing_notes or "",
             "battle_card_count": bc_count.get(r.id, 0)}
            for r in rows
        ],
    }


class CompetitorCreate(BaseModel):
    name: str
    aliases: str = ""
    strengths: str = ""
    weaknesses: str = ""
    pricing_notes: str = ""


@router.post("/competitors")
async def create_competitor(body: CompetitorCreate):
    from sqlalchemy.exc import IntegrityError
    from ..db.models import Competitor
    async with _sm() as s:
        c = Competitor(**body.model_dump())
        s.add(c)
        try:
            await s.commit()
        except IntegrityError:
            await s.rollback()
            raise HTTPException(409, f"competitor '{body.name}' already exists")
        await s.refresh(c)
    await _audit("competitor.create", body.model_dump(), summary=c.id)
    return {"id": c.id, "name": c.name}


class CompetitorPatch(BaseModel):
    name: str | None = None
    aliases: str | None = None
    strengths: str | None = None
    weaknesses: str | None = None
    pricing_notes: str | None = None


@router.patch("/competitors/{comp_id}")
async def patch_competitor(comp_id: str, body: CompetitorPatch):
    from ..db.models import Competitor
    async with _sm() as s:
        c = await s.get(Competitor, comp_id)
        if not c:
            raise HTTPException(404, "competitor not found")
        for f in ("name", "aliases", "strengths", "weaknesses", "pricing_notes"):
            v = getattr(body, f)
            if v is not None: setattr(c, f, v)
        await s.commit()
    await _audit("competitor.patch", {"id": comp_id, **body.model_dump(exclude_none=True)}, summary=comp_id)
    return {"id": comp_id, "updated": True}


@router.delete("/competitors/{comp_id}")
async def delete_competitor(comp_id: str):
    from ..db.models import Competitor
    return await _delete_by_id(Competitor, comp_id, "competitor.delete", "competitor")


# ---- Battle Cards (linked to a competitor) -----------------------


@router.get("/competitors/{comp_id}/battle-cards")
async def list_battle_cards(comp_id: str):
    from ..db.models import BattleCard
    async with _sm() as s:
        rows = (await s.execute(
            select(BattleCard).where(BattleCard.competitor_id == comp_id)
            .order_by(BattleCard.created_at.desc())
        )).scalars().all()
    return {
        "battle_cards": [
            {"id": r.id, "situation": r.situation or "", "content": r.content,
             "created_at": str(r.created_at)}
            for r in rows
        ],
    }


class BattleCardCreate(BaseModel):
    situation: str = ""
    content: str


@router.post("/competitors/{comp_id}/battle-cards")
async def create_battle_card(comp_id: str, body: BattleCardCreate):
    from ..db.models import BattleCard, Competitor
    async with _sm() as s:
        if not await s.get(Competitor, comp_id):
            raise HTTPException(404, "competitor not found")
        bc = BattleCard(competitor_id=comp_id, situation=body.situation, content=body.content)
        s.add(bc); await s.commit(); await s.refresh(bc)
    await _audit("battle_card.create", {"competitor_id": comp_id, **body.model_dump()}, summary=bc.id)
    return {"id": bc.id}


@router.delete("/battle-cards/{bc_id}")
async def delete_battle_card(bc_id: str):
    from ..db.models import BattleCard
    return await _delete_by_id(BattleCard, bc_id, "battle_card.delete", "battle card")


# ---- Proposal Library --------------------------------------------


@router.get("/proposals")
async def list_proposals(section_type: str = "", q: str = "", limit: int = 100):
    from ..db.models import ProposalPrecedent
    async with _sm() as s:
        stmt = select(ProposalPrecedent).order_by(ProposalPrecedent.created_at.desc()).limit(limit)
        clauses = []
        if section_type:
            clauses.append(ProposalPrecedent.section_type == section_type)
        if q:
            clauses.append(
                ProposalPrecedent.title.ilike(f"%{q}%")
                | ProposalPrecedent.content.ilike(f"%{q}%")
                | ProposalPrecedent.tags.ilike(f"%{q}%")
            )
        if clauses:
            stmt = select(ProposalPrecedent).where(*clauses).order_by(ProposalPrecedent.created_at.desc()).limit(limit)
        rows = (await s.execute(stmt)).scalars().all()
    return {
        "proposals": [
            {"id": r.id, "title": r.title, "section_type": r.section_type or "",
             "content": r.content, "tags": r.tags or "",
             "source_deal_id": r.source_deal_id}
            for r in rows
        ],
    }


class ProposalCreate(BaseModel):
    title: str
    section_type: str = ""
    content: str
    tags: str = ""
    source_deal_id: str | None = None


@router.post("/proposals")
async def create_proposal(body: ProposalCreate):
    from ..db.models import ProposalPrecedent
    async with _sm() as s:
        p = ProposalPrecedent(
            title=body.title, section_type=body.section_type,
            content=body.content, tags=body.tags,
            source_deal_id=body.source_deal_id or None,
        )
        s.add(p); await s.commit(); await s.refresh(p)
    await _audit("proposal.create", body.model_dump(), summary=p.id)
    return {"id": p.id}


class ProposalPatch(BaseModel):
    title: str | None = None
    section_type: str | None = None
    content: str | None = None
    tags: str | None = None


@router.patch("/proposals/{p_id}")
async def patch_proposal(p_id: str, body: ProposalPatch):
    from ..db.models import ProposalPrecedent
    async with _sm() as s:
        p = await s.get(ProposalPrecedent, p_id)
        if not p:
            raise HTTPException(404, "proposal precedent not found")
        for f in ("title", "section_type", "content", "tags"):
            v = getattr(body, f)
            if v is not None: setattr(p, f, v)
        await s.commit()
    await _audit("proposal.patch", {"id": p_id, **body.model_dump(exclude_none=True)}, summary=p_id)
    return {"id": p_id, "updated": True}


@router.delete("/proposals/{p_id}")
async def delete_proposal(p_id: str):
    from ..db.models import ProposalPrecedent
    return await _delete_by_id(ProposalPrecedent, p_id, "proposal.delete", "proposal precedent")


# ---- Win/Loss ------------------------------------------------------


@router.get("/win-loss")
async def list_win_loss(limit: int = 100):
    from ..db.models import Deal, WinLossRecord
    async with _sm() as s:
        rows = (await s.execute(
            select(WinLossRecord).order_by(WinLossRecord.created_at.desc()).limit(limit)
        )).scalars().all()
        deal_ids = {r.deal_id for r in rows if r.deal_id}
        deals = {}
        if deal_ids:
            drows = (await s.execute(select(Deal).where(Deal.id.in_(deal_ids)))).scalars().all()
            deals = {d.id: d for d in drows}

    won = [r for r in rows if r.outcome == "won"]
    lost = [r for r in rows if r.outcome == "lost"]
    total_value = sum(r.value_usd or 0 for r in rows)
    win_rate = (len(won) / len(rows)) if rows else 0

    return {
        "stats": {
            "total": len(rows), "won": len(won), "lost": len(lost),
            "win_rate": round(win_rate, 3),
            "total_value": total_value,
            "won_value": sum(r.value_usd or 0 for r in won),
            "lost_value": sum(r.value_usd or 0 for r in lost),
        },
        "records": [
            {"id": r.id, "outcome": r.outcome,
             "winning_competitor": r.winning_competitor or "",
             "primary_reason": r.primary_reason or "",
             "what_worked": r.what_worked or "",
             "what_didnt": r.what_didnt or "",
             "lessons": r.lessons or "",
             "value_usd": r.value_usd or 0,
             "deal_id": r.deal_id,
             "deal_name": deals[r.deal_id].name if r.deal_id in deals else "",
             "created_at": str(r.created_at)}
            for r in rows
        ],
    }


class WinLossCreate(BaseModel):
    deal_id: str
    outcome: str  # won | lost | no_decision
    winning_competitor: str = ""
    primary_reason: str = ""
    what_worked: str = ""
    what_didnt: str = ""
    lessons: str = ""
    value_usd: float = 0.0


@router.post("/win-loss")
async def create_win_loss(body: WinLossCreate):
    from ..db.models import Deal, WinLossRecord
    if body.outcome not in {"won", "lost", "no_decision"}:
        raise HTTPException(400, f"invalid outcome: {body.outcome}")
    async with _sm() as s:
        if not await s.get(Deal, body.deal_id):
            raise HTTPException(404, "deal not found")
        r = WinLossRecord(**body.model_dump())
        s.add(r); await s.commit(); await s.refresh(r)
    await _audit("win_loss.create", body.model_dump(), summary=r.id)
    return {"id": r.id}


@router.delete("/win-loss/{r_id}")
async def delete_win_loss(r_id: str):
    from ..db.models import WinLossRecord
    return await _delete_by_id(WinLossRecord, r_id, "win_loss.delete", "win/loss record")


# =====================================================================
# Integrations — connect/disconnect external services from the UI
# =====================================================================


@router.get("/integrations")
async def integrations_status():
    """Per-provider connection status for the /settings/integrations page."""
    from .. import main as app_module
    from . import microsoft_auth
    from ..db.models import OAuthToken

    settings = getattr(app_module, "settings", None)

    # Telegram — bot token presence + allowed users count
    tg_configured = bool(settings and settings.telegram_bot_token)
    tg_users = (settings.allowed_user_ids if settings else []) or []
    telegram = {
        "id": "telegram",
        "name": "Telegram",
        "kind": "chat",
        "configured": tg_configured,
        "connected": tg_configured,
        "detail": (
            f"{len(tg_users)} authorized user(s)"
            if tg_configured else "Set TELEGRAM_BOT_TOKEN in env to enable."
        ),
    }

    # Google — credentials path presence + token row presence
    google_configured = bool(settings and settings.google_credentials_path)
    google_connected = False
    if google_configured:
        async with _sm() as s:
            row = await s.get(OAuthToken, "google")
            google_connected = bool(row)
    google = {
        "id": "google",
        "name": "Google (Gmail + Calendar)",
        "kind": "productivity",
        "configured": google_configured,
        "connected": google_connected,
        "detail": (
            "Connected" if google_connected
            else ("Set up via Google credentials path" if google_configured else "Set GOOGLE_CREDENTIALS_PATH to enable.")
        ),
    }

    # Microsoft Graph
    ms_status = await microsoft_auth.status()
    base = (settings.app_base_url if settings else "").rstrip("/")
    microsoft = {
        "id": "microsoft",
        "name": "Microsoft 365 (Outlook + Calendar)",
        "kind": "productivity",
        "configured": ms_status.get("configured", False),
        "connected": ms_status.get("connected", False),
        "detail": ms_status.get("message") or (
            "Connected" if ms_status.get("connected") else "Click Connect to authorize."
        ),
        "auth_url": f"{base}/auth/microsoft/login" if ms_status.get("configured") else None,
        "redirect_uri": ms_status.get("redirect_uri"),
    }

    # Anthropic — read mode from token_manager
    anthropic_mode = "uninit"
    tm = getattr(app_module, "token_manager", None)
    if tm:
        anthropic_mode = getattr(tm, "mode", "uninit")
    anthropic = {
        "id": "anthropic",
        "name": "Anthropic (LLM)",
        "kind": "llm",
        "configured": True,
        "connected": anthropic_mode != "uninit",
        "detail": f"Mode: {anthropic_mode}",
    }

    return {"integrations": [anthropic, telegram, google, microsoft]}


@router.post("/integrations/microsoft/disconnect")
async def integrations_microsoft_disconnect():
    from . import microsoft_auth
    await microsoft_auth.disconnect()
    await _audit("integrations.microsoft.disconnect", {}, summary="microsoft tokens cleared")
    return {"ok": True}


# =====================================================================
# Audio capture — upload a recording (any source: phone voice memo, browser
# recording, Otter export). Whisper transcribes, Haiku categorizes,
# extractions get applied or surfaced for one-click apply.
# =====================================================================


# Generous limit — actual transcription handles chunking internally (pydub +
# ffmpeg splits into 10-min mp3 segments before sending to Whisper). 500MB
# covers ~12 hours of typical voice-memo audio.
_AUDIO_MAX_BYTES = 500 * 1024 * 1024


async def _process_audio_upload(meeting_id: str, file: UploadFile) -> dict:
    """Shared body for both upload endpoints — read file, kick off pipeline."""
    from .. import main as app_module
    from ..core.audio_processor import process_meeting_audio

    settings = getattr(app_module, "settings", None)
    agent = getattr(app_module, "agent", None)
    llm_client = getattr(agent, "client", None) if agent else None

    if not settings or not settings.openai_api_key:
        raise HTTPException(503, "OPENAI_API_KEY not configured — required for transcription")
    if llm_client is None:
        raise HTTPException(503, "Agent LLM not ready")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty audio file")
    if len(raw) > _AUDIO_MAX_BYTES:
        raise HTTPException(413, f"audio file exceeds {_AUDIO_MAX_BYTES // (1024*1024)}MB limit")

    fast_model = getattr(settings, "fast_model", "claude-haiku-4-5")
    try:
        result = await process_meeting_audio(
            meeting_id=meeting_id,
            audio_bytes=raw,
            filename=file.filename or "audio.webm",
            session_maker=_sm,
            llm_client=llm_client,
            openai_api_key=settings.openai_api_key,
            fast_model=fast_model,
        )
    except Exception as e:
        await _audit("audio.process.error", {"meeting_id": meeting_id, "filename": file.filename},
                     status="error", summary=str(e)[:300])
        raise HTTPException(500, f"audio processing failed: {e}")

    await _audit("audio.process.done", {"meeting_id": meeting_id, "filename": file.filename},
                 summary=f"transcript {len(result['transcript'])} chars; type={result.get('extracted', {}).get('meeting_type', '?')}")
    return result


@router.post("/meetings/{meeting_id}/audio")
async def upload_meeting_audio(meeting_id: str, file: UploadFile = File(...)):
    """Upload audio to an existing meeting; transcribe + categorize in-place.

    Status updates land on the Meeting row (audio_processing_status field)
    so the UI can poll. Returns the transcript + extracted suggestions
    when done."""
    from ..db.models import Meeting
    async with _sm() as s:
        if not await s.get(Meeting, meeting_id):
            raise HTTPException(404, "meeting not found")
    return await _process_audio_upload(meeting_id, file)


@router.post("/deals/{deal_id}/audio")
async def upload_deal_audio(
    deal_id: str,
    file: UploadFile = File(...),
    attendees: str = Form(""),
):
    """One-shot: create a new Meeting on this deal AND upload audio in
    a single call. Used by the deal-page record/upload widget so the
    user doesn't have to first 'log meeting' then upload."""
    from datetime import datetime as _dt, timezone as _tz
    from ..db.models import Deal, Meeting
    async with _sm() as s:
        if not await s.get(Deal, deal_id):
            raise HTTPException(404, "deal not found")
        m = Meeting(
            deal_id=deal_id,
            attendees=attendees,
            date=_dt.now(_tz.utc),
            audio_processing_status="uploaded",
        )
        s.add(m)
        await s.commit()
        await s.refresh(m)
        meeting_id = m.id
    return await _process_audio_upload(meeting_id, file)


@router.get("/meetings/{meeting_id}/processing")
async def meeting_processing_status(meeting_id: str):
    """Lightweight poll endpoint — returns current processing state +
    extracted fields if done. Avoids re-transferring the full transcript."""
    from ..db.models import Meeting
    async with _sm() as s:
        m = await s.get(Meeting, meeting_id)
        if not m:
            raise HTTPException(404, "meeting not found")
        return {
            "id": m.id,
            "status": m.audio_processing_status,
            "error": m.audio_processing_error or None,
            "meeting_type": m.meeting_type,
            "sentiment": m.sentiment,
            "duration_minutes": m.duration_minutes,
            "competitors_mentioned": m.competitors_mentioned or "",
            "pricing_mentioned": m.pricing_mentioned or "",
            "has_transcript": bool(m.transcript),
        }


