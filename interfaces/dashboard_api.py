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

from fastapi import APIRouter, HTTPException
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
             "attendees": m.attendees, "decisions": m.decisions}
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
    from ..db.models import Bid, Company, Contact, Deal, Plant
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


