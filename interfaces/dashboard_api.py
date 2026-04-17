"""Dashboard read-only API — FastAPI router for the web frontend.

All endpoints are GET, read-only, no auth (internal network). The dashboard
consumes these to render pipeline, deals, contacts, activity, and analytics.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query
from sqlalchemy import func, select, desc

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

_sm = None  # set by mount_dashboard_api()


def mount_dashboard_api(app, session_maker):
    global _sm
    _sm = session_maker
    app.include_router(router)


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

    return {
        "deal": {
            "id": d.id, "name": d.name, "stage": d.stage,
            "value_usd": d.value_usd, "close_date": str(d.close_date) if d.close_date else None,
            "next_step": d.next_step, "notes": d.notes, "competitors": d.competitors,
        },
        "meddic": meddic,
        "meddic_gaps": gaps,
        "stakeholders": [
            {
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
