"""JobSkill — field jobs / work orders.

A Job is the execution layer that picks up after a Bid is won or a Deal closes.
Rich context: crew, budget burn, photos, punchlist, change orders, inspections.
"""
from datetime import date, datetime, timezone

import dateparser
from sqlalchemy import select, or_, func

from ..core.skill_base import Skill, tool
from ..db.models import (
    ChangeOrder,
    DailyLog,
    Inspection,
    Job,
    JobPhoto,
    PunchlistItem,
    Timesheet,
    User,
)


_STAGES = {"scheduled", "in_progress", "punch", "inspected", "closed_out", "warranty"}


def _parse_date(s: str) -> date | None:
    if not s:
        return None
    dt = dateparser.parse(s)
    return dt.date() if dt else None


class JobSkill(Skill):
    name = "job"
    description = "Manage field jobs — crew, budget, schedule, punchlist, photos."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool(
        "Create a new field job. job_number is your internal code (e.g. '25-0142'). "
        "Optionally link to a won Deal (deal_id) or Bid (bid_id) so the scope and "
        "value carry over. stage defaults to 'scheduled'."
    )
    async def create(
        self,
        name: str,
        job_number: str = "",
        company_id: str = "",
        deal_id: str = "",
        bid_id: str = "",
        site_address: str = "",
        scope: str = "",
        contract_value_usd: float = 0.0,
        labor_budget_hours: float = 0.0,
        material_budget_usd: float = 0.0,
        scheduled_start: str = "",
        scheduled_end: str = "",
        project_manager_id: str = "",
        foreman_id: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            j = Job(
                name=name,
                job_number=job_number or None,
                company_id=company_id or None,
                deal_id=deal_id or None,
                bid_id=bid_id or None,
                site_address=site_address,
                scope=scope,
                contract_value_usd=contract_value_usd,
                labor_budget_hours=labor_budget_hours,
                material_budget_usd=material_budget_usd,
                scheduled_start=_parse_date(scheduled_start),
                scheduled_end=_parse_date(scheduled_end),
                project_manager_id=project_manager_id or None,
                foreman_id=foreman_id or None,
            )
            s.add(j)
            await s.commit()
            await s.refresh(j)
            return {"id": j.id, "name": j.name, "job_number": j.job_number, "stage": j.stage}

    @tool("Find jobs by name, job_number, or site_address substring.")
    async def find(self, query: str) -> list[dict]:
        async with self.session_maker() as s:
            q = f"%{query.lower()}%"
            result = await s.execute(
                select(Job).where(
                    or_(
                        Job.name.ilike(q),
                        Job.job_number.ilike(q),
                        Job.site_address.ilike(q),
                    )
                ).limit(20)
            )
            return [
                {
                    "id": r.id,
                    "job_number": r.job_number,
                    "name": r.name,
                    "stage": r.stage,
                    "site_address": r.site_address,
                    "scheduled_start": str(r.scheduled_start) if r.scheduled_start else None,
                }
                for r in result.scalars().all()
            ]

    @tool("List active jobs (stage in: scheduled, in_progress, punch, inspected).")
    async def list_active(self) -> list[dict]:
        async with self.session_maker() as s:
            result = await s.execute(
                select(Job).where(
                    Job.stage.in_({"scheduled", "in_progress", "punch", "inspected"})
                ).order_by(Job.scheduled_start.asc().nullslast())
            )
            return [
                {
                    "id": r.id,
                    "job_number": r.job_number,
                    "name": r.name,
                    "stage": r.stage,
                    "site_address": r.site_address,
                    "scheduled_end": str(r.scheduled_end) if r.scheduled_end else None,
                }
                for r in result.scalars().all()
            ]

    @tool(
        "Update job fields. stage must be: scheduled, in_progress, punch, inspected, "
        "closed_out, warranty. actual_start/actual_end accept natural language dates."
    )
    async def update(
        self,
        job_id: str,
        name: str = "",
        stage: str = "",
        scope: str = "",
        contract_value_usd: float = -1.0,
        labor_budget_hours: float = -1.0,
        material_budget_usd: float = -1.0,
        actual_start: str = "",
        actual_end: str = "",
        notes: str = "",
    ) -> dict:
        async with self.session_maker() as s:
            j = await s.get(Job, job_id)
            if not j:
                return {"error": f"Job {job_id} not found"}
            if stage:
                if stage not in _STAGES:
                    return {"error": f"Invalid stage '{stage}'. Valid: {sorted(_STAGES)}"}
                j.stage = stage
            if name:
                j.name = name
            if scope:
                j.scope = scope
            if contract_value_usd >= 0:
                j.contract_value_usd = contract_value_usd
            if labor_budget_hours >= 0:
                j.labor_budget_hours = labor_budget_hours
            if material_budget_usd >= 0:
                j.material_budget_usd = material_budget_usd
            if actual_start:
                j.actual_start = _parse_date(actual_start)
            if actual_end:
                j.actual_end = _parse_date(actual_end)
            if notes:
                j.notes = (j.notes + "\n" if j.notes else "") + notes
            await s.commit()
            return {"id": j.id, "stage": j.stage, "updated": True}

    @tool(
        "Get full job context: crew, budget burn (hours/dollars vs budget), recent "
        "daily logs, open punchlist, pending change orders, upcoming inspections, "
        "photo counts by category. Use when user asks 'how's X job going?' or to "
        "prep a client status update."
    )
    async def get_context(self, job_id: str) -> dict:
        async with self.session_maker() as s:
            j = await s.get(Job, job_id)
            if not j:
                return {"error": f"Job {job_id} not found"}

            # Budget burn
            hours_result = await s.execute(
                select(func.sum(Timesheet.hours)).where(Timesheet.job_id == job_id)
            )
            hours_burned = float(hours_result.scalar() or 0.0)

            # Recent daily logs (last 5)
            logs = (
                await s.execute(
                    select(DailyLog)
                    .where(DailyLog.job_id == job_id)
                    .order_by(DailyLog.log_date.desc())
                    .limit(5)
                )
            ).scalars().all()

            # Open punchlist
            punch = (
                await s.execute(
                    select(PunchlistItem)
                    .where(
                        PunchlistItem.job_id == job_id,
                        PunchlistItem.status.in_({"open", "in_progress"}),
                    )
                )
            ).scalars().all()

            # Pending change orders
            cos = (
                await s.execute(
                    select(ChangeOrder)
                    .where(
                        ChangeOrder.job_id == job_id,
                        ChangeOrder.status.in_({"draft", "pm_review", "submitted"}),
                    )
                )
            ).scalars().all()

            # Upcoming inspections
            now = datetime.now(timezone.utc)
            insp = (
                await s.execute(
                    select(Inspection)
                    .where(
                        Inspection.job_id == job_id,
                        Inspection.status == "scheduled",
                    )
                    .order_by(Inspection.scheduled_at.asc())
                )
            ).scalars().all()

            # Photo counts by category
            photo_count = await s.execute(
                select(JobPhoto.category, func.count(JobPhoto.id))
                .where(JobPhoto.job_id == job_id)
                .group_by(JobPhoto.category)
            )
            photos_by_cat = {row[0]: row[1] for row in photo_count.all()}

            # Crew
            pm = await s.get(User, j.project_manager_id) if j.project_manager_id else None
            foreman = await s.get(User, j.foreman_id) if j.foreman_id else None

            return {
                "job": {
                    "id": j.id,
                    "job_number": j.job_number,
                    "name": j.name,
                    "stage": j.stage,
                    "site_address": j.site_address,
                    "contract_value_usd": j.contract_value_usd,
                    "scheduled_start": str(j.scheduled_start) if j.scheduled_start else None,
                    "scheduled_end": str(j.scheduled_end) if j.scheduled_end else None,
                },
                "crew": {
                    "project_manager": {"id": pm.id, "name": pm.name} if pm else None,
                    "foreman": {"id": foreman.id, "name": foreman.name} if foreman else None,
                },
                "budget_burn": {
                    "hours_burned": hours_burned,
                    "hours_budget": j.labor_budget_hours,
                    "hours_pct": round(100 * hours_burned / j.labor_budget_hours, 1) if j.labor_budget_hours else None,
                },
                "recent_daily_logs": [
                    {
                        "date": str(l.log_date),
                        "summary": l.summary[:200],
                        "hours_total": l.hours_total,
                    }
                    for l in logs
                ],
                "open_punchlist_count": len(punch),
                "open_punchlist": [
                    {"id": p.id, "description": p.description, "location": p.location, "status": p.status}
                    for p in punch[:10]
                ],
                "pending_change_orders": [
                    {"id": c.id, "co_number": c.co_number, "status": c.status, "price_usd": c.price_usd, "description": c.description[:150]}
                    for c in cos
                ],
                "upcoming_inspections": [
                    {"id": i.id, "kind": i.kind, "scheduled_at": str(i.scheduled_at) if i.scheduled_at else None, "jurisdiction": i.jurisdiction}
                    for i in insp
                ],
                "photos_by_category": photos_by_cat,
            }
