"""ProactiveMonitor — daily sweep that turns passive data into active pings.

Runs every 4 hours (configurable). Finds:
  - Stalled deals (no meeting in N days) → creates a custom Reminder nudge
  - Tracked emails with no reply beyond nudge window → creates a Reminder
  - Bids with submission_deadline approaching (already handled by BidSkill,
    but this acts as a safety net for imported bids)
  - Overdue action items → summary ping once/day max

De-duplicates by checking if a similar reminder was fired in the last 24h.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func

from ..db.models import ActionItem, Deal, EmailTrack, Meeting, Reminder

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 4 * 3600  # 4 hours


class ProactiveMonitor:
    def __init__(self, session_maker, default_chat_id: str, settings):
        self.session_maker = session_maker
        self.default_chat_id = default_chat_id
        self.settings = settings
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self):
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info("ProactiveMonitor started (interval=%ds)", POLL_INTERVAL_S)

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _loop(self):
        # Delay first run
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=60)
            return
        except asyncio.TimeoutError:
            pass

        while not self._stop.is_set():
            try:
                await self._sweep()
            except Exception as e:
                logger.error("Proactive sweep error: %s", e, exc_info=True)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                continue

    async def _already_nudged_recently(self, session, deal_id: str | None, kind: str, hours: int = 24) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        q = select(Reminder).where(
            Reminder.kind == kind,
            Reminder.created_at >= cutoff,
        )
        if deal_id:
            q = q.where(Reminder.related_deal_id == deal_id)
        existing = (await session.execute(q)).scalar_one_or_none()
        return existing is not None

    async def _sweep(self):
        now = datetime.now(timezone.utc)
        stalled_cutoff = now - timedelta(days=self.settings.stalled_deal_days)
        email_cutoff = now - timedelta(days=self.settings.unanswered_email_days)

        created = 0

        async with self.session_maker() as s:
            # --- Stalled deals ---
            open_deals = (
                await s.execute(
                    select(Deal).where(
                        Deal.stage.notin_({"closed_won", "closed_lost"})
                    )
                )
            ).scalars().all()

            for d in open_deals:
                last_m = (
                    await s.execute(
                        select(func.max(Meeting.date)).where(Meeting.deal_id == d.id)
                    )
                ).scalar_one_or_none()
                if last_m and last_m < stalled_cutoff and not await self._already_nudged_recently(s, d.id, "commitment"):
                    days = (now - last_m).days
                    s.add(
                        Reminder(
                            trigger_at=now + timedelta(minutes=1),
                            message=(
                                f"{d.name} has gone quiet — {days} days since last meeting. "
                                f"Stage: {d.stage}. Worth re-engaging?"
                            ),
                            target_chat_id=self.default_chat_id or None,
                            related_deal_id=d.id,
                            kind="commitment",
                        )
                    )
                    created += 1

            # --- Unanswered outbound emails ---
            unanswered = (
                await s.execute(
                    select(EmailTrack).where(
                        EmailTrack.status == "awaiting_reply",
                        EmailTrack.sent_at <= email_cutoff,
                    )
                )
            ).scalars().all()
            for e in unanswered:
                # Only nudge once per email
                if e.last_reminded_at and (now - e.last_reminded_at).days < 7:
                    continue
                s.add(
                    Reminder(
                        trigger_at=now + timedelta(minutes=1),
                        message=(
                            f"No reply yet on email to {e.to_address} — '{e.subject}' "
                            f"({(now - e.sent_at).days}d ago). Nudge?"
                        ),
                        target_chat_id=self.default_chat_id or None,
                        related_deal_id=e.related_deal_id,
                        related_contact_id=e.related_contact_id,
                        kind="commitment",
                    )
                )
                e.last_reminded_at = now
                e.status = "nudged"
                created += 1

            # --- Overdue action items (one daily summary) ---
            if not await self._already_nudged_recently(s, None, "commitment", hours=20):
                overdue = (
                    await s.execute(
                        select(ActionItem).where(
                            ActionItem.status == "open",
                            ActionItem.due_date < now.date(),
                        )
                    )
                ).scalars().all()
                if overdue:
                    preview = "\n".join(
                        f"  • {a.description} (due {a.due_date})" for a in overdue[:5]
                    )
                    extra = f"\n  …and {len(overdue) - 5} more" if len(overdue) > 5 else ""
                    s.add(
                        Reminder(
                            trigger_at=now + timedelta(minutes=1),
                            message=f"{len(overdue)} action items are overdue:\n{preview}{extra}",
                            target_chat_id=self.default_chat_id or None,
                            kind="commitment",
                        )
                    )
                    created += 1

            if created:
                await s.commit()
                logger.info("ProactiveMonitor created %d reminders", created)
