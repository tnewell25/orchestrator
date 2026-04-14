"""CalendarAutoSync — hourly scan of Google Calendar, auto-creates
pre-meeting reminders for events where an attendee matches a known Contact.

This is the telepathic moment: the user never asks for a brief, but gets one
30 min before every meeting with a known contact.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from ..db.models import Contact, Reminder

logger = logging.getLogger(__name__)

SYNC_INTERVAL_S = 3600  # 1 hour
DEFAULT_BRIEF_MINUTES_BEFORE = 30


class CalendarAutoSync:
    def __init__(
        self,
        session_maker,
        calendar_skill,
        default_chat_id: str,
        minutes_before: int = DEFAULT_BRIEF_MINUTES_BEFORE,
    ):
        self.session_maker = session_maker
        self.calendar = calendar_skill
        self.default_chat_id = default_chat_id
        self.minutes_before = minutes_before
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self):
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info("CalendarAutoSync started (interval=%ds)", SYNC_INTERVAL_S)

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _loop(self):
        # Initial delay to let calendar skill finish OAuth
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=30)
            return
        except asyncio.TimeoutError:
            pass

        while not self._stop.is_set():
            try:
                if self.calendar and self.calendar._service:
                    await self._sync_once()
            except Exception as e:
                logger.error("Calendar sync error: %s", e, exc_info=True)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=SYNC_INTERVAL_S)
            except asyncio.TimeoutError:
                continue

    async def _sync_once(self):
        events = await self.calendar.list_upcoming(days=7)
        if not events:
            return

        now = datetime.now(timezone.utc)

        async with self.session_maker() as s:
            # Pre-load all contact emails for matching
            result = await s.execute(select(Contact))
            contacts = list(result.scalars().all())
            email_to_contact = {
                c.email.lower(): c for c in contacts if c.email
            }

            created = 0
            for ev in events:
                start_raw = ev.get("start")
                if not start_raw:
                    continue
                # Parse ISO datetime; skip all-day events
                try:
                    start_dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                except Exception:
                    continue

                trigger = start_dt - timedelta(minutes=self.minutes_before)
                if trigger <= now:
                    continue  # meeting imminent / past

                # Check any attendee matches a known contact
                matched_contact = None
                for email in ev.get("attendees", []):
                    c = email_to_contact.get(email.lower())
                    if c:
                        matched_contact = c
                        break
                if not matched_contact:
                    continue

                # Dedupe: skip if a pre_meeting reminder already exists for this calendar event
                existing = (
                    await s.execute(
                        select(Reminder).where(
                            Reminder.kind == "pre_meeting",
                            Reminder.message.like(f"%{ev['id']}%"),
                        )
                    )
                ).scalar_one_or_none()
                if existing:
                    continue

                msg = (
                    f"Meeting in {self.minutes_before} min: '{ev.get('title', '')}' "
                    f"with {matched_contact.name} (calendar_event={ev['id']})"
                )
                s.add(
                    Reminder(
                        trigger_at=trigger,
                        message=msg,
                        target_chat_id=self.default_chat_id or None,
                        related_contact_id=matched_contact.id,
                        kind="pre_meeting",
                    )
                )
                created += 1

            if created:
                await s.commit()
                logger.info("CalendarAutoSync created %d pre-meeting reminders", created)
