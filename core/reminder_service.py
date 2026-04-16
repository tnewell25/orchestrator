"""ReminderService — polling scheduler for time-based prompts.

Runs as an asyncio background task. Every POLL_INTERVAL seconds it checks the
`reminders` table for rows with status='pending' and trigger_at <= now, then
fires them via the active messaging interface (Telegram). Survives restart
because state lives in Postgres, not in-memory APScheduler.

For pre-meeting reminders, when a Reminder has kind='pre_meeting' and a
related_* id, we enrich the message by calling the agent with a brief-generator
prompt — so the user gets context, not just "reminder fired".
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from ..db.models import Reminder

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 30

# Reminders whose trigger_at is further in the past than this get marked
# 'stale' and SKIPPED. Guards against the agent ever setting a wrong-year
# reminder (training-cutoff hallucination) and spamming stale pings the
# next time the service boots.
STALE_CUTOFF = timedelta(hours=24)


class ReminderService:
    def __init__(self, session_maker, telegram_bot, agent=None):
        self.session_maker = session_maker
        self.telegram_bot = telegram_bot
        self.agent = agent  # optional — used to enrich pre_meeting reminders
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    def set_agent(self, agent):
        """Late-bound because agent and service are both constructed in lifespan."""
        self.agent = agent

    async def start(self):
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info("ReminderService started (poll=%ds)", POLL_INTERVAL_S)

    async def stop(self):
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _loop(self):
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception as e:
                logger.error("Reminder tick error: %s", e, exc_info=True)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                continue

    async def _tick(self):
        now = datetime.now(timezone.utc)
        stale_boundary = now - STALE_CUTOFF
        async with self.session_maker() as s:
            result = await s.execute(
                select(Reminder)
                .where(Reminder.status == "pending", Reminder.trigger_at <= now)
                .order_by(Reminder.trigger_at.asc())
                .limit(20)
            )
            due = list(result.scalars().all())

            for r in due:
                # Skip stale — agent probably set this with a wrong-year
                # hallucination. Mark so we don't re-scan.
                trig = r.trigger_at
                if trig and trig.tzinfo is None:
                    trig = trig.replace(tzinfo=timezone.utc)
                if trig and trig < stale_boundary:
                    logger.warning(
                        "Reminder %s stale (trigger_at=%s, now=%s) — marking skipped",
                        r.id, trig, now,
                    )
                    r.status = "stale"
                    continue

                try:
                    rendered = await self._render(r)
                    await self._deliver(r, rendered)
                    r.status = "sent"
                    r.sent_at = datetime.now(timezone.utc)
                except Exception as e:
                    logger.error("Reminder %s failed: %s", r.id, e, exc_info=True)
                    r.status = "failed"
            if due:
                await s.commit()

    async def _render(self, r: Reminder) -> str:
        """Compose the text that gets sent. For pre-meeting reminders, ask the
        agent to generate a context brief. For everything else, use the stored message."""
        if r.kind == "pre_meeting" and self.agent:
            # Build a prompt that forces the agent to produce a brief
            context_refs = []
            if r.related_deal_id:
                context_refs.append(f"deal_id={r.related_deal_id}")
            if r.related_contact_id:
                context_refs.append(f"contact_id={r.related_contact_id}")
            if r.related_meeting_id:
                context_refs.append(f"prior_meeting_id={r.related_meeting_id}")
            refs_txt = "; ".join(context_refs) if context_refs else "no linked entities"

            prompt = (
                f"[PRE-MEETING BRIEF — auto-triggered]\n"
                f"{r.message}\n\n"
                f"Context refs: {refs_txt}\n\n"
                "Produce a tight pre-meeting brief. Use get_context / find tools to pull:\n"
                "- Deal stage, value, next step, MEDDIC gaps\n"
                "- Last 2 meeting summaries + open commitments on BOTH sides\n"
                "- Personal notes on the attendee (family, hobbies — conversation ammo)\n"
                "- Competitors mentioned and their status\n"
                "Format: 5-10 bullets MAX. Lead with the most important thing. "
                "End with 1-2 suggested opening lines for the meeting."
            )
            try:
                brief = await self.agent.run(
                    prompt,
                    session_id=f"reminder-{r.id}",
                    interface="reminder",
                )
                return f"⏰ {r.message}\n\n{brief}"
            except Exception as e:
                logger.error("Brief generation failed, falling back: %s", e)
                return f"⏰ {r.message}"

        if r.kind == "bid_deadline":
            return f"🚨 BID DEADLINE: {r.message}"
        if r.kind == "commitment":
            return f"📌 Commitment check: {r.message}"
        return f"⏰ {r.message}"

    async def _deliver(self, r: Reminder, text: str):
        if r.interface == "telegram" and self.telegram_bot:
            chat_id = int(r.target_chat_id) if r.target_chat_id else None
            if chat_id:
                await self.telegram_bot._send(chat_id, text)
            else:
                await self.telegram_bot.send_to_owner(text)
        else:
            logger.warning("Reminder %s has no deliverable interface", r.id)
