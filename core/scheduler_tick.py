"""SchedulerTick — periodic publisher of time-based events to the bus.

Fires HOURLY_SWEEP every hour and DAILY_SWEEP once per day at the configured
hour. Replaces the polling logic that lived in ProactiveMonitor by inverting
control: the tick fires events, rules respond.

Lightweight on purpose — actual work happens in rules; this is just the clock.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from .events import EventBus, EventType

logger = logging.getLogger(__name__)


class SchedulerTick:
    def __init__(
        self,
        bus: EventBus,
        default_chat_id: str = "",
        daily_at_hour_utc: int = 13,    # ~13:00 UTC = ~9 AM ET, ~6 AM PT
        sweep_payload_extra: dict | None = None,
    ):
        self.bus = bus
        self.default_chat_id = default_chat_id
        self.daily_at_hour_utc = daily_at_hour_utc
        self.extra = sweep_payload_extra or {}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._last_daily_date: str = ""

    async def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        logger.info("SchedulerTick started (daily at %02d:00 UTC)", self.daily_at_hour_utc)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
        logger.info("SchedulerTick stopped")

    async def _loop(self) -> None:
        while not self._stop.is_set():
            now = datetime.now(timezone.utc)
            payload = {"chat_id": self.default_chat_id, **self.extra}

            try:
                await self.bus.publish(EventType.HOURLY_SWEEP, payload=payload, source="scheduler")
            except Exception as e:
                logger.warning("HOURLY_SWEEP publish failed: %s", e)

            today = now.strftime("%Y-%m-%d")
            if now.hour == self.daily_at_hour_utc and self._last_daily_date != today:
                self._last_daily_date = today
                try:
                    await self.bus.publish(EventType.DAILY_SWEEP, payload=payload, source="scheduler")
                except Exception as e:
                    logger.warning("DAILY_SWEEP publish failed: %s", e)

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=3600)
            except asyncio.TimeoutError:
                pass


__all__ = ["SchedulerTick"]
