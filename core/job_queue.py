"""Durable background-job queue — CLAIM-CONFIRM for crash-safe work.

Pattern:
  1. enqueue(type, payload) — writes pending row
  2. claim() — next pending row flips to processing, returns it
  3. confirm(id, result) — marks completed
  4. fail(id, error) — bumps attempts, sets last_error; abandoned after MAX_ATTEMPTS
  5. recover_stuck() — rows in processing past STALE_CUTOFF reset to pending

Used today by the Compactor. Other expensive async operations can plug in
the same way without re-inventing durability.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..db.models import BackgroundJob

logger = logging.getLogger(__name__)


MAX_ATTEMPTS = 3
# A row that's been "processing" longer than this is assumed orphaned
# (worker crashed) and gets reset.
STALE_CUTOFF = timedelta(minutes=5)


class JobQueue:
    def __init__(self, session_maker: async_sessionmaker):
        self.sm = session_maker

    async def enqueue(self, job_type: str, payload: dict | None = None) -> BackgroundJob:
        payload_str = json.dumps(payload or {}, default=str)
        async with self.sm() as session:
            job = BackgroundJob(job_type=job_type, payload=payload_str, status="pending")
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job

    async def claim(self, job_type: str | None = None) -> BackgroundJob | None:
        """Atomically claim the oldest pending job (optionally filtered by type).

        Returns None if nothing pending. Caller must call confirm() or fail()."""
        async with self.sm() as session:
            q = select(BackgroundJob).where(BackgroundJob.status == "pending")
            if job_type:
                q = q.where(BackgroundJob.job_type == job_type)
            q = q.order_by(BackgroundJob.created_at).limit(1)

            row = (await session.execute(q)).scalar_one_or_none()
            if row is None:
                return None
            row.status = "processing"
            row.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(row)
            return row

    async def confirm(self, job_id: str) -> None:
        async with self.sm() as session:
            row = await session.get(BackgroundJob, job_id)
            if row is None:
                return
            row.status = "completed"
            row.updated_at = datetime.now(timezone.utc)
            await session.commit()

    async def fail(self, job_id: str, error: str) -> BackgroundJob | None:
        async with self.sm() as session:
            row = await session.get(BackgroundJob, job_id)
            if row is None:
                return None
            row.attempts = (row.attempts or 0) + 1
            row.last_error = (error or "")[:500]
            row.status = "abandoned" if row.attempts >= MAX_ATTEMPTS else "pending"
            row.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(row)
            return row

    async def recover_stuck(self) -> int:
        """Reset rows stuck in processing past STALE_CUTOFF back to pending.
        Call on app startup. Returns count reset."""
        cutoff = datetime.now(timezone.utc) - STALE_CUTOFF
        async with self.sm() as session:
            stuck = (
                await session.execute(
                    select(BackgroundJob).where(
                        BackgroundJob.status == "processing",
                        BackgroundJob.updated_at < cutoff,
                    )
                )
            ).scalars().all()
            for row in stuck:
                row.status = "pending"
                row.attempts = (row.attempts or 0) + 1
                row.updated_at = datetime.now(timezone.utc)
                if row.attempts >= MAX_ATTEMPTS:
                    row.status = "abandoned"
            await session.commit()
            return len(stuck)

    async def payload_of(self, job: BackgroundJob) -> dict:
        if not job.payload:
            return {}
        try:
            return json.loads(job.payload)
        except (ValueError, TypeError):
            return {}


__all__ = ["JobQueue", "MAX_ATTEMPTS", "STALE_CUTOFF"]
