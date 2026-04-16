"""ActionGate — intercepts approve_external tool calls and queues them for
user approval instead of executing immediately.

Flow:
  1. Agent emits a tool_use for an approve_external tool
  2. ActionGate.intercept() writes a PendingAction row, returns a synthetic
     tool_result telling the agent the action is queued
  3. UI surfaces the card to the user (Telegram inline keyboard, etc.)
  4. User approves via API or message → ActionGate.approve() runs the actual tool
  5. Agent learns of the result via push (Telegram message) or polling
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..db.models import PendingAction
from .skill_base import Safety

logger = logging.getLogger(__name__)


# How long a pending action stays approvable before auto-expire.
# Defaults to 1 hour — strikes the balance between "user moment to think"
# and "stale request when context has shifted".
DEFAULT_EXPIRY = timedelta(hours=1)


class ActionGate:
    def __init__(self, session_maker: async_sessionmaker, expiry: timedelta = DEFAULT_EXPIRY):
        self.sm = session_maker
        self.expiry = expiry

    async def intercept(
        self,
        session_id: str,
        tool_name: str,
        tool_input: dict,
        summary: str = "",
    ) -> dict:
        """Queue a pending action. Returns a dict suitable to feed back as the
        tool_result content so the agent's loop sees a clean response.
        """
        async with self.sm() as session:
            pa = PendingAction(
                session_id=session_id,
                tool_name=tool_name,
                tool_input=json.dumps(tool_input, default=str),
                summary=summary or _default_summary(tool_name, tool_input),
                expires_at=datetime.now(timezone.utc) + self.expiry,
            )
            session.add(pa)
            await session.commit()
            await session.refresh(pa)
            return {
                "queued_for_approval": True,
                "pending_action_id": pa.id,
                "expires_at": str(pa.expires_at),
                "summary": pa.summary,
            }

    async def approve(self, action_id: str) -> PendingAction | None:
        """Mark a pending action approved. Caller (the executor) then runs the
        actual tool and reports back via mark_executed."""
        async with self.sm() as session:
            row = await session.get(PendingAction, action_id)
            if row is None or row.status != "pending":
                return None
            exp = row.expires_at
            if exp and exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp and exp < datetime.now(timezone.utc):
                row.status = "expired"
                row.decided_at = datetime.now(timezone.utc)
                await session.commit()
                return None
            row.status = "approved"
            row.decided_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(row)
            return row

    async def reject(self, action_id: str) -> PendingAction | None:
        async with self.sm() as session:
            row = await session.get(PendingAction, action_id)
            if row is None or row.status != "pending":
                return None
            row.status = "rejected"
            row.decided_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(row)
            return row

    async def mark_executed(self, action_id: str, result_summary: str) -> None:
        async with self.sm() as session:
            row = await session.get(PendingAction, action_id)
            if row is None:
                return
            row.status = "executed"
            row.executed_at = datetime.now(timezone.utc)
            row.result_summary = (result_summary or "")[:500]
            await session.commit()

    async def mark_failed(self, action_id: str, error: str) -> None:
        async with self.sm() as session:
            row = await session.get(PendingAction, action_id)
            if row is None:
                return
            row.status = "failed"
            row.executed_at = datetime.now(timezone.utc)
            row.result_summary = (error or "")[:500]
            await session.commit()

    async def list_pending(self, session_id: str = "") -> list[PendingAction]:
        async with self.sm() as session:
            q = select(PendingAction).where(PendingAction.status == "pending")
            if session_id:
                q = q.where(PendingAction.session_id == session_id)
            q = q.order_by(PendingAction.created_at.desc()).limit(20)
            rows = (await session.execute(q)).scalars().all()
        return list(rows)

    @staticmethod
    def is_external(safety: str) -> bool:
        return safety == Safety.APPROVE_EXTERNAL


def _default_summary(tool_name: str, tool_input: dict) -> str:
    """Render a one-line preview for the action card. Tool-specific niceties
    can be added here as more external tools land."""
    if tool_name == "gmail-send":
        to = tool_input.get("to", "")
        subj = tool_input.get("subject", "(no subject)")
        return f"Send email to {to}: {subj}"
    if tool_name == "calendar-create_event":
        title = tool_input.get("title", "(untitled)")
        when = tool_input.get("start", "")
        return f"Create calendar event '{title}' at {when}"
    return f"{tool_name}({_compact_args(tool_input)})"


def _compact_args(d: dict, max_len: int = 100) -> str:
    parts = [f"{k}={str(v)[:30]}" for k, v in d.items()]
    s = ", ".join(parts)
    return s[: max_len - 1] + "…" if len(s) > max_len else s


__all__ = ["ActionGate", "DEFAULT_EXPIRY"]
