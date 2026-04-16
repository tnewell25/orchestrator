"""Conversation compactor — keeps the active context window bounded.

When a session grows beyond `compact_threshold` raw turns, the oldest portion
is summarized via a Haiku call into a SessionBrief. Subsequent turns inject
the brief into the system prompt instead of replaying the raw rows.

Result: cache prefix stays stable + token cost stops growing linearly with
conversation length.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..db.models import Conversation, SessionBrief

logger = logging.getLogger(__name__)


_COMPACT_SYSTEM = """You compress a conversation between a senior sales engineer
and his AI chief of staff into a tight retrospective brief.

Output a single dense paragraph (3-6 sentences) capturing:
- Active deals/contacts/companies discussed
- Decisions reached and commitments made
- Open questions or pending items
- Tone shifts that matter for continuity (e.g. user frustrated, deal stalled)

Skip pleasantries, repeated tool-call mechanics, anything obvious from the
current open items. Future-you will read this as a quick prelude before the
next turn — write so they instantly know where things stand."""


class Compactor:
    """Summarize-and-replace older conversation turns."""

    def __init__(
        self,
        session_maker: async_sessionmaker,
        anthropic_client,
        fast_model: str = "claude-haiku-4-5-20251001",
        compact_threshold: int = 40,
        keep_recent: int = 15,
    ):
        self.sm = session_maker
        self.client = anthropic_client
        self.fast_model = fast_model
        # Don't compact until we have at least this many raw rows
        self.compact_threshold = compact_threshold
        # Keep this many most-recent rows uncompacted (the active window)
        self.keep_recent = keep_recent

    async def maybe_compact(self, session_id: str) -> SessionBrief | None:
        """Compact if and only if the live row count exceeds threshold.

        Returns the new brief if one was created, else None. Safe to call
        after every agent turn — fires the LLM only when needed.
        """
        async with self.sm() as session:
            rows = (
                await session.execute(
                    select(Conversation)
                    .where(
                        Conversation.session_id == session_id,
                        Conversation.compacted_into.is_(None),
                    )
                    .order_by(Conversation.timestamp)
                )
            ).scalars().all()

        if len(rows) < self.compact_threshold:
            return None

        # Slice — older rows get rolled up; recent ones stay live.
        n_to_compact = len(rows) - self.keep_recent
        if n_to_compact <= 0:
            return None
        to_compact = rows[:n_to_compact]
        boundary_ts = to_compact[-1].timestamp

        summary_text = await self._summarize(to_compact)
        if not summary_text:
            logger.warning("Compactor produced empty summary, skipping compaction")
            return None

        return await self._write_brief_and_mark_compacted(
            session_id, summary_text, to_compact, boundary_ts,
        )

    async def _summarize(self, rows: list[Conversation]) -> str:
        if not self.client:
            return ""
        # Render the raw transcript — kept compact for Haiku context budget
        transcript = "\n".join(
            f"{r.role}: {r.content[:400]}" for r in rows
        )
        try:
            resp = await self.client.messages.create(
                model=self.fast_model,
                max_tokens=600,
                system=_COMPACT_SYSTEM,
                messages=[{"role": "user", "content": transcript}],
            )
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except Exception as e:
            logger.warning("Compactor LLM call failed: %s", e)
            return ""

    async def _write_brief_and_mark_compacted(
        self,
        session_id: str,
        summary: str,
        rows: list[Conversation],
        boundary_ts: datetime,
    ) -> SessionBrief:
        async with self.sm() as session:
            brief = SessionBrief(
                session_id=session_id,
                summary=summary,
                until_timestamp=boundary_ts,
                rows_compacted=len(rows),
            )
            session.add(brief)
            await session.flush()
            brief_id = brief.id

            # Mark the rolled-up rows so get_conversation skips them.
            ids = [r.id for r in rows]
            await session.execute(
                update(Conversation)
                .where(Conversation.id.in_(ids))
                .values(compacted_into=brief_id)
            )
            await session.commit()
            await session.refresh(brief)
            return brief


__all__ = ["Compactor"]
