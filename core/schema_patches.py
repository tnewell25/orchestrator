"""Idempotent schema migrations for columns added after initial create_all.

`Base.metadata.create_all()` only creates MISSING tables — it doesn't add new
columns to existing ones. When the code evolves and adds columns to a table
that already exists in production, we need explicit `ALTER TABLE ADD COLUMN
IF NOT EXISTS` statements.

Called from `MemoryStore.initialize()` after create_all. Safe to run on every
startup — `IF NOT EXISTS` makes each statement a no-op when the column already
exists.

When adding a new column to an existing model, ALSO add the ALTER here.
"""
from __future__ import annotations

import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)


# Each entry: (table, column_definition) — plain ADD COLUMN IF NOT EXISTS.
# Keep ordering: parent tables created by create_all before children reference them.
_COLUMN_PATCHES: list[tuple[str, str]] = [
    # Conversation — W6 compaction
    ("conversations", "compacted_into VARCHAR"),

    # SemanticMemory — W1 reinforcement + W3 dedup + Opt6 structured
    ("semantic_memories", "reinforcement_count INTEGER DEFAULT 1"),
    ("semantic_memories", "last_reinforced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()"),
    ("semantic_memories", "source VARCHAR DEFAULT 'conversation'"),
    ("semantic_memories", "content_hash VARCHAR(16)"),
    ("semantic_memories", "title VARCHAR(200) DEFAULT ''"),
    ("semantic_memories", "facts_json TEXT DEFAULT ''"),
    ("semantic_memories", "concepts_json TEXT DEFAULT ''"),

    # AuditLog — W8 safety + Opt4 token tracking
    ("audit_log", "safety VARCHAR DEFAULT 'auto'"),
    ("audit_log", "model VARCHAR DEFAULT ''"),
    ("audit_log", "input_tokens INTEGER DEFAULT 0"),
    ("audit_log", "output_tokens INTEGER DEFAULT 0"),
    ("audit_log", "cache_read_tokens INTEGER DEFAULT 0"),
    ("audit_log", "cache_creation_tokens INTEGER DEFAULT 0"),

    # PR3 — industrial data model. Plants attach to deals and bids so
    # account planning happens at the site level (the right unit for
    # multi-site industrial customers like Bosch/Honeywell).
    ("deals", "plant_id VARCHAR REFERENCES plants(id) ON DELETE SET NULL"),
    ("bids", "plant_id VARCHAR REFERENCES plants(id) ON DELETE SET NULL"),

    # Meeting capture + auto-categorization (Phase 1).
    ("meetings", "meeting_type VARCHAR DEFAULT 'other'"),
    ("meetings", "sentiment VARCHAR DEFAULT 'unknown'"),
    ("meetings", "duration_minutes FLOAT DEFAULT 0.0"),
    ("meetings", "audio_processing_status VARCHAR DEFAULT 'idle'"),
    ("meetings", "audio_processing_error VARCHAR(500) DEFAULT ''"),
    ("meetings", "competitors_mentioned TEXT DEFAULT ''"),
    ("meetings", "pricing_mentioned TEXT DEFAULT ''"),
]


# Indexes on columns added above. CREATE INDEX IF NOT EXISTS is Postgres-native.
_INDEX_PATCHES: list[tuple[str, str, str]] = [
    # (index_name, table, column)
    ("ix_conversations_compacted_into", "conversations", "compacted_into"),
    ("ix_semantic_memories_content_hash", "semantic_memories", "content_hash"),
]


async def apply_schema_patches(conn) -> None:
    """Run every patch in order. Each is idempotent."""
    for table, coldef in _COLUMN_PATCHES:
        try:
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {coldef}"))
        except Exception as e:
            # Don't crash startup on one bad patch — log + continue so later
            # patches still apply. Most likely cause: the target TABLE itself
            # doesn't exist yet (first deploy on a fresh DB), which is fine.
            logger.warning("Schema patch failed on %s: %s — %s", table, coldef, e)

    for index_name, table, column in _INDEX_PATCHES:
        try:
            await conn.execute(
                text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} ({column})")
            )
        except Exception as e:
            logger.warning("Index patch failed %s on %s(%s): %s",
                           index_name, table, column, e)


__all__ = ["apply_schema_patches"]
