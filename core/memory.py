"""MemoryStore — conversations, facts, OAuth tokens, semantic recall.

Embeddings are computed locally via fastembed (ONNX) — no external API calls.
CRM entities live in db.models and share the same Base; they're created here
via `Base.metadata.create_all`.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastembed import TextEmbedding
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ..db.models import (
    Base,
    Conversation,
    Fact,
    OAuthToken,
)
from .constants import EntityType

if TYPE_CHECKING:
    from .entity_extractor import EntityExtractor
    from .graph import EntityRef

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(
        self,
        database_url: str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_dim: int = 384,
    ):
        # Normalize URL for asyncpg driver
        if database_url.startswith("postgres://"):
            database_url = "postgresql+asyncpg://" + database_url[len("postgres://") :]
        elif database_url.startswith("postgresql://") and "+asyncpg" not in database_url:
            database_url = "postgresql+asyncpg://" + database_url[len("postgresql://") :]

        self.engine = create_async_engine(
            database_url, pool_pre_ping=True, pool_size=10, max_overflow=5
        )
        self.session_maker = async_sessionmaker(self.engine, expire_on_commit=False)
        self.embedding_dim = embedding_dim

        logger.info(f"Loading embedding model: {embedding_model}")
        self._embedder = TextEmbedding(model_name=embedding_model)
        logger.info("Embedding model loaded")

        # Optional — set via attach_extractor() post-init. When present, remember()
        # auto-creates graph edges from new memories to mentioned entities.
        self._extractor: "EntityExtractor | None" = None
        # Whether to run the LLM-backed extraction in the background after the
        # cheap substring pass. Disabled by default; flip on when a fast model is wired.
        self._llm_extract_in_background: bool = False

    def attach_extractor(self, extractor: "EntityExtractor", llm_background: bool = False):
        """Wire the entity extractor in — called from main.py after construction."""
        self._extractor = extractor
        self._llm_extract_in_background = llm_background

    async def initialize(self):
        """Create all tables + enable pgvector extension + add embedding columns.

        pgvector is optional at init — if the extension isn't available, we still
        create the tables and log a warning. Semantic search then returns empty.
        """
        async with self.engine.begin() as conn:
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                self._vector_enabled = True
                logger.info("pgvector extension enabled")
            except Exception as e:
                self._vector_enabled = False
                logger.warning("pgvector not available, semantic search disabled: %s", e)

            await conn.run_sync(Base.metadata.create_all)

            if getattr(self, "_vector_enabled", False):
                for tbl in ("semantic_memories", "battle_cards", "proposal_precedents"):
                    try:
                        await conn.execute(
                            text(
                                f"ALTER TABLE {tbl} "
                                f"ADD COLUMN IF NOT EXISTS embedding vector({self.embedding_dim})"
                            )
                        )
                    except Exception as e:
                        logger.warning("Could not add embedding to %s: %s", tbl, e)

    # -- Conversations --

    async def add_message(
        self, session_id: str, role: str, content: str, interface: str = "telegram"
    ):
        async with self.session_maker() as session:
            session.add(
                Conversation(
                    session_id=session_id,
                    role=role,
                    content=content,
                    interface=interface,
                )
            )
            await session.commit()

    async def get_conversation(self, session_id: str, limit: int = 30) -> list[dict]:
        async with self.session_maker() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(Conversation.timestamp.desc())
                .limit(limit)
            )
            rows = list(result.scalars().all())
            rows.reverse()
            return [{"role": r.role, "content": r.content} for r in rows]

    # -- Facts --

    async def upsert_fact(self, category: str, key: str, value: str):
        async with self.session_maker() as session:
            existing = (
                await session.execute(
                    select(Fact).where(Fact.category == category, Fact.key == key)
                )
            ).scalar_one_or_none()
            if existing:
                existing.value = value
                existing.updated_at = datetime.now(timezone.utc)
            else:
                session.add(Fact(category=category, key=key, value=value))
            await session.commit()

    async def get_facts(self, category: str | None = None) -> list[dict]:
        async with self.session_maker() as session:
            q = select(Fact)
            if category:
                q = q.where(Fact.category == category)
            result = await session.execute(q.order_by(Fact.category, Fact.key))
            return [
                {"category": r.category, "key": r.key, "value": r.value}
                for r in result.scalars().all()
            ]

    # -- OAuth tokens --

    async def get_oauth_token(self, provider: str = "anthropic") -> dict | None:
        async with self.session_maker() as session:
            row = (
                await session.execute(
                    select(OAuthToken).where(OAuthToken.provider == provider)
                )
            ).scalar_one_or_none()
            if not row:
                return None
            return {
                "provider": row.provider,
                "access_token": row.access_token,
                "refresh_token": row.refresh_token,
                "expires_at": row.expires_at,
                "client_id": row.client_id,
                "source": row.source,
            }

    async def upsert_oauth_token(
        self,
        provider: str,
        access_token: str,
        refresh_token: str,
        expires_at: float | None = None,
        client_id: str = "",
        source: str = "env",
    ):
        async with self.session_maker() as session:
            existing = (
                await session.execute(
                    select(OAuthToken).where(OAuthToken.provider == provider)
                )
            ).scalar_one_or_none()
            if existing:
                existing.access_token = access_token
                existing.refresh_token = refresh_token
                existing.expires_at = expires_at
                existing.client_id = client_id
                existing.source = source
                existing.updated_at = datetime.now(timezone.utc)
            else:
                session.add(
                    OAuthToken(
                        provider=provider,
                        access_token=access_token,
                        refresh_token=refresh_token,
                        expires_at=expires_at,
                        client_id=client_id,
                        source=source,
                    )
                )
            await session.commit()

    # -- Semantic memory (pgvector + local embeddings) --

    async def _embed(self, text_in: str) -> list[float]:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self._embedder.embed([text_in]))
        )
        return embeddings[0].tolist()

    async def store_memory(self, content: str, source: str = "conversation") -> str:
        """Low-level write — embeds and inserts. Returns the memory id.

        Does NOT extract entities. Use remember() for the full pipeline.
        """
        memory_id = str(uuid.uuid4())
        embedding = await self._embed(content)
        now = datetime.now(timezone.utc)
        async with self.session_maker() as session:
            if getattr(self, "_vector_enabled", True):
                await session.execute(
                    text(
                        "INSERT INTO semantic_memories "
                        "(id, content, embedding, timestamp, reinforcement_count, "
                        " last_reinforced_at, source) "
                        "VALUES (:id, :content, CAST(:embedding AS vector), :ts, 1, :ts, :source)"
                    ),
                    {
                        "id": memory_id,
                        "content": content,
                        "embedding": str(embedding),
                        "ts": now,
                        "source": source,
                    },
                )
            else:
                # Fallback when pgvector isn't installed (dev/test against sqlite)
                await session.execute(
                    text(
                        "INSERT INTO semantic_memories "
                        "(id, content, timestamp, reinforcement_count, "
                        " last_reinforced_at, source) "
                        "VALUES (:id, :content, :ts, 1, :ts, :source)"
                    ),
                    {"id": memory_id, "content": content, "ts": now, "source": source},
                )
            await session.commit()
        return memory_id

    async def remember(
        self,
        content: str,
        source: str = "conversation",
        extract_entities: bool = True,
    ) -> str:
        """High-level write — store + extract entities + create graph edges.

        Always runs the cheap substring extraction synchronously. Optionally
        kicks off the LLM extractor in the background (set llm_background=True
        when calling attach_extractor).

        Returns the new memory id so callers can reference it.
        """
        memory_id = await self.store_memory(content, source=source)

        if extract_entities and self._extractor is not None:
            from .graph import EntityRef
            source_ref = EntityRef(EntityType.MEMORY, memory_id)
            try:
                await self._extractor.extract(content, source_ref)
            except Exception as e:
                logger.warning("Substring extraction failed for memory %s: %s",
                               memory_id, e)

            if self._llm_extract_in_background:
                try:
                    await self._extractor.extract_llm_background(content, source_ref)
                except Exception as e:
                    logger.warning("LLM extraction scheduling failed: %s", e)

        return memory_id

    async def recall(
        self, query: str, limit: int = 5, min_similarity: float = 0.3
    ) -> list[dict]:
        embedding = await self._embed(query)
        async with self.session_maker() as session:
            result = await session.execute(
                text(
                    "SELECT content, timestamp, "
                    "  1 - (embedding <=> CAST(:embedding AS vector)) AS similarity "
                    "FROM semantic_memories "
                    "WHERE embedding IS NOT NULL "
                    "  AND 1 - (embedding <=> CAST(:embedding AS vector)) > :min_sim "
                    "ORDER BY embedding <=> CAST(:embedding AS vector) "
                    "LIMIT :limit"
                ),
                {"embedding": str(embedding), "min_sim": min_similarity, "limit": limit},
            )
            return [
                {
                    "content": r[0],
                    "timestamp": str(r[1]),
                    "similarity": round(r[2], 3),
                }
                for r in result.fetchall()
            ]

    # -- Generic vector helpers (reused by battle_cards, proposal_precedents) --

    async def store_vector(self, table: str, row_id: str, content: str):
        """Compute + save embedding for an existing row with a 'embedding vector(N)' column."""
        embedding = await self._embed(content)
        async with self.session_maker() as session:
            await session.execute(
                text(f"UPDATE {table} SET embedding = CAST(:e AS vector) WHERE id = :id"),
                {"e": str(embedding), "id": row_id},
            )
            await session.commit()

    async def search_vector(
        self,
        table: str,
        text_col: str,
        query: str,
        limit: int = 5,
        extra_cols: list[str] | None = None,
        min_similarity: float = 0.25,
    ) -> list[dict]:
        """Semantic search over an arbitrary vector-enabled table."""
        embedding = await self._embed(query)
        cols = ["id", text_col] + (extra_cols or [])
        col_sql = ", ".join(cols)
        async with self.session_maker() as session:
            result = await session.execute(
                text(
                    f"SELECT {col_sql}, "
                    f"  1 - (embedding <=> CAST(:e AS vector)) AS similarity "
                    f"FROM {table} "
                    f"WHERE embedding IS NOT NULL "
                    f"  AND 1 - (embedding <=> CAST(:e AS vector)) > :min_sim "
                    f"ORDER BY embedding <=> CAST(:e AS vector) "
                    f"LIMIT :limit"
                ),
                {"e": str(embedding), "min_sim": min_similarity, "limit": limit},
            )
            rows = []
            for r in result.fetchall():
                row = {c: r[i] for i, c in enumerate(cols)}
                row["similarity"] = round(r[len(cols)], 3)
                rows.append(row)
            return rows

    async def close(self):
        await self.engine.dispose()
