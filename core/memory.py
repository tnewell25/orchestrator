"""MemoryStore — conversations, facts, OAuth tokens, semantic recall.

Embeddings are computed locally via fastembed (ONNX) — no external API calls.
CRM entities live in db.models and share the same Base; they're created here
via `Base.metadata.create_all`.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone

from fastembed import TextEmbedding
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ..db.models import (
    Base,
    Conversation,
    Fact,
    OAuthToken,
)

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

    async def initialize(self):
        """Create all tables + enable pgvector extension + add embedding column."""
        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(
                text(
                    f"ALTER TABLE semantic_memories "
                    f"ADD COLUMN IF NOT EXISTS embedding vector({self.embedding_dim})"
                )
            )

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

    async def store_memory(self, content: str):
        embedding = await self._embed(content)
        async with self.session_maker() as session:
            await session.execute(
                text(
                    "INSERT INTO semantic_memories (id, content, embedding, timestamp) "
                    "VALUES (:id, :content, CAST(:embedding AS vector), :ts)"
                ),
                {
                    "id": str(uuid.uuid4()),
                    "content": content,
                    "embedding": str(embedding),
                    "ts": datetime.now(timezone.utc),
                },
            )
            await session.commit()

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

    async def close(self):
        await self.engine.dispose()
