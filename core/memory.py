"""MemoryStore — conversations, facts, OAuth tokens, semantic recall.

Embeddings are computed locally via fastembed (ONNX) — no external API calls.
CRM entities live in db.models and share the same Base; they're created here
via `Base.metadata.create_all`.
"""
import asyncio
import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ..db.models import (
    Base,
    Conversation,
    Edge,
    Fact,
    OAuthToken,
    SemanticMemory,
    SessionBrief,
)
from .constants import (
    GRAPH_PROXIMITY_BONUS,
    RECENCY_HALF_LIFE_DAYS,
    EntityType,
)

if TYPE_CHECKING:
    from .entity_extractor import EntityExtractor
    from .graph import EntityRef, GraphStore

logger = logging.getLogger(__name__)


def _recency_factor(ts: datetime | None, now: datetime, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """Exponential decay in [~0.4, 1.0]. Floor at 0.4 so old memories never
    drop to zero — they can still surface on strong vector match.

    Sqlite returns naive datetimes; postgres returns aware. Normalize to UTC
    so subtraction never crashes.
    """
    if ts is None:
        return 0.4
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    decay = 0.5 ** (age_days / half_life_days)
    return 0.4 + 0.6 * decay


def _reinforcement_boost(count: int) -> float:
    """log(count) tail boost — small but separates 1-off mentions from facts
    the user has reinforced 10+ times. Returns 0 for count=1, ~0.23 for count=10."""
    if count is None or count < 1:
        return 0.0
    return 0.1 * math.log(count)


# How long after an insert an identical content is treated as a reinforcement
# instead of a new memory. Longer = more aggressive dedup; shorter = more
# lenient for long-running agent sessions where the user might legit restate a fact.
DEDUP_WINDOW_SECONDS = 60


def _json_load_list(s: str | None) -> list:
    """Tolerant JSON array parser — empty string / None / malformed → []."""
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else []
    except (ValueError, TypeError):
        return []


def _content_hash(content: str) -> str:
    """SHA256 truncated to 16 hex chars. Same input → same hash across processes,
    cheap to index, low enough collision risk for our scale (<1M memories)."""
    normalized = (content or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _hybrid_score(
    vector_sim: float,
    recency: float,
    reinforcement_count: int,
    proximity: float,
) -> float:
    """Combine the four signals into a single rank.

    Scale: vector_sim ∈ [0,1] dominates. Other terms are small additions.
    Tunable via constants.RECENCY_HALF_LIFE_DAYS / GRAPH_PROXIMITY_BONUS.
    """
    return (
        vector_sim * recency
        + GRAPH_PROXIMITY_BONUS * proximity
        + _reinforcement_boost(reinforcement_count)
    )


class MemoryStore:
    def __init__(
        self,
        database_url: str,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_dim: int = 384,
        embedder=None,
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

        # Embedder is injectable for tests. In prod, load the local fastembed
        # model lazily — defer the heavy import until we actually need it.
        if embedder is not None:
            self._embedder = embedder
        else:
            from fastembed import TextEmbedding
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
        """Active (non-compacted) messages, oldest→newest."""
        async with self.session_maker() as session:
            result = await session.execute(
                select(Conversation)
                .where(
                    Conversation.session_id == session_id,
                    Conversation.compacted_into.is_(None),
                )
                .order_by(Conversation.timestamp.desc())
                .limit(limit)
            )
            rows = list(result.scalars().all())
            rows.reverse()
            return [{"role": r.role, "content": r.content} for r in rows]

    async def get_latest_session_brief(self, session_id: str) -> str:
        """Return the most-recent compaction summary for a session, or '' if none.

        Injected into the system prompt so the agent retains context that has
        scrolled out of the active window."""
        async with self.session_maker() as session:
            row = (
                await session.execute(
                    select(SessionBrief)
                    .where(SessionBrief.session_id == session_id)
                    .order_by(SessionBrief.until_timestamp.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            return row.summary if row else ""

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
        first = embeddings[0]
        # fastembed returns numpy arrays; injected test embedders may return lists
        return first.tolist() if hasattr(first, "tolist") else list(first)

    async def store_memory(
        self,
        content: str,
        source: str = "conversation",
        title: str = "",
        facts: list[str] | None = None,
        concepts: list[str] | None = None,
    ) -> str:
        """Low-level write — embeds and inserts. Returns the memory id.

        Dedup: if an identical memory was stored in the last DEDUP_WINDOW_SECONDS,
        we bump its reinforcement_count + last_reinforced_at and return the
        existing id instead of re-embedding + re-inserting. This halves DB
        growth on tool-loop retries and is the key to reinforcement_count
        meaningfully tracking "how often I've seen this fact."

        title/facts/concepts are optional structured fields. When present, they
        let Block D render tighter (title + bullets) and hybrid recall filter
        on concept tags. Callers that have structured data (MeetingSkill,
        EmailTriageSkill) should populate them; others can pass raw content only.

        Does NOT extract entities. Use remember() for the full pipeline.
        """
        hash_ = _content_hash(content)
        now = datetime.now(timezone.utc)
        dedup_cutoff = now - timedelta(seconds=DEDUP_WINDOW_SECONDS)

        # Cheap dedup check BEFORE the embedding call — saves the embed cost
        # on the hot path of tool-loop retries.
        async with self.session_maker() as session:
            existing = (
                await session.execute(
                    select(SemanticMemory)
                    .where(
                        SemanticMemory.content_hash == hash_,
                        SemanticMemory.last_reinforced_at >= dedup_cutoff,
                    )
                    .order_by(SemanticMemory.last_reinforced_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if existing is not None:
                existing.reinforcement_count = (existing.reinforcement_count or 1) + 1
                existing.last_reinforced_at = now
                await session.commit()
                return existing.id

        memory_id = str(uuid.uuid4())
        embedding = await self._embed(content)
        facts_json = json.dumps(facts) if facts else ""
        concepts_json = json.dumps(concepts) if concepts else ""
        async with self.session_maker() as session:
            if getattr(self, "_vector_enabled", True):
                await session.execute(
                    text(
                        "INSERT INTO semantic_memories "
                        "(id, content, embedding, timestamp, reinforcement_count, "
                        " last_reinforced_at, source, content_hash, title, facts_json, concepts_json) "
                        "VALUES (:id, :content, CAST(:embedding AS vector), :ts, 1, :ts, "
                        "        :source, :hash, :title, :facts, :concepts)"
                    ),
                    {
                        "id": memory_id, "content": content,
                        "embedding": str(embedding), "ts": now,
                        "source": source, "hash": hash_,
                        "title": title[:200], "facts": facts_json,
                        "concepts": concepts_json,
                    },
                )
            else:
                await session.execute(
                    text(
                        "INSERT INTO semantic_memories "
                        "(id, content, timestamp, reinforcement_count, "
                        " last_reinforced_at, source, content_hash, title, facts_json, concepts_json) "
                        "VALUES (:id, :content, :ts, 1, :ts, :source, :hash, :title, :facts, :concepts)"
                    ),
                    {"id": memory_id, "content": content, "ts": now,
                     "source": source, "hash": hash_,
                     "title": title[:200], "facts": facts_json,
                     "concepts": concepts_json},
                )
            await session.commit()
        return memory_id

    async def remember(
        self,
        content: str,
        source: str = "conversation",
        extract_entities: bool = True,
        title: str = "",
        facts: list[str] | None = None,
        concepts: list[str] | None = None,
    ) -> str:
        """High-level write — store + extract entities + create graph edges.

        Always runs the cheap substring extraction synchronously. Optionally
        kicks off the LLM extractor in the background (set llm_background=True
        when calling attach_extractor).

        Structured args (title/facts/concepts) are pass-through to store_memory;
        populate them when you have them for richer Block D rendering.

        Returns the new memory id so callers can reference it.
        """
        memory_id = await self.store_memory(
            content, source=source, title=title,
            facts=facts, concepts=concepts,
        )

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

    def attach_graph(self, graph: "GraphStore"):
        """Wire in the graph store — required for proximity-aware recall.
        Called from main.py after both objects exist."""
        self._graph = graph

    async def recall(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        focus_ref: "EntityRef | None" = None,
    ) -> list[dict]:
        """Backwards-compatible entry — delegates to recall_hybrid.

        focus_ref is optional; without it, scoring is vector × recency × reinforcement.
        With it, memories graph-proximate to the focus get a proximity bonus.
        """
        return await self.recall_hybrid(
            query=query, focus_ref=focus_ref, limit=limit, min_similarity=min_similarity,
        )

    async def recall_hybrid(
        self,
        query: str,
        focus_ref: "EntityRef | None" = None,
        limit: int = 5,
        candidate_pool: int | None = None,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        """Hybrid scoring: vector_sim × recency × reinforcement_boost + proximity.

        Pulls a wider candidate pool from vector search (default 4×limit), then
        re-ranks in Python using metadata + graph proximity to focus_ref.
        """
        candidate_pool = candidate_pool or max(limit * 4, 12)
        candidates = await self._fetch_vector_candidates(
            query, limit=candidate_pool, min_similarity=min_similarity,
        )
        if not candidates:
            return []

        # Build focus subgraph ONCE for O(1) proximity membership checks.
        proximate_ids: set[str] = set()
        if focus_ref is not None and getattr(self, "_graph", None) is not None:
            try:
                sg = await self._graph.subgraph(focus_ref, max_depth=2, max_nodes=80)
                proximate_ids = {n.id for n in sg.nodes if n.type == EntityType.MEMORY}
            except Exception as e:
                logger.warning("Subgraph fetch for focus %s failed: %s", focus_ref, e)

        now = datetime.now(timezone.utc)
        scored = []
        for c in candidates:
            ts = c["timestamp"] if isinstance(c["timestamp"], datetime) else None
            ref_ts = c["last_reinforced_at"] or ts
            recency = _recency_factor(ref_ts, now)
            proximity = 1.0 if c["id"] in proximate_ids else 0.0
            score = _hybrid_score(
                vector_sim=c["similarity"],
                recency=recency,
                reinforcement_count=c["reinforcement_count"],
                proximity=proximity,
            )
            scored.append({
                "id": c["id"],
                "content": c["content"],
                "title": c.get("title", ""),
                "facts": c.get("facts", []),
                "concepts": c.get("concepts", []),
                "timestamp": str(ts) if ts else "",
                "source": c["source"],
                "similarity": round(c["similarity"], 3),
                "recency": round(recency, 3),
                "reinforcement": c["reinforcement_count"],
                "proximity": proximity,
                "score": round(score, 3),
            })

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:limit]

    async def recall_for_entity(
        self,
        focus_ref: "EntityRef",
        limit: int = 10,
        max_depth: int = 2,
    ) -> list[dict]:
        """Graph-only recall: every memory connected to focus_ref within k hops,
        ordered by recency. No vector search — useful when you want EVERYTHING
        about a deal/contact regardless of how it was phrased.
        """
        if getattr(self, "_graph", None) is None:
            logger.warning("recall_for_entity called without graph attached")
            return []

        sg = await self._graph.subgraph(focus_ref, max_depth=max_depth, max_nodes=200)
        memory_ids = [n.id for n in sg.nodes if n.type == EntityType.MEMORY]
        if not memory_ids:
            return []

        async with self.session_maker() as session:
            rows = (
                await session.execute(
                    select(SemanticMemory)
                    .where(SemanticMemory.id.in_(memory_ids))
                    .order_by(SemanticMemory.last_reinforced_at.desc().nullslast())
                    .limit(limit)
                )
            ).scalars().all()

        now = datetime.now(timezone.utc)
        return [
            {
                "id": r.id,
                "content": r.content,
                "timestamp": str(r.timestamp),
                "source": r.source,
                "reinforcement": r.reinforcement_count or 1,
                "recency": round(_recency_factor(
                    r.last_reinforced_at or r.timestamp, now
                ), 3),
                "via": "graph_proximity",
            }
            for r in rows
        ]

    async def _fetch_vector_candidates(
        self, query: str, limit: int, min_similarity: float
    ) -> list[dict]:
        """Pulls top-N candidates by vector similarity + their hybrid metadata.

        Pgvector path joins on the embedding column for similarity; sqlite
        fallback returns recent memories ordered by reinforced_at (used by tests).
        """
        if getattr(self, "_vector_enabled", True):
            embedding = await self._embed(query)
            async with self.session_maker() as session:
                result = await session.execute(
                    text(
                        "SELECT id, content, timestamp, source, "
                        "       reinforcement_count, last_reinforced_at, "
                        "       title, facts_json, concepts_json, "
                        "       1 - (embedding <=> CAST(:embedding AS vector)) AS similarity "
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
                        "id": r[0], "content": r[1], "timestamp": r[2], "source": r[3],
                        "reinforcement_count": r[4] or 1,
                        "last_reinforced_at": r[5],
                        "title": r[6] or "",
                        "facts": _json_load_list(r[7]),
                        "concepts": _json_load_list(r[8]),
                        "similarity": float(r[9]),
                    }
                    for r in result.fetchall()
                ]

        # Sqlite fallback — no vector search, return all by recency with
        # synthetic similarity = 0.5 so hybrid scoring still differentiates.
        async with self.session_maker() as session:
            result = await session.execute(
                select(SemanticMemory)
                .order_by(SemanticMemory.last_reinforced_at.desc().nullslast())
                .limit(limit)
            )
            rows = result.scalars().all()
        return [
            {
                "id": r.id, "content": r.content, "timestamp": r.timestamp, "source": r.source,
                "reinforcement_count": r.reinforcement_count or 1,
                "last_reinforced_at": r.last_reinforced_at,
                "title": r.title or "",
                "facts": _json_load_list(r.facts_json),
                "concepts": _json_load_list(r.concepts_json),
                "similarity": 0.5,
            }
            for r in rows
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
