"""Knowledge graph layer — typed edges over CRM entities + semantic memories.

Wraps the `edges` table with traversal helpers used by hybrid retrieval, the
planner, and skills that need cross-entity context (e.g. "everything about
Bosch"). Storage is directional; traversal is bidirectional by default.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..db.models import Edge
from .constants import (
    MAX_GRAPH_TRAVERSAL_DEPTH,
    MAX_SUBGRAPH_NODES,
    EdgeKind,
    EntityType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityRef:
    """Address of any node in the graph."""
    type: str
    id: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.type, self.id)

    def __str__(self) -> str:
        return f"{self.type}:{self.id}"


@dataclass
class EdgeRow:
    """In-memory edge record returned by traversal."""
    from_ref: EntityRef
    to_ref: EntityRef
    kind: str
    weight: float = 1.0
    confidence: float = 1.0
    reinforcement_count: int = 1
    source: str = "auto_extract"
    last_reinforced_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Edge) -> "EdgeRow":
        return cls(
            from_ref=EntityRef(row.from_type, row.from_id),
            to_ref=EntityRef(row.to_type, row.to_id),
            kind=row.kind,
            weight=row.weight or 1.0,
            confidence=row.confidence or 1.0,
            reinforcement_count=row.reinforcement_count or 1,
            source=row.source or "auto_extract",
            last_reinforced_at=row.last_reinforced_at,
        )


@dataclass
class Subgraph:
    """A bounded slice of the graph rooted at a focus entity."""
    root: EntityRef
    nodes: set[EntityRef] = field(default_factory=set)
    edges: list[EdgeRow] = field(default_factory=list)

    def neighbors_of(self, ref: EntityRef) -> list[EntityRef]:
        out = []
        for e in self.edges:
            if e.from_ref == ref:
                out.append(e.to_ref)
            elif e.to_ref == ref:
                out.append(e.from_ref)
        return out

    def to_dict(self) -> dict:
        return {
            "root": str(self.root),
            "nodes": [str(n) for n in self.nodes],
            "edges": [
                {
                    "from": str(e.from_ref),
                    "to": str(e.to_ref),
                    "kind": e.kind,
                    "weight": e.weight,
                    "reinforcement": e.reinforcement_count,
                }
                for e in self.edges
            ],
        }


class GraphStore:
    """Async wrapper around the edges table.

    Reuses the project's async session_maker so it shares the connection pool
    with MemoryStore/skills. No state of its own.
    """

    def __init__(self, session_maker: async_sessionmaker):
        self.sm = session_maker

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    async def add_edge(
        self,
        from_ref: EntityRef,
        to_ref: EntityRef,
        kind: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        source: str = "auto_extract",
    ) -> EdgeRow:
        """Insert an edge or reinforce the existing one.

        Reinforcement: if the same (from, to, kind) tuple already exists, bump
        reinforcement_count + last_reinforced_at and take the MAX of the
        confidence values. This is what hardens repeated facts into trust.
        """
        async with self.sm() as session:
            existing = (
                await session.execute(
                    select(Edge).where(
                        Edge.from_type == from_ref.type,
                        Edge.from_id == from_ref.id,
                        Edge.to_type == to_ref.type,
                        Edge.to_id == to_ref.id,
                        Edge.kind == kind,
                    )
                )
            ).scalar_one_or_none()

            if existing:
                existing.reinforcement_count = (existing.reinforcement_count or 1) + 1
                existing.last_reinforced_at = datetime.now(timezone.utc)
                existing.confidence = max(existing.confidence or 0.0, confidence)
                existing.weight = max(existing.weight or 0.0, weight)
                await session.commit()
                await session.refresh(existing)
                return EdgeRow.from_row(existing)

            row = Edge(
                from_type=from_ref.type,
                from_id=from_ref.id,
                to_type=to_ref.type,
                to_id=to_ref.id,
                kind=kind,
                weight=weight,
                confidence=confidence,
                source=source,
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                # Race: another writer inserted the same edge between SELECT and INSERT.
                # Re-fetch and reinforce instead of crashing.
                await session.rollback()
                return await self.add_edge(
                    from_ref, to_ref, kind, weight=weight,
                    confidence=confidence, source=source,
                )
            await session.refresh(row)
            return EdgeRow.from_row(row)

    async def add_edges_bulk(self, edges: Iterable[tuple[EntityRef, EntityRef, str]]) -> int:
        """Insert many edges — reuses add_edge so each gets reinforcement semantics.

        Not the fastest path for large batches, but correctness > speed for now.
        """
        n = 0
        for from_ref, to_ref, kind in edges:
            await self.add_edge(from_ref, to_ref, kind)
            n += 1
        return n

    async def remove_edge(
        self, from_ref: EntityRef, to_ref: EntityRef, kind: str
    ) -> bool:
        async with self.sm() as session:
            row = (
                await session.execute(
                    select(Edge).where(
                        Edge.from_type == from_ref.type,
                        Edge.from_id == from_ref.id,
                        Edge.to_type == to_ref.type,
                        Edge.to_id == to_ref.id,
                        Edge.kind == kind,
                    )
                )
            ).scalar_one_or_none()
            if not row:
                return False
            await session.delete(row)
            await session.commit()
            return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def neighbors(
        self,
        ref: EntityRef,
        kinds: Iterable[str] | None = None,
        directions: tuple[str, ...] = ("out", "in"),
    ) -> list[EdgeRow]:
        """All immediate neighbors of `ref`. Bidirectional by default."""
        kinds_list = list(kinds) if kinds else None

        clauses = []
        if "out" in directions:
            clauses.append(and_(Edge.from_type == ref.type, Edge.from_id == ref.id))
        if "in" in directions:
            clauses.append(and_(Edge.to_type == ref.type, Edge.to_id == ref.id))
        if not clauses:
            return []

        q = select(Edge).where(or_(*clauses))
        if kinds_list:
            q = q.where(Edge.kind.in_(kinds_list))

        async with self.sm() as session:
            result = await session.execute(q)
            return [EdgeRow.from_row(r) for r in result.scalars().all()]

    async def subgraph(
        self,
        root: EntityRef,
        max_depth: int = MAX_GRAPH_TRAVERSAL_DEPTH,
        max_nodes: int = MAX_SUBGRAPH_NODES,
        kinds: Iterable[str] | None = None,
    ) -> Subgraph:
        """BFS expansion from root, bounded by depth + node count.

        Used by "what's the full Bosch story?" — one call returns the whole
        connected slice instead of N sequential tool calls.
        """
        kinds_list = list(kinds) if kinds else None
        sg = Subgraph(root=root)
        sg.nodes.add(root)

        seen_edges: set[tuple] = set()
        queue: deque[tuple[EntityRef, int]] = deque([(root, 0)])

        while queue and len(sg.nodes) < max_nodes:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue

            edges = await self.neighbors(node, kinds=kinds_list)
            for e in edges:
                key = (e.from_ref.as_tuple(), e.to_ref.as_tuple(), e.kind)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                sg.edges.append(e)

                other = e.to_ref if e.from_ref == node else e.from_ref
                if other not in sg.nodes:
                    sg.nodes.add(other)
                    queue.append((other, depth + 1))
                    if len(sg.nodes) >= max_nodes:
                        break

        return sg

    async def shortest_path(
        self,
        from_ref: EntityRef,
        to_ref: EntityRef,
        max_hops: int = 4,
    ) -> list[EdgeRow] | None:
        """BFS shortest path. Returns None if no path within max_hops.

        Useful for "how do I know this person?" — finds the chain of edges
        connecting two contacts via mutual deals/meetings.
        """
        if from_ref == to_ref:
            return []

        # parent[ref] = (predecessor_ref, edge_used_to_reach_it)
        parent: dict[EntityRef, tuple[EntityRef, EdgeRow]] = {from_ref: (from_ref, None)}
        queue: deque[tuple[EntityRef, int]] = deque([(from_ref, 0)])

        while queue:
            node, hops = queue.popleft()
            if hops >= max_hops:
                continue

            edges = await self.neighbors(node)
            for e in edges:
                other = e.to_ref if e.from_ref == node else e.from_ref
                if other in parent:
                    continue
                parent[other] = (node, e)
                if other == to_ref:
                    # Reconstruct path
                    path: list[EdgeRow] = []
                    cur = to_ref
                    while parent[cur][0] != cur:
                        prev, edge = parent[cur]
                        path.append(edge)
                        cur = prev
                    path.reverse()
                    return path
                queue.append((other, hops + 1))

        return None

    async def proximity(
        self,
        a: EntityRef,
        b: EntityRef,
        max_hops: int = 3,
    ) -> float:
        """Inverse-hop similarity in [0, 1]. Used by hybrid retrieval scoring.

        - 1.0 if same node
        - 1 / (1 + hops) for connected nodes within max_hops
        - 0.0 if no path within budget
        """
        if a == b:
            return 1.0
        path = await self.shortest_path(a, b, max_hops=max_hops)
        if path is None:
            return 0.0
        return 1.0 / (1.0 + len(path))


__all__ = ["EntityRef", "EdgeRow", "Subgraph", "GraphStore", "EdgeKind", "EntityType"]
