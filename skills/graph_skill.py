"""GraphSkill — exposes the knowledge graph to the agent as callable tools.

The headline win: graph-context_for(entity_type, entity_id) returns a single
subgraph slice (deal + stakeholders + meetings + memories + emails) instead of
forcing the agent to chain 5+ sequential tool calls.
"""
from __future__ import annotations

from sqlalchemy import select

from ..core.constants import EdgeKind, EntityType, MAX_GRAPH_TRAVERSAL_DEPTH
from ..core.graph import EntityRef, GraphStore
from ..core.skill_base import Skill, tool
from ..db.models import Company, Contact, Deal, Meeting


class GraphSkill(Skill):
    name = "graph"
    description = (
        "Cross-entity relationship queries. Use graph-context_for to get a "
        "complete picture of any deal/contact/company in one call."
    )

    def __init__(self, session_maker, graph: GraphStore):
        super().__init__()
        self.session_maker = session_maker
        self.graph = graph

    @tool(
        "Get a subgraph slice rooted at one entity — returns all connected "
        "memories, contacts, deals, meetings within max_depth hops. "
        "entity_type must be one of: Memory, Contact, Deal, Company, Meeting. "
        "Use this INSTEAD of chaining find + get_context + list_meetings calls."
    )
    async def context_for(
        self,
        entity_type: str,
        entity_id: str,
        max_depth: int = 2,
        max_nodes: int = 30,
    ) -> dict:
        if entity_type not in EntityType.ALL:
            return {"error": f"unknown entity_type {entity_type!r}"}

        root = EntityRef(entity_type, entity_id)
        sg = await self.graph.subgraph(root, max_depth=max_depth, max_nodes=max_nodes)

        # Hydrate node labels — the raw subgraph only knows (type, id);
        # the agent wants names so it can present a coherent narrative.
        labels = await self._label_nodes(sg.nodes)

        return {
            "root": str(root),
            "node_count": len(sg.nodes),
            "edge_count": len(sg.edges),
            "nodes": [
                {"ref": str(n), "label": labels.get(n, "")}
                for n in sg.nodes
            ],
            "edges": [
                {
                    "from": str(e.from_ref),
                    "to": str(e.to_ref),
                    "kind": e.kind,
                    "reinforcement": e.reinforcement_count,
                    "confidence": round(e.confidence, 2),
                }
                for e in sg.edges
            ],
        }

    @tool(
        "Find the chain of relationships connecting two entities ('how do I "
        "know X via Y'). Returns the edges in the shortest path, or null if "
        "no path within 4 hops."
    )
    async def path_between(
        self,
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
    ) -> dict:
        a = EntityRef(from_type, from_id)
        b = EntityRef(to_type, to_id)
        path = await self.graph.shortest_path(a, b, max_hops=4)
        if path is None:
            return {"path": None, "hops": None}
        return {
            "from": str(a),
            "to": str(b),
            "hops": len(path),
            "path": [
                {"from": str(e.from_ref), "to": str(e.to_ref), "kind": e.kind}
                for e in path
            ],
        }

    @tool(
        "List all neighbors of an entity (one hop only). Useful for 'who else "
        "is involved in this deal' or 'what deals mention this person'. "
        "Optional kinds filter: comma-separated edge kinds."
    )
    async def neighbors(
        self,
        entity_type: str,
        entity_id: str,
        kinds: str = "",
    ) -> dict:
        ref = EntityRef(entity_type, entity_id)
        kind_list = [k.strip() for k in kinds.split(",") if k.strip()] or None
        edges = await self.graph.neighbors(ref, kinds=kind_list)
        # Resolve the OTHER endpoint per edge so the response is symmetric
        others = {
            (e.to_ref if e.from_ref == ref else e.from_ref) for e in edges
        }
        labels = await self._label_nodes(others)
        return {
            "root": str(ref),
            "neighbors": [
                {
                    "ref": str(e.to_ref if e.from_ref == ref else e.from_ref),
                    "label": labels.get(
                        (e.to_ref if e.from_ref == ref else e.from_ref), ""
                    ),
                    "via_kind": e.kind,
                    "reinforcement": e.reinforcement_count,
                }
                for e in edges
            ],
        }

    async def _label_nodes(self, refs) -> dict:
        """Bulk-fetch human-readable labels for a set of refs.

        One query per type to keep this O(types) not O(refs).
        """
        by_type: dict[str, list[str]] = {}
        for r in refs:
            by_type.setdefault(r.type, []).append(r.id)

        labels: dict = {}
        async with self.session_maker() as s:
            for t, ids in by_type.items():
                if not ids:
                    continue
                model = {
                    EntityType.CONTACT: Contact,
                    EntityType.COMPANY: Company,
                    EntityType.DEAL: Deal,
                    EntityType.MEETING: Meeting,
                }.get(t)
                if model is None:
                    # Memories/Emails/etc. — no name; use the truncated id
                    for i in ids:
                        labels[EntityRef(t, i)] = f"{t}:{i[:8]}"
                    continue
                rows = (
                    await s.execute(
                        select(model.id, model.name if hasattr(model, "name") else model.id)
                        .where(model.id.in_(ids))
                    )
                ).all()
                for rid, rname in rows:
                    labels[EntityRef(t, rid)] = rname or f"{t}:{rid[:8]}"
        return labels
