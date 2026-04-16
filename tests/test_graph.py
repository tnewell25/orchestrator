"""GraphStore correctness — edge writes (with reinforcement), neighbors,
subgraph BFS bounded by depth + node cap, shortest path, proximity scoring."""
import pytest

from orchestrator.core.constants import EdgeKind, EntityType
from orchestrator.core.graph import EntityRef, GraphStore


@pytest.mark.asyncio
async def test_add_edge_creates_row(session_maker):
    g = GraphStore(session_maker)
    a = EntityRef(EntityType.MEMORY, "m1")
    b = EntityRef(EntityType.CONTACT, "c1")

    edge = await g.add_edge(a, b, EdgeKind.MENTIONS)
    assert edge.from_ref == a
    assert edge.to_ref == b
    assert edge.kind == EdgeKind.MENTIONS
    assert edge.reinforcement_count == 1


@pytest.mark.asyncio
async def test_add_edge_is_idempotent_and_reinforces(session_maker):
    g = GraphStore(session_maker)
    a = EntityRef(EntityType.MEMORY, "m1")
    b = EntityRef(EntityType.CONTACT, "c1")

    await g.add_edge(a, b, EdgeKind.MENTIONS)
    await g.add_edge(a, b, EdgeKind.MENTIONS)
    third = await g.add_edge(a, b, EdgeKind.MENTIONS, confidence=0.95)

    assert third.reinforcement_count == 3
    assert third.confidence == pytest.approx(1.0)  # max(1.0, 0.95)


@pytest.mark.asyncio
async def test_neighbors_bidirectional_by_default(session_maker):
    g = GraphStore(session_maker)
    mem = EntityRef(EntityType.MEMORY, "m1")
    contact = EntityRef(EntityType.CONTACT, "c1")
    deal = EntityRef(EntityType.DEAL, "d1")

    await g.add_edge(mem, contact, EdgeKind.MENTIONS)        # mem → contact
    await g.add_edge(contact, deal, EdgeKind.STAKEHOLDER_IN) # contact → deal

    # Contact should see both (incoming from memory + outgoing to deal)
    edges = await g.neighbors(contact)
    assert len(edges) == 2
    kinds = sorted(e.kind for e in edges)
    assert kinds == [EdgeKind.MENTIONS, EdgeKind.STAKEHOLDER_IN]


@pytest.mark.asyncio
async def test_neighbors_filtered_by_kind(session_maker):
    g = GraphStore(session_maker)
    mem = EntityRef(EntityType.MEMORY, "m1")
    contact = EntityRef(EntityType.CONTACT, "c1")
    deal = EntityRef(EntityType.DEAL, "d1")

    await g.add_edge(mem, contact, EdgeKind.MENTIONS)
    await g.add_edge(mem, deal, EdgeKind.REFERENCES)

    only_mentions = await g.neighbors(mem, kinds=[EdgeKind.MENTIONS])
    assert len(only_mentions) == 1
    assert only_mentions[0].to_ref == contact


@pytest.mark.asyncio
async def test_subgraph_bfs_bounded_by_depth(session_maker):
    g = GraphStore(session_maker)
    a = EntityRef(EntityType.MEMORY, "m1")
    b = EntityRef(EntityType.CONTACT, "c1")
    c = EntityRef(EntityType.DEAL, "d1")
    d = EntityRef(EntityType.MEETING, "mt1")

    await g.add_edge(a, b, EdgeKind.MENTIONS)        # depth 1 from a
    await g.add_edge(b, c, EdgeKind.STAKEHOLDER_IN)  # depth 2 from a
    await g.add_edge(c, d, EdgeKind.ABOUT)           # depth 3 from a — should be excluded at depth=2

    sg = await g.subgraph(a, max_depth=2, max_nodes=10)

    refs = sg.nodes
    assert a in refs and b in refs and c in refs
    assert d not in refs


@pytest.mark.asyncio
async def test_subgraph_respects_node_cap(session_maker):
    g = GraphStore(session_maker)
    root = EntityRef(EntityType.DEAL, "deal-1")

    # Wide fan-out — one deal connected to 20 contacts
    for i in range(20):
        await g.add_edge(
            EntityRef(EntityType.CONTACT, f"c{i}"), root, EdgeKind.STAKEHOLDER_IN
        )

    sg = await g.subgraph(root, max_depth=2, max_nodes=5)
    assert len(sg.nodes) <= 5


@pytest.mark.asyncio
async def test_shortest_path_finds_link(session_maker):
    g = GraphStore(session_maker)
    a = EntityRef(EntityType.CONTACT, "anja")
    deal = EntityRef(EntityType.DEAL, "bosch-deal")
    markus = EntityRef(EntityType.CONTACT, "markus")

    await g.add_edge(a, deal, EdgeKind.STAKEHOLDER_IN)
    await g.add_edge(markus, deal, EdgeKind.STAKEHOLDER_IN)

    path = await g.shortest_path(a, markus, max_hops=3)
    assert path is not None
    assert len(path) == 2
    # Edges should chain: a→deal, then deal↔markus
    assert any(e.from_ref == a or e.to_ref == a for e in path)
    assert any(e.from_ref == markus or e.to_ref == markus for e in path)


@pytest.mark.asyncio
async def test_shortest_path_returns_none_when_unreachable(session_maker):
    g = GraphStore(session_maker)
    a = EntityRef(EntityType.CONTACT, "a")
    b = EntityRef(EntityType.CONTACT, "b")
    # No edges connecting them
    path = await g.shortest_path(a, b)
    assert path is None


@pytest.mark.asyncio
async def test_proximity_score(session_maker):
    g = GraphStore(session_maker)
    a = EntityRef(EntityType.CONTACT, "a")
    b = EntityRef(EntityType.DEAL, "d")
    c = EntityRef(EntityType.CONTACT, "c")
    isolated = EntityRef(EntityType.CONTACT, "z")

    await g.add_edge(a, b, EdgeKind.STAKEHOLDER_IN)
    await g.add_edge(b, c, EdgeKind.STAKEHOLDER_IN)

    assert await g.proximity(a, a) == 1.0
    assert await g.proximity(a, b) == pytest.approx(0.5)   # 1 hop
    assert 0.3 < await g.proximity(a, c) < 0.4              # 2 hops → 1/3
    assert await g.proximity(a, isolated) == 0.0


@pytest.mark.asyncio
async def test_remove_edge(session_maker):
    g = GraphStore(session_maker)
    a = EntityRef(EntityType.MEMORY, "m")
    b = EntityRef(EntityType.CONTACT, "c")
    await g.add_edge(a, b, EdgeKind.MENTIONS)

    removed = await g.remove_edge(a, b, EdgeKind.MENTIONS)
    assert removed is True

    edges = await g.neighbors(a)
    assert edges == []

    # Removing again is a no-op
    assert await g.remove_edge(a, b, EdgeKind.MENTIONS) is False
