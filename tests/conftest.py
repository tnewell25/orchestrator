"""Shared fixtures — async sqlite session_maker for graph + entity tests.

We use sqlite (no pgvector) because these layers don't need embeddings —
graph traversal, edge writes, and entity-name extraction are all backend-agnostic.
The MemoryStore semantic-recall layer is integration-tested separately against
real Postgres in CI.
"""
import asyncio

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Ensure the package can be imported when pytest is run from repo root.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))  # so `import orchestrator` works
PACKAGE_NAME = ROOT.name


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def session_maker():
    """Fresh in-memory sqlite per test. Builds the full schema."""
    # Local import so the package metadata module is loaded under whatever
    # name the parent dir uses (works for both `orchestrator` and `repos.orchestrator`).
    from orchestrator.db.models import Base

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    yield sm
    await engine.dispose()
