"""End-to-end tests for dashboard write endpoints.

Exercises the FastAPI router against an in-memory sqlite to confirm round-trips
hit the same tables the bot reads/writes.
"""
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from orchestrator.interfaces.dashboard_api import mount_dashboard_api


@pytest_asyncio.fixture
async def client(session_maker):
    app = FastAPI()
    mount_dashboard_api(app, session_maker)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_create_and_patch_deal(client, session_maker):
    from orchestrator.db.models import Deal

    r = await client.post("/api/dashboard/deals", json={
        "name": "Bosch Forge", "stage": "proposal", "value_usd": 500_000,
        "next_step": "Send pricing", "competitors": "Siemens",
    })
    assert r.status_code == 200
    deal_id = r.json()["id"]

    r = await client.patch(f"/api/dashboard/deals/{deal_id}", json={
        "stage": "negotiation", "metrics": "20% downtime reduction",
        "notes_append": "Champion confirmed budget approved",
    })
    assert r.status_code == 200

    async with session_maker() as s:
        d = await s.get(Deal, deal_id)
        assert d.stage == "negotiation"
        assert d.metrics == "20% downtime reduction"
        assert "Champion confirmed budget approved" in d.notes


@pytest.mark.asyncio
async def test_invalid_stage_rejected(client):
    r = await client.post("/api/dashboard/deals", json={"name": "X", "stage": "fake"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_create_action_and_toggle_done(client, session_maker):
    from orchestrator.db.models import ActionItem, Deal

    deal = (await client.post("/api/dashboard/deals", json={"name": "X"})).json()
    a = (await client.post(f"/api/dashboard/deals/{deal['id']}/actions", json={
        "description": "Email Brian re NDA", "due_date": "2026-04-25",
    })).json()
    assert a["status"] == "open"

    r = await client.patch(f"/api/dashboard/actions/{a['id']}", json={"status": "done"})
    assert r.status_code == 200

    async with session_maker() as s:
        ai = await s.get(ActionItem, a["id"])
        assert ai.status == "done"
        assert ai.completed_at is not None


@pytest.mark.asyncio
async def test_stakeholder_create_and_patch(client, session_maker):
    from orchestrator.db.models import DealStakeholder

    deal = (await client.post("/api/dashboard/deals", json={"name": "X"})).json()
    contact = (await client.post("/api/dashboard/contacts", json={"name": "Brian"})).json()

    sh = (await client.post(
        f"/api/dashboard/deals/{deal['id']}/stakeholders",
        json={"contact_id": contact["id"], "role": "champion", "sentiment": "supportive"},
    )).json()
    assert "id" in sh

    r = await client.patch(f"/api/dashboard/stakeholders/{sh['id']}", json={
        "sentiment": "neutral", "influence": "high",
    })
    assert r.status_code == 200

    async with session_maker() as s:
        st = await s.get(DealStakeholder, sh["id"])
        assert st.sentiment == "neutral"
        assert st.influence == "high"


@pytest.mark.asyncio
async def test_contact_patch(client, session_maker):
    from orchestrator.db.models import Contact

    c = (await client.post("/api/dashboard/contacts", json={"name": "Lena"})).json()
    r = await client.patch(f"/api/dashboard/contacts/{c['id']}", json={
        "personal_notes": "Two kids — soccer Saturday mornings",
        "title": "VP Engineering",
    })
    assert r.status_code == 200

    async with session_maker() as s:
        contact = await s.get(Contact, c["id"])
        assert "soccer" in contact.personal_notes
        assert contact.title == "VP Engineering"


@pytest.mark.asyncio
async def test_audit_log_written(client, session_maker):
    from sqlalchemy import select
    from orchestrator.db.models import AuditLog

    await client.post("/api/dashboard/deals", json={"name": "Audited"})
    async with session_maker() as s:
        rows = (await s.execute(
            select(AuditLog).where(AuditLog.tool_name == "dashboard:deal.create")
        )).scalars().all()
        assert len(rows) == 1
        assert rows[0].session_id == "dashboard"
