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


# ---------------------------------------------------------------------
# PR1 — universal delete + companies detail + bids CRUD + meetings CRUD
# + extended industrial stakeholder roles
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_deal(client, session_maker):
    from orchestrator.db.models import Deal
    deal_id = (await client.post("/api/dashboard/deals", json={"name": "Doomed"})).json()["id"]
    r = await client.delete(f"/api/dashboard/deals/{deal_id}")
    assert r.status_code == 200
    async with session_maker() as s:
        assert await s.get(Deal, deal_id) is None


@pytest.mark.asyncio
async def test_delete_404(client):
    r = await client.delete("/api/dashboard/deals/does-not-exist")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_each_entity(client):
    """Smoke each delete route — ensures every model class resolves correctly."""
    contact_id = (await client.post("/api/dashboard/contacts", json={"name": "X"})).json()["id"]
    company_id = (await client.post("/api/dashboard/companies", json={"name": "Y"})).json()["id"]
    deal_id = (await client.post("/api/dashboard/deals", json={"name": "Z"})).json()["id"]
    action_id = (await client.post(f"/api/dashboard/deals/{deal_id}/actions",
                                   json={"description": "do thing"})).json()["id"]
    meeting_id = (await client.post(f"/api/dashboard/deals/{deal_id}/meetings",
                                    json={"summary": "kickoff"})).json()["id"]
    bid_id = (await client.post("/api/dashboard/bids", json={"name": "RFP-001"})).json()["id"]

    for path in [
        f"/api/dashboard/actions/{action_id}",
        f"/api/dashboard/meetings/{meeting_id}",
        f"/api/dashboard/bids/{bid_id}",
        f"/api/dashboard/deals/{deal_id}",
        f"/api/dashboard/contacts/{contact_id}",
        f"/api/dashboard/companies/{company_id}",
    ]:
        r = await client.delete(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"


@pytest.mark.asyncio
async def test_company_detail_rolls_up(client):
    co = (await client.post("/api/dashboard/companies", json={"name": "Honeywell"})).json()
    deal = (await client.post("/api/dashboard/deals", json={
        "name": "DCS Migration", "company_id": co["id"], "value_usd": 750_000,
    })).json()
    await client.post("/api/dashboard/contacts",
                      json={"name": "Lena", "company_id": co["id"]})
    await client.post("/api/dashboard/bids",
                      json={"name": "RFP-2026-Q2", "company_id": co["id"], "value_usd": 200_000})
    await client.post(f"/api/dashboard/deals/{deal['id']}/actions",
                      json={"description": "Send pricing"})

    r = await client.get(f"/api/dashboard/companies/{co['id']}")
    assert r.status_code == 200
    body = r.json()
    assert body["company"]["name"] == "Honeywell"
    assert body["stats"]["deal_count"] == 1
    assert body["stats"]["active_pipeline_value"] == 750_000
    assert body["stats"]["contact_count"] == 1
    assert body["stats"]["open_bid_count"] == 1
    assert len(body["deals"]) == 1
    assert len(body["bids"]) == 1
    assert len(body["recent_actions"]) == 1


@pytest.mark.asyncio
async def test_company_patch(client, session_maker):
    from orchestrator.db.models import Company
    co = (await client.post("/api/dashboard/companies", json={"name": "ACME"})).json()
    r = await client.patch(f"/api/dashboard/companies/{co['id']}", json={
        "industry": "Industrial Automation", "website": "acme.com",
    })
    assert r.status_code == 200
    async with session_maker() as s:
        c = await s.get(Company, co["id"])
        assert c.industry == "Industrial Automation"
        assert c.website == "acme.com"


@pytest.mark.asyncio
async def test_bid_create_list_patch(client):
    co = (await client.post("/api/dashboard/companies", json={"name": "Bosch"})).json()
    bid = (await client.post("/api/dashboard/bids", json={
        "name": "Forge Line Controls", "company_id": co["id"],
        "stage": "in_progress", "value_usd": 1_500_000,
        "submission_deadline": "2026-05-01",
        "deliverables": "Tech proposal, BOM, schedule",
    })).json()

    r = await client.get("/api/dashboard/bids")
    assert r.status_code == 200
    bids = r.json()["bids"]
    assert any(b["id"] == bid["id"] and b["company"] == "Bosch" for b in bids)

    r = await client.get("/api/dashboard/bids", params={"stage": "in_progress"})
    assert all(b["stage"] == "in_progress" for b in r.json()["bids"])

    r = await client.patch(f"/api/dashboard/bids/{bid['id']}", json={"stage": "submitted"})
    assert r.status_code == 200

    r = await client.get(f"/api/dashboard/bids/{bid['id']}")
    assert r.json()["bid"]["stage"] == "submitted"
    assert r.json()["bid"]["deliverables"] == "Tech proposal, BOM, schedule"


@pytest.mark.asyncio
async def test_bid_invalid_stage(client):
    r = await client.post("/api/dashboard/bids", json={"name": "X", "stage": "invalid"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_meeting_create_patch(client, session_maker):
    from orchestrator.db.models import Meeting
    deal = (await client.post("/api/dashboard/deals", json={"name": "X"})).json()
    m = (await client.post(f"/api/dashboard/deals/{deal['id']}/meetings", json={
        "attendees": "Lena, Brian", "summary": "Kickoff with controls team",
    })).json()
    r = await client.patch(f"/api/dashboard/meetings/{m['id']}", json={
        "decisions": "Move to FAT in May",
    })
    assert r.status_code == 200
    async with session_maker() as s:
        meeting = await s.get(Meeting, m["id"])
        assert meeting.decisions == "Move to FAT in May"
        assert "Lena" in meeting.attendees


@pytest.mark.asyncio
async def test_reminder_patch_done(client, session_maker):
    """Marking a sent reminder as done removes it from the inbox."""
    from datetime import datetime, timezone
    from orchestrator.db.models import Reminder

    async with session_maker() as s:
        r = Reminder(
            trigger_at=datetime.now(timezone.utc), message="Re-engage Bosch",
            status="sent",
        )
        s.add(r)
        await s.commit()
        await s.refresh(r)
        rid = r.id

    inbox = (await client.get("/api/dashboard/inbox")).json()
    assert any(it["id"] == rid for it in inbox["items"])

    resp = await client.patch(f"/api/dashboard/reminders/{rid}", json={"status": "done"})
    assert resp.status_code == 200

    inbox = (await client.get("/api/dashboard/inbox")).json()
    assert not any(it["id"] == rid for it in inbox["items"])


@pytest.mark.asyncio
async def test_reminder_snooze_pushes_trigger_forward(client, session_maker):
    from datetime import datetime, timedelta, timezone
    from orchestrator.db.models import Reminder

    async with session_maker() as s:
        r = Reminder(
            trigger_at=datetime.now(timezone.utc) - timedelta(hours=2),
            message="Old reminder", status="sent",
        )
        s.add(r)
        await s.commit()
        await s.refresh(r)
        rid = r.id
        original = r.trigger_at

    resp = await client.post(f"/api/dashboard/reminders/{rid}/snooze", json={"hours": 48})
    assert resp.status_code == 200

    async with session_maker() as s:
        rem = await s.get(Reminder, rid)
        assert rem.status == "pending"
        assert rem.sent_at is None
        # Trigger pushed at least ~47h forward (allow tolerance for clock)
        assert (rem.trigger_at - original).total_seconds() > 47 * 3600


@pytest.mark.asyncio
async def test_inbox_dedupes_duplicate_reminders(client, session_maker):
    """The Pipeline Watcher firing two near-identical alerts for the same
    deal in the same window should collapse to one inbox row with a
    dup_count, not two rows the user has to dismiss separately."""
    from datetime import datetime, timezone
    from orchestrator.db.models import Deal, Reminder

    async with session_maker() as s:
        d = Deal(name="Bosch Pilot")
        s.add(d)
        await s.commit()
        await s.refresh(d)
        dup_text = "Re-engage Bosch — 465 days stale, critical MEDDIC gaps"
        for _ in range(3):
            s.add(Reminder(
                trigger_at=datetime.now(timezone.utc),
                message=dup_text, status="sent", related_deal_id=d.id,
            ))
        await s.commit()

    inbox = (await client.get("/api/dashboard/inbox")).json()
    bosch_items = [it for it in inbox["items"] if it.get("deal_id") == d.id]
    assert len(bosch_items) == 1, f"expected 1 deduped item, got {len(bosch_items)}"
    assert bosch_items[0]["dup_count"] == 3


@pytest.mark.asyncio
async def test_inbox_includes_action_items_due_soon(client, session_maker):
    """Action items with a due_date in the next 7 days surface in the inbox
    so the user doesn't have to dig into each deal page."""
    from datetime import date, timedelta
    from orchestrator.db.models import ActionItem, Deal

    async with session_maker() as s:
        d = Deal(name="Honeywell")
        s.add(d)
        await s.commit()
        await s.refresh(d)
        s.add(ActionItem(
            deal_id=d.id, description="Send pricing",
            due_date=date.today() + timedelta(days=2), status="open",
        ))
        # An item due far in the future should NOT appear
        s.add(ActionItem(
            deal_id=d.id, description="Q3 follow-up",
            due_date=date.today() + timedelta(days=60), status="open",
        ))
        await s.commit()

    inbox = (await client.get("/api/dashboard/inbox")).json()
    descs = [it["title"] for it in inbox["items"] if it["kind"] == "action_item"]
    assert "Send pricing" in descs
    assert "Q3 follow-up" not in descs


@pytest.mark.asyncio
async def test_inbox_includes_pending_actions(client, session_maker):
    from orchestrator.db.models import PendingAction

    async with session_maker() as s:
        p = PendingAction(
            session_id="dashboard", tool_name="email.send",
            tool_input='{"to":"x@y"}', summary="Send re-engage to Bosch champion",
            status="pending",
        )
        s.add(p)
        await s.commit()

    inbox = (await client.get("/api/dashboard/inbox")).json()
    pa = [it for it in inbox["items"] if it["kind"] == "pending_action"]
    assert len(pa) == 1
    assert pa[0]["tool_name"] == "email.send"


@pytest.mark.asyncio
async def test_plant_crud_and_company_rollup(client, session_maker):
    """Plants are first-class — a deal at Bosch Stuttgart is not the same
    as a deal at Bosch Mexico. The company rollup must include plants so
    account planning happens at the right level."""
    from orchestrator.db.models import Plant

    co = (await client.post("/api/dashboard/companies", json={"name": "Bosch"})).json()
    p1 = (await client.post("/api/dashboard/plants", json={
        "name": "Stuttgart Forge", "company_id": co["id"],
        "site_address": "Stuttgart, DE", "site_type": "manufacturing",
    })).json()
    p2 = (await client.post("/api/dashboard/plants", json={
        "name": "San Luis Potosi", "company_id": co["id"],
        "site_type": "manufacturing",
    })).json()

    # Plant detail rolls up deals + bids at this site
    deal = (await client.post("/api/dashboard/deals", json={
        "name": "Stuttgart DCS upgrade", "company_id": co["id"],
        "plant_id": p1["id"], "value_usd": 1_200_000,
    })).json()
    detail = (await client.get(f"/api/dashboard/plants/{p1['id']}")).json()
    assert detail["plant"]["name"] == "Stuttgart Forge"
    assert len(detail["deals"]) == 1
    assert detail["deals"][0]["id"] == deal["id"]

    # Company rollup includes plants
    co_detail = (await client.get(f"/api/dashboard/companies/{co['id']}")).json()
    assert co_detail["stats"]["plant_count"] == 2
    plant_names = {p["name"] for p in co_detail["plants"]}
    assert plant_names == {"Stuttgart Forge", "San Luis Potosi"}

    # Patch + delete
    await client.patch(f"/api/dashboard/plants/{p2['id']}", json={"site_type": "chemical"})
    async with session_maker() as s:
        plant = await s.get(Plant, p2["id"])
        assert plant.site_type == "chemical"

    r = await client.delete(f"/api/dashboard/plants/{p2['id']}")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_invalid_site_type_rejected(client):
    co = (await client.post("/api/dashboard/companies", json={"name": "X"})).json()
    r = await client.post("/api/dashboard/plants", json={
        "name": "Bad", "company_id": co["id"], "site_type": "invalid",
    })
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_specs_library(client):
    sp = (await client.post("/api/dashboard/specs", json={
        "code": "ATEX-Zone1", "name": "ATEX Zone 1 (gas, intermittent)",
        "family": "hazardous_area",
        "scope": "Equipment for areas where explosive atmospheres likely occur in normal operation",
    })).json()
    assert sp["code"] == "ATEX-Zone1"

    # Duplicate code rejected
    dup = await client.post("/api/dashboard/specs", json={
        "code": "ATEX-Zone1", "name": "dup",
    })
    assert dup.status_code == 409

    # Family filter
    r = (await client.get("/api/dashboard/specs", params={"family": "hazardous_area"})).json()
    codes = [s["code"] for s in r["specs"]]
    assert "ATEX-Zone1" in codes


@pytest.mark.asyncio
async def test_compliance_matrix_lifecycle(client, session_maker):
    """The compliance matrix is the bid-scoring document. Procurement uses
    it to disqualify, so structured tracking of every clause is the moat."""
    from orchestrator.db.models import ComplianceMatrixItem

    bid = (await client.post("/api/dashboard/bids", json={"name": "RFP-2026-Q2"})).json()
    spec = (await client.post("/api/dashboard/specs", json={
        "code": "SIL-2", "name": "Safety Integrity Level 2",
        "family": "functional_safety",
    })).json()

    item = (await client.post(f"/api/dashboard/bids/{bid['id']}/compliance", json={
        "clause_section": "4.2.1",
        "clause_text": "All field-mounted equipment in hazardous areas must be ATEX certified.",
        "our_response": "All instruments ATEX Zone 2 certified. Cert numbers in Appendix C.",
        "status": "compliant",
        "spec_ids": [spec["id"]],
    })).json()

    listed = (await client.get(f"/api/dashboard/bids/{bid['id']}/compliance")).json()
    assert listed["total"] == 1
    assert listed["summary"]["compliant"] == 1
    assert listed["items"][0]["spec_ids"] == [spec["id"]]

    # Patch status to exception (the row procurement will probe in clarification)
    await client.patch(f"/api/dashboard/compliance/{item['id']}", json={
        "status": "exception",
        "notes": "We propose Zone 2 alternative; vendor cert pending.",
    })
    async with session_maker() as s:
        row = await s.get(ComplianceMatrixItem, item["id"])
        assert row.status == "exception"
        assert "Zone 2 alternative" in row.notes

    # Delete
    r = await client.delete(f"/api/dashboard/compliance/{item['id']}")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_compliance_bulk_import(client):
    """Bulk-paste mode: paste an RFP's clause list, get N rows created
    with section markers parsed out."""
    bid = (await client.post("/api/dashboard/bids", json={"name": "BulkBid"})).json()
    text = """
    4.2.1 All field-mounted equipment must be ATEX certified.
    4.2.2 SIL-2 rating required for safety-critical loops.
    4.3.1 IEC 62443-3-2 zone & conduit drawing must be provided.
    Vendor must hold ISO 9001 certification.
    """
    r = await client.post(f"/api/dashboard/bids/{bid['id']}/compliance/bulk", json={"text": text})
    assert r.status_code == 200
    assert r.json()["created"] == 4

    listed = (await client.get(f"/api/dashboard/bids/{bid['id']}/compliance")).json()
    sections = [it["clause_section"] for it in listed["items"]]
    assert "4.2.1" in sections
    assert "4.3.1" in sections


@pytest.mark.asyncio
async def test_industrial_stakeholder_roles_accepted(client):
    """The expanded role taxonomy (ot_cyber, parent_company_standards, etc.)
    must be valid post-PR1 — buying committees in industrial sales include
    these roles, and rejecting them would block the dashboard from modeling
    real deals."""
    deal = (await client.post("/api/dashboard/deals", json={"name": "X"})).json()
    contact = (await client.post("/api/dashboard/contacts", json={"name": "C"})).json()
    for role in ("ot_cyber", "parent_company_standards", "operations",
                 "maintenance", "procurement", "legal", "finance", "it_cyber"):
        r = await client.post(f"/api/dashboard/deals/{deal['id']}/stakeholders", json={
            "contact_id": contact["id"], "role": role,
        })
        assert r.status_code == 200, f"role {role} rejected"
