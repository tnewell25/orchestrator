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
async def test_chat_history_round_trip(client, session_maker):
    """The web chat panel needs to render the same conversation history
    the user would see in Telegram for a given session."""
    from datetime import datetime, timezone, timedelta
    from orchestrator.db.models import Conversation

    sid = "web-thomas"
    base = datetime.now(timezone.utc) - timedelta(minutes=10)
    async with session_maker() as s:
        for i, (role, content) in enumerate([
            ("user", "what's the bosch deal at"),
            ("assistant", "Bosch Forge is in proposal at $500k."),
            ("user", "move it to negotiation"),
            ("assistant", "Done. Moved to negotiation."),
        ]):
            s.add(Conversation(
                session_id=sid, role=role, content=content,
                interface="web",
                timestamp=base + timedelta(seconds=i),
            ))
        # Add a compacted row that should NOT be returned
        s.add(Conversation(
            session_id=sid, role="user",
            content="this was compacted",
            timestamp=base - timedelta(hours=1),
            compacted_into="some-brief-id",
        ))
        await s.commit()

    resp = (await client.get(f"/api/dashboard/chat/{sid}")).json()
    assert len(resp["messages"]) == 4
    assert resp["messages"][0]["role"] == "user"
    assert resp["messages"][0]["content"] == "what's the bosch deal at"
    assert resp["messages"][3]["role"] == "assistant"
    # Compacted row excluded
    assert all("compacted" not in m["content"] for m in resp["messages"])


@pytest.mark.asyncio
async def test_forecast_buckets_strong_negotiation_to_commit(client, session_maker):
    """A negotiation-stage deal with high MEDDIC fill and a supportive
    high-influence champion belongs in Commit. The rationale should
    explain why so the rep trusts the placement."""
    from datetime import datetime, timedelta, timezone
    from orchestrator.db.models import Contact, Deal, DealStakeholder

    async with session_maker() as s:
        contact = Contact(name="Lena Müller", title="VP Eng",
                          last_touch=datetime.now(timezone.utc) - timedelta(days=3))
        s.add(contact)
        await s.commit()
        await s.refresh(contact)

        deal = Deal(
            name="Honeywell DCS Migration",
            stage="negotiation",
            value_usd=2_500_000,
            metrics="Reduce unplanned downtime 18%",
            decision_criteria="Cybersecurity, lifecycle support, OT vendor lock-in",
            decision_process="VP Eng → CapEx review → corporate standards committee",
            paper_process="Legal review T-30, MSA already in place, PO via Ariba",
            pain="Aging Yokogawa platform, vendor exiting US support 2027",
            champion_id=contact.id,
        )
        s.add(deal)
        await s.commit()
        await s.refresh(deal)
        s.add(DealStakeholder(
            deal_id=deal.id, contact_id=contact.id,
            role="champion", sentiment="supportive", influence="high",
        ))
        await s.commit()

    forecast = (await client.get("/api/dashboard/forecast")).json()
    commits = forecast["buckets"]["commit"]
    assert any(d["name"] == "Honeywell DCS Migration" for d in commits)
    deal_row = next(d for d in commits if d["name"] == "Honeywell DCS Migration")
    assert deal_row["champion_score"] >= 90
    assert deal_row["meddic_pct"] >= 70
    # No close_date set → +10 slip; pure negotiation base is 15, so ≤30
    assert deal_row["slip_risk"] <= 30
    assert len(deal_row["reasons"]) > 0


@pytest.mark.asyncio
async def test_forecast_no_champion_lands_in_pipeline(client, session_maker):
    """A deal with no champion mapped is structurally at risk regardless
    of stage. Forecast must reflect that."""
    from orchestrator.db.models import Deal

    async with session_maker() as s:
        s.add(Deal(name="Lonely Deal", stage="proposal", value_usd=300_000))
        await s.commit()

    forecast = (await client.get("/api/dashboard/forecast")).json()
    pipeline = forecast["buckets"]["pipeline"]
    assert any(d["name"] == "Lonely Deal" for d in pipeline)
    deal_row = next(d for d in pipeline if d["name"] == "Lonely Deal")
    assert deal_row["champion_score"] == 0
    # Some reason should mention champion
    assert any("champion" in r.lower() for r in deal_row["reasons"])


@pytest.mark.asyncio
async def test_deal_health_endpoint(client, session_maker):
    """Per-deal health scorecard for the deal-detail page header."""
    from orchestrator.db.models import Deal

    async with session_maker() as s:
        d = Deal(
            name="Mid Deal", stage="qualified", value_usd=200_000,
            metrics="some metrics", pain="some pain",
        )
        s.add(d)
        await s.commit()
        await s.refresh(d)

    health = (await client.get(f"/api/dashboard/deals/{d.id}/health")).json()
    assert "meddic_pct" in health
    assert "champion_score" in health
    assert "forecast_bucket" in health
    assert "reasons" in health
    assert health["forecast_bucket"] in ("commit", "best_case", "pipeline")


@pytest.mark.asyncio
async def test_unified_search(client):
    """Cmd-K palette hits one endpoint that returns matches across
    every entity type — no client-side fan-out."""
    co = (await client.post("/api/dashboard/companies", json={"name": "Bosch"})).json()
    await client.post("/api/dashboard/deals", json={"name": "Bosch Forge", "company_id": co["id"]})
    await client.post("/api/dashboard/contacts", json={"name": "Brian Bosch"})
    await client.post("/api/dashboard/bids", json={"name": "Bosch RFP-2026"})
    await client.post("/api/dashboard/plants", json={"name": "Bosch Stuttgart", "company_id": co["id"]})

    r = (await client.get("/api/dashboard/search", params={"q": "bosch"})).json()
    kinds = {item["kind"] for item in r["results"]}
    assert kinds == {"deal", "contact", "company", "bid", "plant"}

    # Empty query returns nothing
    empty = (await client.get("/api/dashboard/search", params={"q": ""})).json()
    assert empty["results"] == []


@pytest.mark.asyncio
async def test_deal_audit_filters_by_deal_id(client, session_maker):
    """The per-deal audit log shows tool calls + dashboard mutations
    that mention this deal_id in their args_summary."""
    from datetime import datetime, timezone
    from orchestrator.db.models import AuditLog

    deal = (await client.post("/api/dashboard/deals", json={"name": "AuditTest"})).json()
    deal_id = deal["id"]
    other_deal = (await client.post("/api/dashboard/deals", json={"name": "OtherDeal"})).json()

    # Manually insert a few audit rows referencing different deals
    async with session_maker() as s:
        s.add(AuditLog(
            tool_name="bot:deal.update",
            args_summary=f'{{"deal_id":"{deal_id}","stage":"negotiation"}}',
            result_status="ok", session_id="telegram",
            timestamp=datetime.now(timezone.utc),
        ))
        s.add(AuditLog(
            tool_name="bot:reminder.create",
            args_summary=f'{{"related_deal_id":"{other_deal["id"]}"}}',
            result_status="ok", session_id="telegram",
            timestamp=datetime.now(timezone.utc),
        ))
        s.add(AuditLog(
            tool_name="_turn", args_summary="iter=1",
            result_status="ok", session_id="telegram",
            timestamp=datetime.now(timezone.utc),
        ))
        await s.commit()

    audit = (await client.get(f"/api/dashboard/deals/{deal_id}/audit")).json()
    items = audit["items"]
    # We expect: the dashboard:deal.create row + bot:deal.update row
    tool_names = [it["tool_name"] for it in items]
    assert "bot:deal.update" in tool_names
    assert "dashboard:deal.create" in tool_names
    # Other-deal row should NOT appear
    assert all("OtherDeal" not in it["args_summary"] and other_deal["id"] not in it["args_summary"] for it in items)
    # _turn row should NOT appear
    assert "_turn" not in tool_names
    # Source classification works
    sources = {it["source"] for it in items}
    assert "dashboard" in sources
    assert "bot" in sources


@pytest.mark.asyncio
async def test_asset_lifecycle_and_plant_rollup(client, session_maker):
    """Installed-base asset attached to a plant. Plant detail rolls it up
    so 'what's at Bosch Stuttgart?' is a one-click answer."""
    from orchestrator.db.models import Asset

    co = (await client.post("/api/dashboard/companies", json={"name": "Bosch"})).json()
    plant = (await client.post("/api/dashboard/plants", json={
        "name": "Stuttgart", "company_id": co["id"], "site_type": "manufacturing",
    })).json()
    asset = (await client.post("/api/dashboard/assets", json={
        "plant_id": plant["id"], "name": "Process unit DCS",
        "manufacturer": "Honeywell", "model": "Experion PKS",
        "asset_type": "dcs", "vendor": "competitor",
        "end_of_life_date": "2027-06-30",
    })).json()

    # Plant detail includes the asset
    pd = (await client.get(f"/api/dashboard/plants/{plant['id']}")).json()
    assert len(pd["assets"]) == 1
    assert pd["assets"][0]["manufacturer"] == "Honeywell"
    assert pd["assets"][0]["end_of_life_date"] == "2027-06-30"

    # Update vendor to "us" (we won the migration)
    await client.patch(f"/api/dashboard/assets/{asset['id']}", json={"vendor": "us"})
    async with session_maker() as s:
        a = await s.get(Asset, asset["id"])
        assert a.vendor == "us"

    # Filter by vendor
    listed = (await client.get("/api/dashboard/assets", params={"vendor": "us"})).json()
    assert any(a["id"] == asset["id"] for a in listed["assets"])


@pytest.mark.asyncio
async def test_invalid_asset_type_rejected(client):
    co = (await client.post("/api/dashboard/companies", json={"name": "X"})).json()
    plant = (await client.post("/api/dashboard/plants", json={
        "name": "P", "company_id": co["id"],
    })).json()
    r = await client.post("/api/dashboard/assets", json={
        "plant_id": plant["id"], "name": "X", "asset_type": "invalid",
    })
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_co_seller_lifecycle(client, session_maker):
    """Co-seller models the partner rep on the deal — Bosch field engineer
    sponsoring you internally, etc. Tracking explicitly (not as free-text
    in notes) means commission splits + status are queryable."""
    from orchestrator.db.models import CoSeller

    deal = (await client.post("/api/dashboard/deals", json={"name": "Co-sell test"})).json()
    contact = (await client.post("/api/dashboard/contacts", json={"name": "Hans (Bosch)"})).json()

    cs = (await client.post(f"/api/dashboard/deals/{deal['id']}/co-sellers", json={
        "org_name": "Bosch", "role": "oem_rep",
        "contact_id": contact["id"], "commission_pct": 30.0,
    })).json()

    listed = (await client.get(f"/api/dashboard/deals/{deal['id']}/co-sellers")).json()
    assert len(listed["co_sellers"]) == 1
    assert listed["co_sellers"][0]["org_name"] == "Bosch"
    assert listed["co_sellers"][0]["contact_name"] == "Hans (Bosch)"
    assert listed["co_sellers"][0]["commission_pct"] == 30.0

    # Patch + delete
    await client.patch(f"/api/dashboard/co-sellers/{cs['id']}", json={"status": "dormant"})
    async with session_maker() as s:
        row = await s.get(CoSeller, cs["id"])
        assert row.status == "dormant"

    r = await client.delete(f"/api/dashboard/co-sellers/{cs['id']}")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_service_contract_lifecycle(client):
    """Renewal date drives ordering — the most-imminent renewal is
    surfaced first so it never gets missed."""
    co = (await client.post("/api/dashboard/companies", json={"name": "Honeywell"})).json()
    plant = (await client.post("/api/dashboard/plants", json={
        "name": "Site A", "company_id": co["id"],
    })).json()

    c1 = (await client.post("/api/dashboard/contracts", json={
        "company_id": co["id"], "plant_id": plant["id"],
        "name": "Site A PM 2026", "contract_type": "pm_annual",
        "value_usd_annual": 240_000, "renewal_date": "2026-06-30",
    })).json()
    (await client.post("/api/dashboard/contracts", json={
        "company_id": co["id"],
        "name": "Distant renewal", "renewal_date": "2027-12-31",
        "value_usd_annual": 50_000,
    })).json()

    # List orders by renewal date (most urgent first)
    listed = (await client.get("/api/dashboard/contracts")).json()
    contracts = listed["contracts"]
    assert contracts[0]["name"] == "Site A PM 2026"
    assert contracts[0]["company"] == "Honeywell"
    assert contracts[0]["plant"] == "Site A"

    # Filter by company works
    by_company = (await client.get("/api/dashboard/contracts", params={"company_id": co["id"]})).json()
    assert all(c["company_id"] == co["id"] for c in by_company["contracts"])

    # Plant detail rolls up the contract
    pd = (await client.get(f"/api/dashboard/plants/{plant['id']}")).json()
    assert any(c["id"] == c1["id"] for c in pd["contracts"])


@pytest.mark.asyncio
async def test_job_lifecycle_with_logs_changes_punch(client, session_maker):
    """Job + nested daily logs / change orders / punchlist round-trip."""
    from orchestrator.db.models import ChangeOrder, DailyLog, PunchlistItem

    co = (await client.post("/api/dashboard/companies", json={"name": "Site Owner"})).json()
    job = (await client.post("/api/dashboard/jobs", json={
        "name": "Plant 7 Install", "company_id": co["id"],
        "stage": "in_progress", "contract_value_usd": 850_000,
    })).json()

    # Daily log
    log = (await client.post(f"/api/dashboard/jobs/{job['id']}/daily-logs", json={
        "summary": "Crew of 4 ran conduit on north wall",
        "hours_total": 32,
    })).json()
    # Change order with status transition
    chg = (await client.post(f"/api/dashboard/jobs/{job['id']}/change-orders", json={
        "description": "Add fiber drop in MCC-2 per owner request",
        "co_number": "CO-001", "price_usd": 12_500, "labor_hours": 18,
    })).json()
    await client.patch(f"/api/dashboard/change-orders/{chg['id']}", json={
        "status": "approved", "approver": "Hans (owner)",
    })
    # Punchlist
    p = (await client.post(f"/api/dashboard/jobs/{job['id']}/punchlist", json={
        "description": "Touch-up paint on conduit clamps",
        "location": "North wall, panel 3",
    })).json()
    await client.patch(f"/api/dashboard/punchlist/{p['id']}", json={"status": "done"})

    # Job detail rolls them all up
    detail = (await client.get(f"/api/dashboard/jobs/{job['id']}")).json()
    assert len(detail["daily_logs"]) == 1
    assert len(detail["change_orders"]) == 1
    assert detail["change_orders"][0]["status"] == "approved"
    assert len(detail["punchlist"]) == 1
    assert detail["punchlist"][0]["status"] == "done"

    # Approved CO has approved_at set
    async with session_maker() as s:
        co_row = await s.get(ChangeOrder, chg["id"])
        assert co_row.approved_at is not None
        # Done punch has completed_at set
        p_row = await s.get(PunchlistItem, p["id"])
        assert p_row.completed_at is not None
        # Daily log persisted
        log_row = await s.get(DailyLog, log["id"])
        assert log_row.hours_total == 32


@pytest.mark.asyncio
async def test_competitor_with_battle_cards(client):
    comp = (await client.post("/api/dashboard/competitors", json={
        "name": "Siemens", "strengths": "Brand, breadth, EU footprint",
        "weaknesses": "Slow PM cycle, expensive spares",
    })).json()

    # Duplicate name rejected
    dup = await client.post("/api/dashboard/competitors", json={"name": "Siemens"})
    assert dup.status_code == 409

    bc = (await client.post(f"/api/dashboard/competitors/{comp['id']}/battle-cards", json={
        "situation": "Competing against PCS 7 in chemicals",
        "content": "Lead with our 24x7 spares depot in Houston. Their Phoenix cuts response time.",
    })).json()

    listed = (await client.get(f"/api/dashboard/competitors/{comp['id']}/battle-cards")).json()
    assert len(listed["battle_cards"]) == 1
    assert listed["battle_cards"][0]["id"] == bc["id"]


@pytest.mark.asyncio
async def test_proposal_search(client):
    p1 = (await client.post("/api/dashboard/proposals", json={
        "title": "DCS migration scope template",
        "section_type": "scope",
        "content": "Phase 1 baseline survey, FAT/SAT, cutover plan...",
        "tags": "dcs, migration, brownfield",
    })).json()
    (await client.post("/api/dashboard/proposals", json={
        "title": "Standard warranty terms",
        "section_type": "warranty",
        "content": "Two-year parts + labor on installed equipment.",
        "tags": "warranty",
    })).json()

    listed = (await client.get("/api/dashboard/proposals", params={"q": "migration"})).json()
    assert any(p["id"] == p1["id"] for p in listed["proposals"])
    by_type = (await client.get("/api/dashboard/proposals", params={"section_type": "warranty"})).json()
    assert all(p["section_type"] == "warranty" for p in by_type["proposals"])


@pytest.mark.asyncio
async def test_win_loss_aggregation(client):
    deal_won = (await client.post("/api/dashboard/deals", json={"name": "Won deal", "value_usd": 500_000})).json()
    deal_lost = (await client.post("/api/dashboard/deals", json={"name": "Lost deal", "value_usd": 300_000})).json()

    await client.post("/api/dashboard/win-loss", json={
        "deal_id": deal_won["id"], "outcome": "won", "value_usd": 500_000,
        "primary_reason": "Champion strength + 24x7 service", "what_worked": "Plant visit early",
    })
    await client.post("/api/dashboard/win-loss", json={
        "deal_id": deal_lost["id"], "outcome": "lost", "value_usd": 300_000,
        "winning_competitor": "Siemens", "primary_reason": "Parent-co standards override",
    })
    bad = await client.post("/api/dashboard/win-loss", json={"deal_id": deal_won["id"], "outcome": "fake"})
    assert bad.status_code == 400

    summary = (await client.get("/api/dashboard/win-loss")).json()
    assert summary["stats"]["total"] == 2
    assert summary["stats"]["won"] == 1
    assert summary["stats"]["lost"] == 1
    assert summary["stats"]["win_rate"] == 0.5
    assert summary["stats"]["won_value"] == 500_000


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
