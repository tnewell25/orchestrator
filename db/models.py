"""SQLAlchemy models — agent memory tables + CRM domain entities.

All models share a single Base so metadata.create_all() builds the full schema.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Float,
    Integer,
    ForeignKey,
    UniqueConstraint,
    Index,
    Date,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# Agent memory (conversations, facts, OAuth tokens, semantic recall, audit)
# ----------------------------------------------------------------------


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    interface = Column(String, default="telegram")
    timestamp = Column(DateTime(timezone=True), default=_now)
    # Set by compactor when this row's content is rolled into a SessionBrief.
    # The agent excludes compacted rows from get_conversation; the brief is
    # injected into the prompt instead. Kept for audit, not deleted.
    compacted_into = Column(String, ForeignKey("session_briefs.id", ondelete="SET NULL"), nullable=True, index=True)


class BackgroundJob(Base):
    """Durable queue for background work that must survive process restarts.

    CLAIM-CONFIRM pattern: worker marks pending → processing, does the work,
    marks completed. On startup, rows stuck in processing past stale_cutoff
    (default 5min) reset to pending so a crashed worker doesn't lose the work.

    Used by the Compactor (and future: extractor, watcher) so expensive LLM
    calls aren't silently dropped when the server bounces mid-batch.
    """

    __tablename__ = "background_jobs"

    id = Column(String, primary_key=True, default=_uuid)
    job_type = Column(String, nullable=False, index=True)   # compaction | extraction | watcher | ...
    payload = Column(Text, default="")                       # JSON-encoded args
    # status: pending | processing | completed | failed | abandoned
    status = Column(String, default="pending", index=True)
    attempts = Column(Integer, default=0)
    last_error = Column(String(500), default="")
    created_at = Column(DateTime(timezone=True), default=_now, index=True)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)


class SessionBrief(Base):
    """A summary of older conversation turns, replacing them in the active
    window. Multiple briefs can exist per session; only the most recent
    (largest until_timestamp) is injected into the prompt.

    The compactor runs after a turn whenever the live conversation grows beyond
    threshold — it summarizes the oldest N rows, marks them compacted_into=this
    brief, and links until_timestamp to the boundary.
    """

    __tablename__ = "session_briefs"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, nullable=False, index=True)
    summary = Column(Text, nullable=False)
    until_timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    rows_compacted = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=_now)


class Fact(Base):
    """Structured facts about the user or domain that the agent should always know."""

    __tablename__ = "facts"

    id = Column(String, primary_key=True, default=_uuid)
    category = Column(String, nullable=False, index=True)
    key = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (UniqueConstraint("category", "key", name="uq_facts_cat_key"),)


class OAuthToken(Base):
    __tablename__ = "oauth_tokens"

    provider = Column(String, primary_key=True)
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=False, default="")
    expires_at = Column(Float, nullable=True)
    client_id = Column(String, nullable=False, default="")
    source = Column(String, nullable=False, default="env")
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(String, primary_key=True, default=_uuid)
    timestamp = Column(DateTime(timezone=True), default=_now, index=True)
    tool_name = Column(String, nullable=False, index=True)
    args_summary = Column(String(500), nullable=False, default="")
    result_status = Column(String, nullable=False, default="ok")
    result_summary = Column(String(500), nullable=False, default="")
    session_id = Column(String, nullable=False, default="")
    duration_ms = Column(Integer, nullable=False, default=0)
    safety = Column(String, default="auto")  # auto | confirm | approve_external

    # Token accounting for the messages.create API call. Populated on
    # rows where tool_name="_turn"; zero on per-tool-call rows.
    # cache_read = hit on the cached prefix (90% cheaper than regular input).
    # cache_creation = first time a prefix was cached (priced ~25% higher).
    # Seeing cache_read >> cache_creation across a session means W5 is working.
    model = Column(String, default="")
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cache_read_tokens = Column(Integer, default=0)
    cache_creation_tokens = Column(Integer, default=0)


class PendingAction(Base):
    """Externally-visible actions queued for user approval before execution.

    The agent emits a tool_use for an approve_external tool; instead of running,
    the agent writes a PendingAction row and tells the user via the active
    interface (Telegram inline keyboard, web card, etc.). User taps Approve →
    status flips to 'approved' → executor picks it up and runs the tool.
    """

    __tablename__ = "pending_actions"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, nullable=False, index=True)
    tool_name = Column(String, nullable=False)
    tool_input = Column(Text, nullable=False)         # JSON-encoded args
    summary = Column(Text, default="")                # human-readable preview
    # status: pending | approved | rejected | executed | failed | expired
    status = Column(String, default="pending", index=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    decided_at = Column(DateTime(timezone=True), nullable=True)
    executed_at = Column(DateTime(timezone=True), nullable=True)
    result_summary = Column(String(500), default="")
    related_deal_id = Column(String, ForeignKey("deals.id", ondelete="SET NULL"), nullable=True)
    related_contact_id = Column(String, ForeignKey("contacts.id", ondelete="SET NULL"), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)


class SemanticMemory(Base):
    """pgvector-backed long-term memory. embedding column is added via raw SQL."""

    __tablename__ = "semantic_memories"

    id = Column(String, primary_key=True, default=_uuid)
    content = Column(Text, nullable=False)
    # embedding vector(384) added in MemoryStore.initialize()
    timestamp = Column(DateTime(timezone=True), default=_now)
    # Hardens repeated facts: each time the same content (or near-duplicate) is
    # seen, bump count + last_reinforced_at. Hybrid recall ranks high-count
    # memories above one-off mentions.
    reinforcement_count = Column(Integer, default=1)
    last_reinforced_at = Column(DateTime(timezone=True), default=_now)
    source = Column(String, default="conversation")  # conversation | voice | email | meeting | system
    # SHA256(content)[:16] — checked before insert so a 60s dedup window
    # reinforces instead of creating duplicates. Indexed for O(1) lookup.
    content_hash = Column(String(16), index=True, nullable=True)
    # Structured fields (all optional) — let callers attach a one-line title +
    # typed facts/concepts lists so Block D can render tighter than a content
    # blob and hybrid recall can filter/rank on more axes.
    title = Column(String(200), default="")
    facts_json = Column(Text, default="")        # JSON array of short bullets
    concepts_json = Column(Text, default="")     # JSON array of entity/topic tags


class Edge(Base):
    """Typed graph edge between any two entities — the cross-reference layer.

    Direction is from→to. For symmetric semantics (e.g. "competes_with") insert
    both directions or use the bidirectional traversal helper in core.graph.

    A single (from, to, kind) tuple is unique; subsequent observations bump
    reinforcement_count instead of inserting duplicates.
    """

    __tablename__ = "edges"

    id = Column(String, primary_key=True, default=_uuid)
    from_type = Column(String, nullable=False)
    from_id = Column(String, nullable=False)
    to_type = Column(String, nullable=False)
    to_id = Column(String, nullable=False)
    kind = Column(String, nullable=False)

    weight = Column(Float, default=1.0)        # ranking signal for retrieval
    confidence = Column(Float, default=1.0)    # extractor's confidence
    source = Column(String, default="auto_extract")  # auto_extract | manual | system | rule

    reinforcement_count = Column(Integer, default=1)
    last_reinforced_at = Column(DateTime(timezone=True), default=_now)
    created_at = Column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        UniqueConstraint(
            "from_type", "from_id", "to_type", "to_id", "kind",
            name="uq_edge_unique",
        ),
        Index("ix_edges_from", "from_type", "from_id"),
        Index("ix_edges_to", "to_type", "to_id"),
        Index("ix_edges_kind", "kind"),
    )


# ----------------------------------------------------------------------
# CRM domain — Company, Contact, Deal, Meeting, ActionItem, Note
# ----------------------------------------------------------------------


class Company(Base):
    __tablename__ = "companies"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False, index=True)
    industry = Column(String, default="")
    website = Column(String, default="")
    notes = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    contacts = relationship("Contact", back_populates="company", cascade="all, delete-orphan")
    deals = relationship("Deal", back_populates="company", cascade="all, delete-orphan")


class Contact(Base):
    __tablename__ = "contacts"

    id = Column(String, primary_key=True, default=_uuid)
    company_id = Column(String, ForeignKey("companies.id", ondelete="SET NULL"), nullable=True, index=True)
    name = Column(String, nullable=False, index=True)
    title = Column(String, default="")
    email = Column(String, default="", index=True)
    phone = Column(String, default="")
    linkedin = Column(String, default="")
    # personal_notes: kids, hobbies, interests — the relationship memory
    personal_notes = Column(Text, default="")
    last_touch = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    company = relationship("Company", back_populates="contacts")


class Deal(Base):
    __tablename__ = "deals"

    id = Column(String, primary_key=True, default=_uuid)
    company_id = Column(String, ForeignKey("companies.id", ondelete="SET NULL"), nullable=True, index=True)
    name = Column(String, nullable=False)
    # stage: prospect | qualified | proposal | negotiation | closed_won | closed_lost
    stage = Column(String, default="prospect", index=True)
    value_usd = Column(Float, default=0.0)
    close_date = Column(Date, nullable=True)
    competitors = Column(Text, default="")  # comma-separated
    next_step = Column(Text, default="")
    notes = Column(Text, default="")

    # MEDDIC — drives elite selling discipline. All optional, but get_context
    # surfaces gaps so the agent nudges the user to fill them in next meeting.
    economic_buyer_id = Column(String, ForeignKey("contacts.id", ondelete="SET NULL"), nullable=True)
    champion_id = Column(String, ForeignKey("contacts.id", ondelete="SET NULL"), nullable=True)
    metrics = Column(Text, default="")           # M — quantifiable impact ("reduce downtime 15%")
    decision_criteria = Column(Text, default="")  # DC — what they're evaluating on
    decision_process = Column(Text, default="")   # DP — how they'll decide
    paper_process = Column(Text, default="")      # PP — legal/procurement/security steps
    pain = Column(Text, default="")               # I — identified pain being solved

    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    company = relationship("Company", back_populates="deals")
    meetings = relationship("Meeting", back_populates="deal", cascade="all, delete-orphan")
    action_items = relationship("ActionItem", back_populates="deal", cascade="all, delete-orphan")
    bids = relationship("Bid", back_populates="deal", cascade="all, delete-orphan")
    economic_buyer = relationship("Contact", foreign_keys=[economic_buyer_id])
    champion = relationship("Contact", foreign_keys=[champion_id])


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(String, primary_key=True, default=_uuid)
    deal_id = Column(String, ForeignKey("deals.id", ondelete="SET NULL"), nullable=True, index=True)
    date = Column(DateTime(timezone=True), default=_now, index=True)
    attendees = Column(Text, default="")  # comma-separated names
    summary = Column(Text, default="")
    decisions = Column(Text, default="")
    transcript = Column(Text, default="")  # raw voice-note transcript if any
    created_at = Column(DateTime(timezone=True), default=_now)

    deal = relationship("Deal", back_populates="meetings")


class ActionItem(Base):
    """Follow-ups, commitments, tasks. Named ActionItem to avoid collision with
    the `tasks` / TaskCreate conversation concept."""

    __tablename__ = "action_items"

    id = Column(String, primary_key=True, default=_uuid)
    deal_id = Column(String, ForeignKey("deals.id", ondelete="SET NULL"), nullable=True, index=True)
    contact_id = Column(String, ForeignKey("contacts.id", ondelete="SET NULL"), nullable=True, index=True)
    description = Column(Text, nullable=False)
    due_date = Column(Date, nullable=True, index=True)
    status = Column(String, default="open", index=True)  # open | done | snoozed
    source = Column(String, default="manual")  # manual | meeting | email | proactive
    created_at = Column(DateTime(timezone=True), default=_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    deal = relationship("Deal", back_populates="action_items")


class Note(Base):
    """Free-form notes attached to any entity."""

    __tablename__ = "notes"

    id = Column(String, primary_key=True, default=_uuid)
    subject_type = Column(String, nullable=False)  # "company" | "contact" | "deal" | "bid"
    subject_id = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)


class Bid(Base):
    """RFP / bid / tender. Distinct from Deal because a single Deal may span
    multiple bids, and some bids stand alone before being tied to a Deal.
    Deadlines drive automatic reminders (T-7d / T-3d / T-1d)."""

    __tablename__ = "bids"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    company_id = Column(String, ForeignKey("companies.id", ondelete="SET NULL"), nullable=True, index=True)
    deal_id = Column(String, ForeignKey("deals.id", ondelete="SET NULL"), nullable=True, index=True)

    # Deadlines — all UTC
    submission_deadline = Column(DateTime(timezone=True), nullable=True, index=True)
    qa_deadline = Column(DateTime(timezone=True), nullable=True)

    value_usd = Column(Float, default=0.0)
    # stage: evaluating | in_progress | submitted | won | lost | withdrawn
    stage = Column(String, default="evaluating", index=True)

    rfp_url = Column(String, default="")
    deliverables = Column(Text, default="")  # what must be included in the submission
    notes = Column(Text, default="")

    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    deal = relationship("Deal", back_populates="bids")


class User(Base):
    """A person using the system (sales engineer, PM, tech, foreman, admin).
    Role drives what the agent emphasizes in responses."""

    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    telegram_user_id = Column(String, unique=True, nullable=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, default="")
    phone = Column(String, default="")
    # role: owner | sales_engineer | project_manager | foreman | technician | dispatcher | admin
    role = Column(String, default="sales_engineer", index=True)
    active = Column(String, default="yes")  # "yes" | "no"
    # Tech-specific fields (used when role includes technician/foreman)
    trade_level = Column(String, default="")  # apprentice | j-man | master | foreman | gf
    hourly_rate_usd = Column(Float, default=0.0)
    certifications = Column(Text, default="")  # comma-separated with expiry dates in freeform
    created_at = Column(DateTime(timezone=True), default=_now)


class Job(Base):
    """A field job / work order. Bridges from won bid/deal → actual execution."""

    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=_uuid)
    job_number = Column(String, unique=True, index=True)  # internal job code e.g. "25-0142"
    name = Column(String, nullable=False)
    company_id = Column(String, ForeignKey("companies.id", ondelete="SET NULL"), nullable=True, index=True)
    deal_id = Column(String, ForeignKey("deals.id", ondelete="SET NULL"), nullable=True, index=True)
    bid_id = Column(String, ForeignKey("bids.id", ondelete="SET NULL"), nullable=True, index=True)

    # Site info
    site_address = Column(String, default="")
    site_contact_id = Column(String, ForeignKey("contacts.id", ondelete="SET NULL"), nullable=True)
    gc_name = Column(String, default="")  # general contractor if sub work

    # Scope + budget
    scope = Column(Text, default="")
    contract_value_usd = Column(Float, default=0.0)
    labor_budget_hours = Column(Float, default=0.0)
    material_budget_usd = Column(Float, default=0.0)

    # Status
    # stage: scheduled | in_progress | punch | inspected | closed_out | warranty
    stage = Column(String, default="scheduled", index=True)
    scheduled_start = Column(Date, nullable=True)
    scheduled_end = Column(Date, nullable=True)
    actual_start = Column(Date, nullable=True)
    actual_end = Column(Date, nullable=True)

    # PM assignment
    project_manager_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    foreman_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    notes = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)


class Timesheet(Base):
    """A single labor entry. Geofence/voice-driven clock in/out."""

    __tablename__ = "timesheets"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    clock_in = Column(DateTime(timezone=True), nullable=False, default=_now)
    clock_out = Column(DateTime(timezone=True), nullable=True)
    hours = Column(Float, default=0.0)  # computed or manually entered
    billable = Column(String, default="yes")
    source = Column(String, default="manual")  # manual | geofence | voice
    notes = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=_now)


class DailyLog(Base):
    """End-of-day recap per tech per job. Auto-generated from voice notes.
    Drives the PM/client email that replaces the form-filling tax."""

    __tablename__ = "daily_logs"

    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    log_date = Column(Date, nullable=False, default=lambda: _now().date(), index=True)

    summary = Column(Text, default="")
    work_performed = Column(Text, default="")
    issues = Column(Text, default="")          # anything that blocked or slowed the crew
    materials_used = Column(Text, default="")  # freeform or structured JSON-like
    hours_total = Column(Float, default=0.0)
    next_day_plan = Column(Text, default="")
    transcript = Column(Text, default="")      # raw voice-note text if voice-driven

    # Did this log get emailed to client / PM?
    emailed_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=_now)


class ChangeOrder(Base):
    """Captured mid-job scope changes. Biggest source of revenue leakage if not tracked."""

    __tablename__ = "change_orders"

    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by_user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    # status: draft | pm_review | submitted | approved | rejected | invoiced
    status = Column(String, default="draft", index=True)
    co_number = Column(String, default="")  # e.g. "CO-001"

    requested_by = Column(String, default="")  # customer rep name
    description = Column(Text, nullable=False)
    labor_hours = Column(Float, default=0.0)
    material_cost_usd = Column(Float, default=0.0)
    price_usd = Column(Float, default=0.0)

    captured_source = Column(String, default="voice")  # voice | manual | email
    approved_at = Column(DateTime(timezone=True), nullable=True)
    approver = Column(String, default="")  # who on customer side approved
    notes = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=_now)


class JobPhoto(Base):
    """Photo record with searchable metadata. File lives elsewhere (S3 later);
    here we store reference + tags."""

    __tablename__ = "job_photos"

    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    taken_at = Column(DateTime(timezone=True), default=_now, index=True)

    storage_url = Column(String, default="")   # S3/GCS path (future)
    telegram_file_id = Column(String, default="")  # Telegram's file_id for immediate re-fetch
    caption = Column(Text, default="")
    tags = Column(Text, default="")  # comma-separated: "panel,mcc-2,as-built"
    location_desc = Column(String, default="")  # "Room 204", "East wall"

    # Category for quick filtering
    # category: install | as_built | issue | pre_inspection | post_inspection | safety | material | receipt
    category = Column(String, default="install", index=True)
    created_at = Column(DateTime(timezone=True), default=_now)


class PunchlistItem(Base):
    """Items remaining at end of job. Tracked to unlock final payment."""

    __tablename__ = "punchlist_items"

    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    description = Column(Text, nullable=False)
    location = Column(String, default="")
    assigned_to_user_id = Column(String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    # status: open | in_progress | done | waived
    status = Column(String, default="open", index=True)
    completion_photo_id = Column(String, ForeignKey("job_photos.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class Inspection(Base):
    """AHJ / owner / commissioning inspections."""

    __tablename__ = "inspections"

    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    # kind: rough | final | commissioning | fire_alarm | owner_walkthrough | other
    kind = Column(String, nullable=False)
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    inspector_name = Column(String, default="")
    jurisdiction = Column(String, default="")  # city/county AHJ name
    # status: scheduled | passed | failed | rescheduled
    status = Column(String, default="scheduled", index=True)
    result_notes = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=_now)


class DealStakeholder(Base):
    """Contact's role in a specific deal. A person can play different roles on
    different deals, so this is a join with role metadata."""

    __tablename__ = "deal_stakeholders"

    id = Column(String, primary_key=True, default=_uuid)
    deal_id = Column(String, ForeignKey("deals.id", ondelete="CASCADE"), nullable=False, index=True)
    contact_id = Column(String, ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False, index=True)
    # role: champion | economic_buyer | technical_buyer | blocker | coach | user
    role = Column(String, nullable=False, index=True)
    # sentiment: supportive | neutral | opposed | unknown
    sentiment = Column(String, default="unknown")
    influence = Column(String, default="medium")  # low | medium | high
    notes = Column(Text, default="")
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    __table_args__ = (
        UniqueConstraint("deal_id", "contact_id", "role", name="uq_deal_stake_role"),
    )


class Competitor(Base):
    __tablename__ = "competitors"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False, unique=True, index=True)
    aliases = Column(Text, default="")
    strengths = Column(Text, default="")
    weaknesses = Column(Text, default="")
    pricing_notes = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=_now)


class BattleCard(Base):
    """Reusable content for beating a competitor. Embedding column added in MemoryStore.initialize."""

    __tablename__ = "battle_cards"

    id = Column(String, primary_key=True, default=_uuid)
    competitor_id = Column(String, ForeignKey("competitors.id", ondelete="CASCADE"), nullable=True)
    situation = Column(Text, default="")
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)


class WinLossRecord(Base):
    __tablename__ = "win_loss_records"

    id = Column(String, primary_key=True, default=_uuid)
    deal_id = Column(String, ForeignKey("deals.id", ondelete="CASCADE"), nullable=False, index=True)
    outcome = Column(String, nullable=False)
    winning_competitor = Column(String, default="")
    primary_reason = Column(Text, default="")
    what_worked = Column(Text, default="")
    what_didnt = Column(Text, default="")
    lessons = Column(Text, default="")
    value_usd = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=_now)


class EmailTrack(Base):
    """Tracks outgoing emails so we can nudge if no reply."""

    __tablename__ = "email_tracks"

    id = Column(String, primary_key=True, default=_uuid)
    gmail_message_id = Column(String, index=True)
    thread_id = Column(String, default="")
    to_address = Column(String, default="")
    subject = Column(String, default="")
    sent_at = Column(DateTime(timezone=True), default=_now, index=True)
    related_deal_id = Column(String, ForeignKey("deals.id", ondelete="SET NULL"), nullable=True)
    related_contact_id = Column(String, ForeignKey("contacts.id", ondelete="SET NULL"), nullable=True)
    status = Column(String, default="awaiting_reply", index=True)
    nudge_after_days = Column(Integer, default=5)
    last_reminded_at = Column(DateTime(timezone=True), nullable=True)


class ProposalPrecedent(Base):
    """Reusable proposal section/paragraph. Embedding added via raw SQL at init."""

    __tablename__ = "proposal_precedents"

    id = Column(String, primary_key=True, default=_uuid)
    title = Column(String, nullable=False)
    section_type = Column(String, default="")  # intro | scope | pricing | warranty | etc.
    content = Column(Text, nullable=False)
    tags = Column(Text, default="")  # comma-separated context tags
    source_deal_id = Column(String, ForeignKey("deals.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)


class WarrantyRecord(Base):
    """Per-job warranty tracking for post-closeout service calls."""

    __tablename__ = "warranty_records"

    id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    warranty_start = Column(Date, nullable=True)
    warranty_end = Column(Date, nullable=True, index=True)
    coverage = Column(Text, default="")  # what's covered
    exclusions = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=_now)


class Reminder(Base):
    """Time-based prompts fired to the owner via the active messaging interface.
    Polled every 30s by ReminderService; no APScheduler persistence complexity."""

    __tablename__ = "reminders"

    id = Column(String, primary_key=True, default=_uuid)
    trigger_at = Column(DateTime(timezone=True), nullable=False, index=True)
    message = Column(Text, nullable=False)

    # Target channel
    target_chat_id = Column(String, nullable=True)   # Telegram chat_id (as string for flexibility)
    interface = Column(String, default="telegram")

    # Context — what this reminder is about (rendered by the agent when it fires)
    related_deal_id = Column(String, ForeignKey("deals.id", ondelete="SET NULL"), nullable=True, index=True)
    related_meeting_id = Column(String, ForeignKey("meetings.id", ondelete="SET NULL"), nullable=True, index=True)
    related_contact_id = Column(String, ForeignKey("contacts.id", ondelete="SET NULL"), nullable=True, index=True)
    related_bid_id = Column(String, ForeignKey("bids.id", ondelete="SET NULL"), nullable=True, index=True)
    kind = Column(String, default="custom")  # custom | pre_meeting | bid_deadline | commitment

    # Lifecycle
    status = Column(String, default="pending", index=True)  # pending | sent | cancelled | failed
    sent_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
