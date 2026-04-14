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


class SemanticMemory(Base):
    """pgvector-backed long-term memory. embedding column is added via raw SQL."""

    __tablename__ = "semantic_memories"

    id = Column(String, primary_key=True, default=_uuid)
    content = Column(Text, nullable=False)
    # embedding vector(384) added in MemoryStore.initialize()
    timestamp = Column(DateTime(timezone=True), default=_now)


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
