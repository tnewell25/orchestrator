"""Shared constants — entity types, edge kinds, traversal limits.

Entity types match SQLAlchemy class names (case-sensitive) so a (type, id) pair
deterministically identifies any node in the graph.
"""


class EntityType:
    MEMORY = "Memory"
    CONTACT = "Contact"
    DEAL = "Deal"
    COMPANY = "Company"
    MEETING = "Meeting"
    BID = "Bid"
    EMAIL = "Email"
    ACTION_ITEM = "ActionItem"
    JOB = "Job"
    REMINDER = "Reminder"

    ALL = (
        MEMORY, CONTACT, DEAL, COMPANY, MEETING, BID, EMAIL, ACTION_ITEM, JOB, REMINDER,
    )


class EdgeKind:
    # Memory → entity: a memory text references this entity
    MENTIONS = "mentions"
    # Generic linkage when relationship is fuzzy
    REFERENCES = "references"
    # Meeting ↔ Contact attendance
    ATTENDED = "attended"
    # Stakeholder roles (mirror DealStakeholder for graph traversal)
    CHAMPION_OF = "champion_of"
    ECONOMIC_BUYER_OF = "economic_buyer_of"
    BLOCKER_OF = "blocker_of"
    STAKEHOLDER_IN = "stakeholder_in"
    # Org structure
    REPORTS_TO = "reports_to"
    WORKS_AT = "works_at"
    # Deal/competitor map
    COMPETES_WITH = "competes_with"
    # Provenance — this entity was derived from another (DailyLog from Meeting, etc.)
    DERIVED_FROM = "derived_from"
    # Temporal anchor — Reminder/Email/ActionItem about this Deal
    ABOUT = "about"
    # Email thread → contact
    SENT_TO = "sent_to"
    SENT_FROM = "sent_from"


# Hybrid retrieval tuning (used by W2 — kept here so the constants live in one place)
RECENCY_HALF_LIFE_DAYS = 14.0
GRAPH_PROXIMITY_BONUS = 0.25       # added to score when memory is graph-proximate to focus
MAX_GRAPH_TRAVERSAL_DEPTH = 3
MAX_SUBGRAPH_NODES = 50
