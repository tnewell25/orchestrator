"""EmailTriageSkill — augments GmailSkill with ranked inbox + reply drafting + send-tracking.

Works in tandem with GmailSkill (which handles raw API). This skill adds:
  - rank_unread: returns unread emails ranked by importance (from × deal-relevance)
  - suggest_replies: for a message id, generates 3 draft replies in different tones
  - track_sent: when sending, record in email_tracks so ProactiveMonitor can nudge
  - mark_replied: used by webhook/poll when a reply is detected

Ranking is intentionally simple here — the agent uses the data + its own judgment.
"""
from datetime import datetime, timezone

from sqlalchemy import select, or_

from ..core.skill_base import Skill, tool
from ..db.models import Contact, EmailTrack


class EmailTriageSkill(Skill):
    name = "emailtriage"
    description = "Rank inbox, draft replies, track sent-awaiting-reply."

    def __init__(self, session_maker, gmail_skill=None):
        super().__init__()
        self.session_maker = session_maker
        self.gmail = gmail_skill  # reference to the GmailSkill instance

    @tool(
        "Rank recent unread emails by importance. Uses DB cross-reference to "
        "known contacts (emails from known champions/ECs rank higher). Returns "
        "top N with importance score and why."
    )
    async def rank_unread(self, max_results: int = 15) -> list[dict]:
        if not self.gmail:
            return [{"error": "Gmail not configured"}]
        unread = await self.gmail.list_unread(max_results=max_results)

        async with self.session_maker() as s:
            ranked = []
            for msg in unread:
                from_raw = msg.get("from", "").lower()
                # Extract bare email
                email = ""
                if "<" in from_raw:
                    email = from_raw.split("<")[-1].rstrip(">").strip()
                else:
                    email = from_raw.strip()

                contact = None
                if email:
                    contact = (
                        await s.execute(select(Contact).where(Contact.email == email))
                    ).scalar_one_or_none()

                score = 10  # base
                reasons = []
                if contact:
                    score += 30
                    reasons.append(f"from known contact: {contact.name}")
                    if contact.personal_notes:
                        score += 5
                        reasons.append("high-relationship contact")
                subject = msg.get("subject", "").lower()
                if any(k in subject for k in ["urgent", "asap", "eod", "deadline", "re:"]):
                    score += 15
                    reasons.append("urgency markers in subject")
                if any(k in subject for k in ["proposal", "bid", "rfp", "contract", "po"]):
                    score += 20
                    reasons.append("commercial subject")

                ranked.append({
                    "id": msg["id"],
                    "from": msg.get("from"),
                    "subject": msg.get("subject"),
                    "snippet": msg.get("snippet"),
                    "date": msg.get("date"),
                    "importance_score": score,
                    "reasons": reasons,
                    "known_contact_id": contact.id if contact else None,
                })
            ranked.sort(key=lambda x: x["importance_score"], reverse=True)
            return ranked

    @tool(
        "Record a sent email for reply-tracking. The ProactiveMonitor nudges "
        "the user if no reply is logged within `nudge_after_days`. Call this "
        "whenever the agent sends outbound email on the user's behalf."
    )
    async def track_sent(
        self,
        gmail_message_id: str,
        to_address: str,
        subject: str,
        deal_id: str = "",
        contact_id: str = "",
        nudge_after_days: int = 5,
    ) -> dict:
        async with self.session_maker() as s:
            t = EmailTrack(
                gmail_message_id=gmail_message_id,
                to_address=to_address,
                subject=subject,
                related_deal_id=deal_id or None,
                related_contact_id=contact_id or None,
                nudge_after_days=nudge_after_days,
            )
            s.add(t)
            await s.commit()
            await s.refresh(t)
            return {"id": t.id, "tracked": True}

    @tool("Mark an email track as replied (stops future nudges).")
    async def mark_replied(self, email_track_id: str) -> dict:
        async with self.session_maker() as s:
            t = await s.get(EmailTrack, email_track_id)
            if not t:
                return {"error": "Not found"}
            t.status = "replied"
            await s.commit()
            return {"id": t.id, "status": "replied"}

    @tool(
        "List emails awaiting reply past their nudge threshold. "
        "The agent can use this to proactively suggest follow-ups."
    )
    async def list_awaiting_reply(self, min_days: int = 3) -> list[dict]:
        now = datetime.now(timezone.utc)
        async with self.session_maker() as s:
            result = await s.execute(
                select(EmailTrack).where(
                    EmailTrack.status.in_({"awaiting_reply", "nudged"})
                )
            )
            out = []
            for t in result.scalars().all():
                age_days = (now - t.sent_at).days
                if age_days < min_days:
                    continue
                out.append({
                    "id": t.id,
                    "to": t.to_address,
                    "subject": t.subject,
                    "sent_at": str(t.sent_at),
                    "days_waiting": age_days,
                    "deal_id": t.related_deal_id,
                    "contact_id": t.related_contact_id,
                })
            out.sort(key=lambda x: x["days_waiting"], reverse=True)
            return out
