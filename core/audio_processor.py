"""Audio → transcript → categorization pipeline.

Single entry point: `process_audio(meeting_id, audio_bytes, filename)`.
Pipeline:
  1. POST audio to OpenAI Whisper API → transcript text
  2. LLM categorization pass (Haiku) → meeting_type, sentiment, attendees,
     decisions, action items, MEDDIC deltas, competitors, pricing
  3. Persist transcript + categorization fields to the Meeting row
  4. Discard raw audio (transcript IS the audit; saves storage + GDPR)

Runs synchronously in the request for files <10MB. Larger files should
be queued via JobQueue (not yet wired here — defer to Phase 2).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


_CATEGORIZE_PROMPT = """You are categorizing a sales meeting transcript for a senior industrial sales engineer (think Bosch / Honeywell-class buyers).

Read the transcript and return STRICT JSON in this exact shape (no markdown fences, no commentary):

{{
  "meeting_type": "discovery" | "technical_deep_dive" | "pricing" | "negotiation" | "status" | "kickoff" | "closing" | "other",
  "sentiment": "positive" | "neutral" | "concerning",
  "attendees_mentioned": ["Full Name 1", "Full Name 2"],
  "summary": "2-3 sentences of what happened, in past tense, specific.",
  "key_decisions": ["one-line decision 1", "..."],
  "action_items": [
    {{"description": "what needs to happen", "owner": "us" | "customer" | "joint", "due_hint": "free-text date hint or ''"}}
  ],
  "meddic_deltas": {{
    "metrics": "..." | null,
    "decision_criteria": "..." | null,
    "decision_process": "..." | null,
    "paper_process": "..." | null,
    "pain": "..." | null
  }},
  "competitors_mentioned": ["competitor name 1"],
  "pricing_mentioned": "free-text summary of any $ figures or pricing discussion, or ''",
  "follow_up_concern": "free-text — anything urgent the rep should know, or ''"
}}

Rules:
- ONLY include MEDDIC deltas where the transcript provides concrete new information. Use null otherwise.
- attendees_mentioned: only people NAMED in the transcript. Do not invent.
- summary: under 400 chars. Past tense. Specific.
- key_decisions: under 5 items. Each one-line.
- action_items: things SOMEONE committed to. due_hint can be like "by Friday", "next week", "Q3", or empty string.
- competitors_mentioned: only proper noun company names (Siemens, Yokogawa, ABB, etc).
- Be conservative — empty arrays/null are better than hallucinated content.

DEAL CONTEXT:
{deal_context}

TRANSCRIPT:
{transcript}

Return JSON only."""


async def transcribe_audio(audio_bytes: bytes, filename: str, openai_api_key: str) -> str:
    """Upload audio to OpenAI Whisper API, return the transcript text.

    Whisper accepts mp3/mp4/mpeg/mpga/m4a/wav/webm up to 25 MB.
    """
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured — required for transcription")

    async with httpx.AsyncClient(timeout=180) as c:
        files = {"file": (filename, audio_bytes, "application/octet-stream")}
        data = {"model": "whisper-1", "response_format": "text"}
        headers = {"Authorization": f"Bearer {openai_api_key}"}
        r = await c.post(
            "https://api.openai.com/v1/audio/transcriptions",
            files=files, data=data, headers=headers,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Whisper API error {r.status_code}: {r.text[:300]}")
        return r.text.strip()


async def categorize_transcript(
    transcript: str, deal_context: str, llm_client, model: str,
) -> dict[str, Any]:
    """Run the LLM categorization pass, return the parsed JSON dict.

    Defensive parsing — strips ```json fences if the model wraps them.
    """
    prompt = _CATEGORIZE_PROMPT.format(
        deal_context=deal_context[:1500],
        transcript=transcript[:8000],  # cap context to keep token use sane
    )
    resp = await llm_client.messages.create(
        model=model, max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in resp.content:
        if hasattr(block, "text"):
            text += block.text
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Categorization returned non-JSON: %s ... err: %s", text[:200], e)
        return {}


async def process_meeting_audio(
    *,
    meeting_id: str,
    audio_bytes: bytes,
    filename: str,
    session_maker,
    llm_client,
    openai_api_key: str,
    fast_model: str = "claude-haiku-4-5",
) -> dict[str, Any]:
    """End-to-end: transcribe + categorize + persist. Returns the Meeting
    row's updated state + the raw extracted suggestions.

    This runs synchronously. For files >10MB or transcripts likely >5 min,
    callers should hand off to the job queue instead.
    """
    from sqlalchemy import select
    from ..db.models import Deal, Meeting

    # Mark transcribing
    async with session_maker() as s:
        m = await s.get(Meeting, meeting_id)
        if not m:
            raise ValueError(f"meeting {meeting_id} not found")
        m.audio_processing_status = "transcribing"
        m.audio_processing_error = ""
        await s.commit()

    # Step 1 — transcribe
    try:
        transcript = await transcribe_audio(audio_bytes, filename, openai_api_key)
    except Exception as e:
        async with session_maker() as s:
            m = await s.get(Meeting, meeting_id)
            if m:
                m.audio_processing_status = "failed"
                m.audio_processing_error = f"transcription: {str(e)[:400]}"
                await s.commit()
        raise

    # Step 2 — categorize
    deal_context = "(no linked deal)"
    async with session_maker() as s:
        m = await s.get(Meeting, meeting_id)
        if m and m.deal_id:
            d = await s.get(Deal, m.deal_id)
            if d:
                deal_context = (
                    f"Deal: {d.name} (stage: {d.stage}, value: ${d.value_usd or 0:,.0f})\n"
                    f"Next step: {d.next_step or '(none)'}\n"
                    f"Existing MEDDIC — metrics: {d.metrics or '(empty)'} | "
                    f"DC: {d.decision_criteria or '(empty)'} | "
                    f"DP: {d.decision_process or '(empty)'} | "
                    f"PP: {d.paper_process or '(empty)'} | "
                    f"Pain: {d.pain or '(empty)'}\n"
                    f"Competitors: {d.competitors or '(none)'}"
                )

    async with session_maker() as s:
        m = await s.get(Meeting, meeting_id)
        if m:
            m.audio_processing_status = "categorizing"
            m.transcript = transcript
            await s.commit()

    extracted: dict[str, Any] = {}
    try:
        extracted = await categorize_transcript(transcript, deal_context, llm_client, fast_model)
    except Exception as e:
        logger.exception("Categorization failed")
        async with session_maker() as s:
            m = await s.get(Meeting, meeting_id)
            if m:
                m.audio_processing_status = "failed"
                m.audio_processing_error = f"categorization: {str(e)[:400]}"
                await s.commit()
        # Still return the transcript-only state — user can apply manually
        return {
            "meeting_id": meeting_id,
            "transcript": transcript,
            "extracted": {},
            "status": "failed",
        }

    # Step 3 — persist categorization
    async with session_maker() as s:
        m = await s.get(Meeting, meeting_id)
        if m:
            m.meeting_type = extracted.get("meeting_type", "other") or "other"
            m.sentiment = extracted.get("sentiment", "unknown") or "unknown"
            m.summary = extracted.get("summary", m.summary) or m.summary
            decisions_list = extracted.get("key_decisions", [])
            if decisions_list:
                m.decisions = "\n".join(f"- {d}" for d in decisions_list)
            attendees_list = extracted.get("attendees_mentioned", [])
            if attendees_list and not m.attendees:
                m.attendees = ", ".join(attendees_list)
            competitors_list = extracted.get("competitors_mentioned", [])
            if competitors_list:
                m.competitors_mentioned = ", ".join(competitors_list)
            pricing = extracted.get("pricing_mentioned", "")
            if pricing:
                m.pricing_mentioned = pricing
            m.audio_processing_status = "done"
            await s.commit()

    return {
        "meeting_id": meeting_id,
        "transcript": transcript,
        "extracted": extracted,
        "status": "done",
    }
