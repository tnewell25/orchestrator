"""Audio → transcript → categorization pipeline.

Single entry point: `process_meeting_audio(meeting_id, audio_bytes, filename)`.
Pipeline:
  1. If file >25MB, chunk it via pydub/ffmpeg into 10-min segments
  2. POST each chunk to OpenAI Whisper API in parallel (asyncio.gather)
  3. Concatenate transcripts in order
  4. LLM categorization pass (Haiku) on the combined transcript
  5. Persist transcript + categorization fields to the Meeting row
  6. Discard raw audio (transcript IS the audit; saves storage + GDPR)

Whisper hard caps at 25MB per file. We chunk to 22MB safety margin.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Whisper's 25MB cap with safety margin. 10-min chunks at OpenAI's recommended
# bitrates land comfortably under this even for high-quality recordings.
_CHUNK_MAX_BYTES = 22 * 1024 * 1024
_CHUNK_DURATION_MS = 10 * 60 * 1000  # 10 minutes per chunk


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

If the transcript has 'Speaker 0:', 'Speaker 1:' etc (diarized), use them to:
- Distinguish what THE REP committed to (action_items owner='us') vs what THE CUSTOMER committed to (owner='customer')
- Attribute MEDDIC pain/criteria to whoever stated it (more reliable signal when the customer says it)
- Identify the EB/champion by who's leading the conversation vs who defers to whom

DEAL CONTEXT:
{deal_context}

TRANSCRIPT:
{transcript}

Return JSON only."""


async def _whisper_one(client: httpx.AsyncClient, audio_bytes: bytes, filename: str, key: str) -> str:
    """Single Whisper API call. Caller handles chunking + concatenation."""
    files = {"file": (filename, audio_bytes, "application/octet-stream")}
    data = {"model": "whisper-1", "response_format": "text"}
    headers = {"Authorization": f"Bearer {key}"}
    r = await client.post(
        "https://api.openai.com/v1/audio/transcriptions",
        files=files, data=data, headers=headers,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Whisper API error {r.status_code}: {r.text[:300]}")
    return r.text.strip()


async def _transcribe_via_deepgram(
    audio_bytes: bytes, filename: str, api_key: str, model: str = "nova-3",
) -> str:
    """Deepgram prerecorded transcription with speaker diarization.

    Better than Whisper for our use case:
    - No file-size cap (Whisper hard-caps at 25MB)
    - Speaker diarization: knows who said what — huge for MEDDIC quality
    - ~30% cheaper for long-form ($0.0043/min vs Whisper's $0.006/min)
    - 2-5x faster

    Output is formatted as 'Speaker N: utterance' blocks so the
    categorizer can distinguish rep speech from customer speech.
    """
    # Infer Content-Type from filename so Deepgram can decode any container
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()
    mime = {
        "mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/m4a",
        "mp4": "audio/mp4", "webm": "audio/webm", "ogg": "audio/ogg",
        "oga": "audio/ogg", "opus": "audio/opus", "flac": "audio/flac",
        "aac": "audio/aac",
    }.get(ext, "application/octet-stream")

    params = {
        "model": model,
        "smart_format": "true",
        "diarize": "true",
        "punctuate": "true",
        "paragraphs": "true",
        "utterances": "true",
        "language": "en",
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": mime,
    }

    # Long uploads + long server-side processing — generous timeout
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.post(
            "https://api.deepgram.com/v1/listen",
            params=params, headers=headers, content=audio_bytes,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Deepgram error {r.status_code}: {r.text[:300]}")
        body = r.json()

    # Format the transcript with speaker labels. Group consecutive
    # words/utterances by speaker so the output reads like a script.
    try:
        utterances = body["results"].get("utterances", [])
        if utterances:
            lines = []
            current_speaker = None
            current_text: list[str] = []
            for u in utterances:
                spk = u.get("speaker", 0)
                if spk != current_speaker and current_text:
                    lines.append(f"Speaker {current_speaker}: {' '.join(current_text)}")
                    current_text = []
                current_speaker = spk
                current_text.append(u.get("transcript", "").strip())
            if current_text:
                lines.append(f"Speaker {current_speaker}: {' '.join(current_text)}")
            return "\n\n".join(l for l in lines if l).strip()

        # Fallback: paragraphs without diarization
        paragraphs = body["results"]["channels"][0]["alternatives"][0].get("paragraphs", {})
        if paragraphs:
            text = paragraphs.get("transcript", "")
            if text:
                return text.strip()

        # Last fallback: flat transcript
        return body["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except (KeyError, IndexError) as e:
        logger.warning("Deepgram response shape unexpected: %s", e)
        # Try the simplest extraction
        try:
            return body["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        except Exception:
            raise RuntimeError("Deepgram returned no usable transcript")


def _chunk_audio_via_ffmpeg(audio_bytes: bytes, src_filename: str) -> list[tuple[bytes, str]]:
    """Split a >25MB audio file into 10-min mp3 chunks via pydub/ffmpeg.

    Returns list of (bytes, filename) tuples in time order. Re-encodes
    to mp3 64kbps mono (Whisper-recommended) which dramatically reduces
    file size — a 60-min recording at this bitrate is ~28MB total split
    across 6 chunks of ~5MB each.
    """
    from pydub import AudioSegment

    # Infer format from filename extension; pydub falls back to ffmpeg for
    # any container ffmpeg can read (m4a, mp3, wav, webm, ogg, mp4, flac, …).
    ext = (src_filename.rsplit(".", 1)[-1] if "." in src_filename else "").lower() or "mp3"
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)

    # Whisper-friendly settings: 16kHz mono, mp3 64kbps. Drops file size
    # by ~5-10x vs original without hurting transcription quality.
    audio = audio.set_channels(1).set_frame_rate(16000)

    chunks: list[tuple[bytes, str]] = []
    n_chunks = max(1, (len(audio) + _CHUNK_DURATION_MS - 1) // _CHUNK_DURATION_MS)
    for i in range(n_chunks):
        start = i * _CHUNK_DURATION_MS
        end = min(start + _CHUNK_DURATION_MS, len(audio))
        segment = audio[start:end]
        buf = io.BytesIO()
        segment.export(buf, format="mp3", bitrate="64k", parameters=["-ac", "1"])
        chunks.append((buf.getvalue(), f"chunk-{i+1:03d}.mp3"))
    return chunks


async def transcribe_audio(
    audio_bytes: bytes,
    filename: str,
    openai_api_key: str = "",
    deepgram_api_key: str = "",
    deepgram_model: str = "nova-3",
) -> str:
    """Transcribe an audio file. Routes to the best available backend.

    Preferred: Deepgram (no size cap, speaker diarization, faster, cheaper).
    Fallback: OpenAI Whisper with pydub/ffmpeg chunking for files >22MB.

    The output transcript may include 'Speaker N:' labels when Deepgram
    diarization is used — the categorizer prompt knows how to read these.
    """
    if deepgram_api_key:
        logger.info("Transcribing %.1fMB via Deepgram (%s, diarized)",
                    len(audio_bytes) / 1024 / 1024, deepgram_model)
        return await _transcribe_via_deepgram(
            audio_bytes, filename, deepgram_api_key, deepgram_model,
        )

    if not openai_api_key:
        raise ValueError(
            "No transcription backend configured. Set DEEPGRAM_API_KEY "
            "(preferred) or OPENAI_API_KEY."
        )

    async with httpx.AsyncClient(timeout=300) as c:
        # Single-shot Whisper for small files — avoids ffmpeg overhead.
        if len(audio_bytes) <= _CHUNK_MAX_BYTES:
            return await _whisper_one(c, audio_bytes, filename, openai_api_key)

        # Chunk + parallel transcribe via Whisper (legacy path when only
        # Whisper is configured). Deepgram avoids this entirely.
        logger.info("Audio %s is %.1fMB — chunking before Whisper",
                    filename, len(audio_bytes) / 1024 / 1024)
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(None, _chunk_audio_via_ffmpeg, audio_bytes, filename)
        logger.info("Split into %d chunks", len(chunks))

        results = await asyncio.gather(
            *(_whisper_one(c, b, name, openai_api_key) for (b, name) in chunks),
            return_exceptions=True,
        )

        transcript_parts = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error("Chunk %d failed: %s", i + 1, r)
                transcript_parts.append(f"[chunk {i+1} transcription failed]")
            else:
                transcript_parts.append(r)
        return "\n\n".join(transcript_parts)


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
    openai_api_key: str = "",
    deepgram_api_key: str = "",
    deepgram_model: str = "nova-3",
    fast_model: str = "claude-haiku-4-5",
) -> dict[str, Any]:
    """End-to-end: transcribe + categorize + persist. Returns the Meeting
    row's updated state + the raw extracted suggestions.

    Deepgram is preferred when configured (no size cap, speaker diarization).
    Whisper is the fallback (with pydub/ffmpeg chunking for >22MB files).
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

    # Step 1 — transcribe (Deepgram preferred, Whisper fallback)
    try:
        transcript = await transcribe_audio(
            audio_bytes, filename,
            openai_api_key=openai_api_key,
            deepgram_api_key=deepgram_api_key,
            deepgram_model=deepgram_model,
        )
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
