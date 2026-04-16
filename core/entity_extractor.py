"""Write-time entity extraction — links new memories/messages to known entities.

Two paths:
  1. SUBSTRING — fast, deterministic, free. Whole-word match against a cached
     index of known Contact/Company/Deal names. Catches the easy 80%.
  2. LLM (Haiku) — handles pronouns, partial names, and proposes NEW entities
     ("Anja from Bosch" when Anja isn't in the contact DB yet). Optional;
     skipped if no Anthropic credentials.

The substring path is run synchronously inside store_memory. The LLM path runs
in the background via asyncio.create_task so user-facing latency stays flat.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..db.models import Company, Contact, Deal
from .constants import EdgeKind, EntityType
from .graph import EntityRef, GraphStore

logger = logging.getLogger(__name__)


# Words ignored as entity candidates even if they happen to match an alias
_STOP_NAMES = {
    "the", "and", "for", "you", "they", "them", "this", "that", "with",
    "from", "have", "has", "his", "her", "him", "she", "our", "your",
}


@dataclass
class ExtractedEntity:
    ref: EntityRef
    matched_text: str         # exact substring (or pronoun phrase) that triggered the link
    confidence: float = 1.0
    is_new: bool = False      # True if the LLM proposed creating this entity (not yet in DB)
    proposed_name: str = ""   # populated when is_new


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    edges: list[tuple[EntityRef, EntityRef, str]] = field(default_factory=list)

    def add(self, source_ref: EntityRef, target: ExtractedEntity, kind: str = EdgeKind.MENTIONS):
        self.entities.append(target)
        if not target.is_new:
            self.edges.append((source_ref, target.ref, kind))


class EntityIndex:
    """In-memory cache of known entity names → refs. Rebuilt periodically.

    Held by EntityExtractor. Not thread-safe across processes; that's fine
    because the agent runs in a single asyncio loop per worker.
    """

    def __init__(self):
        # Lowercased name (or alias) → list of refs that share that name
        self._name_to_refs: dict[str, list[EntityRef]] = {}

    def add(self, name: str, ref: EntityRef):
        if not name:
            return
        key = name.strip().lower()
        if not key or key in _STOP_NAMES:
            return
        self._name_to_refs.setdefault(key, []).append(ref)

    def lookup_substring(self, text: str) -> list[ExtractedEntity]:
        """Whole-word case-insensitive substring search.

        Returns one ExtractedEntity per UNIQUE ref matched. If a name maps to
        multiple entities (two contacts both named "Markus"), all are returned
        with reduced confidence — the LLM disambiguator can resolve later.
        """
        if not text or not self._name_to_refs:
            return []

        lowered = text.lower()
        out: dict[EntityRef, ExtractedEntity] = {}

        for name, refs in self._name_to_refs.items():
            # \b matches word boundary — avoids "Mark" matching inside "Markus"
            try:
                pattern = re.compile(r"\b" + re.escape(name) + r"\b")
            except re.error:
                continue
            if not pattern.search(lowered):
                continue
            confidence = 0.95 if len(refs) == 1 else max(0.4, 0.95 / len(refs))
            for ref in refs:
                if ref in out:
                    continue
                out[ref] = ExtractedEntity(
                    ref=ref, matched_text=name, confidence=confidence
                )
        return list(out.values())

    def known_names_summary(self, limit: int = 200) -> str:
        """Compact list for LLM prompting: 'Contacts: Anja Weber, Markus ... | Deals: ...'"""
        by_type: dict[str, list[str]] = {}
        for name, refs in self._name_to_refs.items():
            for ref in refs:
                by_type.setdefault(ref.type, []).append(name)
        lines = []
        for t, names in by_type.items():
            unique = sorted(set(names))[:limit]
            lines.append(f"{t}s: {', '.join(unique)}")
        return " | ".join(lines)


class EntityExtractor:
    """Extracts entity references from free text and creates graph edges.

    Lifecycle:
        ext = EntityExtractor(session_maker, graph, anthropic_client=client)
        await ext.refresh_index()         # call at startup + after big writes
        result = await ext.extract(text, source_ref)  # synchronous substring path
        # If LLM path enabled, call ext.extract_llm_async(text, source_ref) as a task
    """

    def __init__(
        self,
        session_maker: async_sessionmaker,
        graph: GraphStore,
        anthropic_client=None,
        fast_model: str = "claude-haiku-4-5-20251001",
    ):
        self.sm = session_maker
        self.graph = graph
        self.client = anthropic_client
        self.fast_model = fast_model
        self.index = EntityIndex()
        self._refresh_lock = asyncio.Lock()

    async def refresh_index(self) -> int:
        """Rebuild the name→ref cache from the DB. Returns total names indexed."""
        async with self._refresh_lock:
            new_index = EntityIndex()
            async with self.sm() as session:
                contacts = (await session.execute(select(Contact.id, Contact.name))).all()
                for cid, cname in contacts:
                    new_index.add(cname, EntityRef(EntityType.CONTACT, cid))
                    # First-name alias for fast pronoun-adjacent matching
                    first = (cname or "").strip().split(" ", 1)[0]
                    if first and first.lower() != (cname or "").strip().lower():
                        new_index.add(first, EntityRef(EntityType.CONTACT, cid))

                companies = (await session.execute(select(Company.id, Company.name))).all()
                for cid, cname in companies:
                    new_index.add(cname, EntityRef(EntityType.COMPANY, cid))

                deals = (await session.execute(select(Deal.id, Deal.name))).all()
                for did, dname in deals:
                    new_index.add(dname, EntityRef(EntityType.DEAL, did))

            self.index = new_index
            return sum(len(v) for v in new_index._name_to_refs.values())

    async def extract(
        self,
        text: str,
        source_ref: EntityRef,
        edge_kind: str = EdgeKind.MENTIONS,
        write_edges: bool = True,
    ) -> ExtractionResult:
        """Synchronous substring extraction + edge writes.

        Idempotent: re-running on the same text reinforces existing edges
        (count++) rather than creating duplicates.
        """
        result = ExtractionResult()
        matches = self.index.lookup_substring(text)
        for m in matches:
            result.add(source_ref, m, kind=edge_kind)

        if write_edges:
            for from_ref, to_ref, kind in result.edges:
                try:
                    await self.graph.add_edge(
                        from_ref, to_ref, kind,
                        confidence=next(
                            (m.confidence for m in matches if m.ref == to_ref), 1.0
                        ),
                        source="auto_extract",
                    )
                except Exception as e:
                    logger.warning("Edge write failed (%s → %s, %s): %s",
                                   from_ref, to_ref, kind, e)

        return result

    async def extract_llm(
        self,
        text: str,
        source_ref: EntityRef,
        edge_kind: str = EdgeKind.MENTIONS,
        write_edges: bool = True,
    ) -> ExtractionResult:
        """Smarter extraction via Haiku. Catches pronouns + proposes new entities.

        Returns an ExtractionResult that may include both linked-to-existing AND
        is_new=True candidates (the agent decides whether to create them).
        Falls back to substring extraction if no LLM client is configured.
        """
        if not self.client:
            return await self.extract(text, source_ref, edge_kind, write_edges)

        prompt = _build_extraction_prompt(text, self.index.known_names_summary())
        try:
            resp = await self.client.messages.create(
                model=self.fast_model,
                max_tokens=600,
                system=_EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(b.text for b in resp.content if b.type == "text").strip()
            parsed = _parse_llm_response(raw)
        except Exception as e:
            logger.warning("LLM extraction failed, falling back to substring: %s", e)
            return await self.extract(text, source_ref, edge_kind, write_edges)

        result = ExtractionResult()
        for item in parsed:
            kind_str = (item.get("kind") or "").strip()
            id_str = (item.get("id") or "").strip()
            name_str = (item.get("name") or "").strip()
            confidence = float(item.get("confidence") or 0.7)
            matched = (item.get("matched") or name_str).strip()

            if not kind_str or kind_str not in EntityType.ALL:
                continue

            if id_str:
                ent = ExtractedEntity(
                    ref=EntityRef(kind_str, id_str),
                    matched_text=matched, confidence=confidence,
                )
            else:
                # New entity proposal — surface but DON'T auto-create.
                # Skill code (or the agent itself) decides whether to materialize.
                ent = ExtractedEntity(
                    ref=EntityRef(kind_str, ""),
                    matched_text=matched, confidence=confidence,
                    is_new=True, proposed_name=name_str,
                )
            result.add(source_ref, ent, kind=edge_kind)

        if write_edges:
            for from_ref, to_ref, kind in result.edges:
                try:
                    await self.graph.add_edge(
                        from_ref, to_ref, kind,
                        confidence=next(
                            (e.confidence for e in result.entities if e.ref == to_ref), 0.7
                        ),
                        source="llm_extract",
                    )
                except Exception as e:
                    logger.warning("LLM edge write failed: %s", e)

        return result

    async def extract_llm_background(
        self,
        text: str,
        source_ref: EntityRef,
        edge_kind: str = EdgeKind.MENTIONS,
    ) -> asyncio.Task:
        """Fire-and-forget LLM extraction. Returns the task so callers can await
        it during tests; production code can ignore the return value."""
        return asyncio.create_task(
            self.extract_llm(text, source_ref, edge_kind, write_edges=True)
        )


# ----------------------------------------------------------------------
# LLM prompt + parsing
# ----------------------------------------------------------------------


_EXTRACTION_SYSTEM = """You are an entity extraction service for a sales CRM.
Read the input text and identify references to people (Contact), companies
(Company), or deals (Deal).

Output ONLY a JSON array. No prose, no markdown fences. Each element:
  {
    "kind": "Contact" | "Company" | "Deal",
    "id": "<uuid if you can confidently match a known entity, else empty>",
    "name": "<canonical name>",
    "matched": "<exact substring that triggered the match>",
    "confidence": 0.0-1.0
  }

Rules:
- If the text mentions a known entity by name, use its id.
- If a pronoun ("he", "she") clearly refers to a known entity in context, link it.
- If a NEW person/company is mentioned (not in the known list), include with id="".
- Skip generic mentions ("the customer", "our team") with no proper noun.
- Be precise — wrong links are worse than missed links."""


def _build_extraction_prompt(text: str, known_summary: str) -> str:
    return (
        f"Known entities (id NOT shown — match by name):\n{known_summary or '(none)'}\n\n"
        f"Input text:\n{text}\n\n"
        f"Return JSON array."
    )


def _parse_llm_response(raw: str) -> list[dict]:
    """Tolerant JSON parser — handles bare arrays and accidental fences."""
    if not raw:
        return []
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # Strip optional language tag
        if "\n" in cleaned:
            cleaned = cleaned.split("\n", 1)[1]
    cleaned = cleaned.strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to locate the outermost [...] in case the model added prose
        m = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


__all__ = [
    "EntityExtractor",
    "EntityIndex",
    "ExtractedEntity",
    "ExtractionResult",
]
