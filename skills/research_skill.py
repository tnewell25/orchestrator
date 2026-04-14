"""ResearchSkill — web-based company/exec/competitive deep dives.

Uses Serper (google search) if SERPER_API_KEY is set; otherwise falls back to
direct DuckDuckGo HTML scraping. Fetches top results, extracts text, returns
structured findings. The agent can then synthesize these into a brief.

Light on magic — heavy on returning raw material the agent weaves together.
"""
import asyncio
import logging
from urllib.parse import quote_plus

import aiohttp
from bs4 import BeautifulSoup

from ..core.skill_base import Skill, tool

logger = logging.getLogger(__name__)

SERPER_URL = "https://google.serper.dev/search"
DDG_URL = "https://html.duckduckgo.com/html/?q="


class ResearchSkill(Skill):
    name = "research"
    description = "Web research — company deep dives, exec bios, competitive analysis."

    def __init__(self, serper_api_key: str = ""):
        super().__init__()
        self.serper_api_key = serper_api_key

    async def _search(self, query: str, num: int = 10) -> list[dict]:
        """Returns list of {title, link, snippet}."""
        if self.serper_api_key:
            return await self._serper_search(query, num)
        return await self._ddg_search(query, num)

    async def _serper_search(self, query: str, num: int) -> list[dict]:
        headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": num}
        async with aiohttp.ClientSession() as s:
            async with s.post(SERPER_URL, json=payload, headers=headers, timeout=15) as r:
                data = await r.json()
        out = []
        for item in data.get("organic", [])[:num]:
            out.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return out

    async def _ddg_search(self, query: str, num: int) -> list[dict]:
        async with aiohttp.ClientSession() as s:
            async with s.get(DDG_URL + quote_plus(query), timeout=15) as r:
                html = await r.text()
        soup = BeautifulSoup(html, "html.parser")
        out = []
        for a in soup.select("a.result__a")[:num]:
            link = a.get("href", "")
            title = a.get_text(strip=True)
            parent = a.find_parent("div", class_="result")
            snippet = ""
            if parent:
                snip_el = parent.select_one(".result__snippet")
                if snip_el:
                    snippet = snip_el.get_text(strip=True)
            out.append({"title": title, "link": link, "snippet": snippet})
        return out

    async def _fetch_page_text(self, url: str, limit: int = 8000) -> str:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}) as r:
                    html = await r.text()
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
            return text[:limit]
        except Exception as e:
            logger.warning("Fetch failed for %s: %s", url, e)
            return ""

    @tool(
        "Web search. Returns top N results with title, link, snippet. "
        "Use this when you need to look something up — exec bios, company news, "
        "competitor feature comparisons, regulatory changes, etc."
    )
    async def search(self, query: str, num: int = 8) -> list[dict]:
        return await self._search(query, num)

    @tool(
        "Deep dive on a company. Pulls: recent news, leadership page, about page. "
        "Returns raw search snippets and extracted text chunks the agent can synthesize. "
        "Use for 'research Honeywell Aerospace strategy' type questions."
    )
    async def company_deepdive(self, company_name: str) -> dict:
        queries = [
            f"{company_name} latest news 2026",
            f"{company_name} leadership team",
            f"{company_name} strategy recent announcements",
        ]
        all_results = await asyncio.gather(*[self._search(q, 5) for q in queries])
        # Fetch top-2 URLs for page text
        page_texts = []
        seen = set()
        for rs in all_results:
            for item in rs[:2]:
                link = item.get("link")
                if link and link not in seen:
                    seen.add(link)
                    text = await self._fetch_page_text(link, 3000)
                    if text:
                        page_texts.append({"url": link, "text": text})
        return {
            "company": company_name,
            "search_results": {q: rs for q, rs in zip(queries, all_results)},
            "page_extracts": page_texts[:5],
        }

    @tool("Exec bio lookup. Pulls LinkedIn + recent mentions for a named exec at a company.")
    async def exec_bio(self, name: str, company: str) -> dict:
        queries = [
            f"{name} {company} linkedin",
            f"{name} {company} background biography",
            f"{name} {company} recent interview OR keynote",
        ]
        results = await asyncio.gather(*[self._search(q, 5) for q in queries])
        return {
            "name": name,
            "company": company,
            "linkedin_and_bio": results[0],
            "background": results[1],
            "recent_speaking": results[2],
        }

    @tool(
        "Compare our offering vs a named competitor for a specific use case. "
        "Pulls competitor's own marketing + critical reviews + alternatives."
    )
    async def competitive_analysis(self, competitor: str, use_case: str) -> dict:
        queries = [
            f"{competitor} {use_case} features pricing",
            f"{competitor} reviews criticism 2026",
            f"{competitor} vs alternatives {use_case}",
            f"{competitor} customer complaints {use_case}",
        ]
        results = await asyncio.gather(*[self._search(q, 5) for q in queries])
        return {
            "competitor": competitor,
            "use_case": use_case,
            "features_pricing": results[0],
            "criticism": results[1],
            "alternatives": results[2],
            "complaints": results[3],
        }
