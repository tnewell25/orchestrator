"""MicrosoftCalendarSkill — read + create Outlook/Microsoft 365 calendar events.

Mirrors CalendarSkill (Google) so the agent picks whichever provider is
connected. Both can be active simultaneously — the agent will list events
from both when asked "what's on my calendar today".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ..core.skill_base import Skill, tool
from ..interfaces.microsoft_auth import get_valid_access_token

GRAPH = "https://graph.microsoft.com/v1.0"


class MicrosoftCalendarSkill(Skill):
    name = "ms_calendar"
    description = (
        "Read and schedule Microsoft 365 / Outlook calendar events. Use when "
        "the user is on Outlook (Microsoft shop) — distinct from Google "
        "Calendar."
    )

    async def _token(self) -> str | None:
        return await get_valid_access_token()

    async def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        token = await self._token()
        if not token:
            return {"error": "Microsoft not connected. Visit /settings/integrations."}
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{GRAPH}{path}", headers={"Authorization": f"Bearer {token}"}, params=params or {})
            if r.status_code != 200:
                return {"error": f"Graph error {r.status_code}: {r.text[:200]}"}
            return r.json()

    async def _post(self, path: str, body: dict) -> dict[str, Any]:
        token = await self._token()
        if not token:
            return {"error": "Microsoft not connected."}
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{GRAPH}{path}", headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=body)
            if r.status_code not in (200, 201):
                return {"error": f"Graph error {r.status_code}: {r.text[:200]}"}
            return r.json()

    @tool("List today's Microsoft 365 / Outlook calendar events.")
    async def list_today(self) -> list[dict]:
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return await self._range(start, end)

    @tool("List upcoming Microsoft 365 / Outlook events in the next N days (default 7).")
    async def list_upcoming(self, days: int = 7) -> list[dict]:
        now = datetime.now(timezone.utc)
        return await self._range(now, now + timedelta(days=days))

    async def _range(self, start: datetime, end: datetime) -> list[dict]:
        params = {
            "startDateTime": start.isoformat(),
            "endDateTime": end.isoformat(),
            "$select": "id,subject,start,end,attendees,location,bodyPreview",
            "$orderby": "start/dateTime",
            "$top": 50,
        }
        data = await self._get("/me/calendarView", params=params)
        if "error" in data:
            return [data]
        out = []
        for e in data.get("value", []):
            out.append({
                "id": e["id"],
                "title": e.get("subject", ""),
                "start": e.get("start", {}).get("dateTime", ""),
                "end": e.get("end", {}).get("dateTime", ""),
                "attendees": ", ".join(
                    a.get("emailAddress", {}).get("address", "")
                    for a in e.get("attendees", [])
                ),
                "location": e.get("location", {}).get("displayName", ""),
            })
        return out

    @tool(
        "Create an Outlook calendar event. start and end in ISO 'YYYY-MM-DDTHH:MM:SS' "
        "(UTC assumed). attendee_emails is comma-separated."
    )
    async def create_event(
        self, title: str, start: str, end: str,
        attendee_emails: str = "", description: str = "", location: str = "",
    ) -> dict:
        attendees = [
            {"emailAddress": {"address": e.strip()}, "type": "required"}
            for e in attendee_emails.split(",") if e.strip()
        ]
        body = {
            "subject": title,
            "start": {"dateTime": start, "timeZone": "UTC"},
            "end": {"dateTime": end, "timeZone": "UTC"},
            "body": {"contentType": "HTML", "content": description or ""},
            "attendees": attendees,
        }
        if location:
            body["location"] = {"displayName": location}
        result = await self._post("/me/events", body)
        if "error" in result:
            return result
        return {"id": result["id"], "title": result.get("subject", ""), "web_link": result.get("webLink", "")}

    @tool("Get details for a specific Outlook event by id.")
    async def get_event(self, event_id: str) -> dict:
        data = await self._get(f"/me/events/{event_id}")
        if "error" in data:
            return data
        return {
            "id": data["id"], "title": data.get("subject", ""),
            "start": data.get("start", {}).get("dateTime", ""),
            "end": data.get("end", {}).get("dateTime", ""),
            "attendees": ", ".join(
                a.get("emailAddress", {}).get("address", "")
                for a in data.get("attendees", [])
            ),
            "body": data.get("body", {}).get("content", ""),
            "location": data.get("location", {}).get("displayName", ""),
        }
