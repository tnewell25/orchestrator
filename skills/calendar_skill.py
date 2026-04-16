"""CalendarSkill — read Google Calendar + create events.

Shares the same OAuth token as GmailSkill (different scope set though — we
request calendar scope alongside gmail so the single consent covers both).
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from ..core.skill_base import Skill, tool

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
]

DEFAULT_TOKEN_PATH = os.path.expanduser("~/.orchestrator/google_token.json")


class CalendarSkill(Skill):
    name = "calendar"
    description = "Read and schedule Google Calendar events."

    def __init__(self, credentials_path: str, token_path: str = DEFAULT_TOKEN_PATH):
        super().__init__()
        self.credentials_path = credentials_path
        self.token_path = token_path
        self._service = None

    async def setup(self):
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            logger.warning("Calendar credentials not configured — skill disabled")
            return
        creds = await asyncio.get_event_loop().run_in_executor(None, self._load_creds)
        self._service = build("calendar", "v3", credentials=creds, cache_discovery=False)

    def _load_creds(self) -> Credentials:
        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
            with open(self.token_path, "w") as f:
                f.write(creds.to_json())
        return creds

    def _require(self):
        if not self._service:
            raise RuntimeError("Calendar skill not configured — set GOOGLE_CREDENTIALS_PATH")

    @tool("List today's calendar events.")
    async def list_today(self) -> list[dict]:
        return await self.list_upcoming(days=1)

    @tool("List upcoming events in the next N days (default 7). Returns id, title, start, end, attendees.")
    async def list_upcoming(self, days: int = 7) -> list[dict]:
        self._require()

        def _fetch():
            now = datetime.now(timezone.utc)
            end = now + timedelta(days=days)
            result = self._service.events().list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=end.isoformat(),
                singleEvents=True,
                orderBy="startTime",
                maxResults=50,
            ).execute()
            events = result.get("items", [])
            out = []
            for e in events:
                start = e.get("start", {}).get("dateTime") or e.get("start", {}).get("date")
                end_ = e.get("end", {}).get("dateTime") or e.get("end", {}).get("date")
                attendees = [a.get("email", "") for a in e.get("attendees", [])]
                out.append({
                    "id": e.get("id"),
                    "title": e.get("summary", "(no title)"),
                    "start": start,
                    "end": end_,
                    "location": e.get("location", ""),
                    "description": (e.get("description", "") or "")[:500],
                    "attendees": attendees,
                    "organizer": e.get("organizer", {}).get("email", ""),
                })
            return out

        return await asyncio.get_event_loop().run_in_executor(None, _fetch)

    @tool(
        "Create a calendar event. start and end in ISO format 'YYYY-MM-DDTHH:MM:SS' "
        "(UTC assumed if no tz). attendee_emails is comma-separated.",
        safety="approve_external",
    )
    async def create_event(
        self,
        title: str,
        start: str,
        end: str,
        attendee_emails: str = "",
        description: str = "",
        location: str = "",
    ) -> dict:
        self._require()

        def _create():
            attendees = []
            if attendee_emails:
                attendees = [{"email": e.strip()} for e in attendee_emails.split(",") if e.strip()]
            body = {
                "summary": title,
                "location": location,
                "description": description,
                "start": {"dateTime": start, "timeZone": "UTC"},
                "end": {"dateTime": end, "timeZone": "UTC"},
                "attendees": attendees,
            }
            event = self._service.events().insert(calendarId="primary", body=body).execute()
            return {"event_id": event["id"], "html_link": event.get("htmlLink", "")}

        return await asyncio.get_event_loop().run_in_executor(None, _create)

    @tool("Get details for a specific event by id.")
    async def get_event(self, event_id: str) -> dict:
        self._require()

        def _fetch():
            e = self._service.events().get(calendarId="primary", eventId=event_id).execute()
            return {
                "id": e.get("id"),
                "title": e.get("summary", ""),
                "start": e.get("start", {}),
                "end": e.get("end", {}),
                "description": e.get("description", ""),
                "attendees": [a.get("email", "") for a in e.get("attendees", [])],
                "location": e.get("location", ""),
            }

        return await asyncio.get_event_loop().run_in_executor(None, _fetch)
