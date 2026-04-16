"""GmailSkill — compose drafts, send, triage inbox.

OAuth flow:
  - On first use: InstalledAppFlow opens a local browser, user grants consent,
    token saved to GOOGLE_TOKEN_PATH (default ~/.orchestrator/google_token.json).
  - On subsequent runs: token auto-refreshes.

Scopes: gmail.modify (read + draft + send).
"""
import asyncio
import base64
import logging
import os
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from ..core.skill_base import Safety, Skill, tool

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]

DEFAULT_TOKEN_PATH = os.path.expanduser("~/.orchestrator/google_token.json")


class GmailSkill(Skill):
    name = "gmail"
    description = "Read, draft, and send Gmail."

    def __init__(self, credentials_path: str, token_path: str = DEFAULT_TOKEN_PATH):
        super().__init__()
        self.credentials_path = credentials_path
        self.token_path = token_path
        self._service = None

    async def setup(self):
        if not self.credentials_path or not os.path.exists(self.credentials_path):
            logger.warning("Gmail credentials not configured — skill disabled")
            return
        creds = await asyncio.get_event_loop().run_in_executor(None, self._load_creds)
        self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)

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
            raise RuntimeError("Gmail skill not configured — set GOOGLE_CREDENTIALS_PATH")

    # -- tools --

    @tool(
        "List recent unread emails (newest first). Returns id, from, subject, snippet. "
        "Use max_results to cap; default 10."
    )
    async def list_unread(self, max_results: int = 10) -> list[dict]:
        self._require()

        def _fetch():
            result = self._service.users().messages().list(
                userId="me", labelIds=["UNREAD", "INBOX"], maxResults=max_results
            ).execute()
            msgs = result.get("messages", [])
            out = []
            for m in msgs:
                detail = self._service.users().messages().get(
                    userId="me", id=m["id"], format="metadata",
                    metadataHeaders=["From", "Subject", "Date"]
                ).execute()
                headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
                out.append({
                    "id": m["id"],
                    "from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", ""),
                    "snippet": detail.get("snippet", ""),
                })
            return out

        return await asyncio.get_event_loop().run_in_executor(None, _fetch)

    @tool("Get the full body of an email by message_id.")
    async def get_message(self, message_id: str) -> dict:
        self._require()

        def _fetch():
            msg = self._service.users().messages().get(
                userId="me", id=message_id, format="full"
            ).execute()
            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
            body = self._extract_body(msg.get("payload", {}))
            return {
                "id": message_id,
                "from": headers.get("From", ""),
                "to": headers.get("To", ""),
                "subject": headers.get("Subject", ""),
                "date": headers.get("Date", ""),
                "body": body[:10000],  # cap
            }

        return await asyncio.get_event_loop().run_in_executor(None, _fetch)

    def _extract_body(self, payload: dict) -> str:
        if payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
        for part in payload.get("parts", []):
            if part.get("mimeType", "").startswith("text/"):
                data = part.get("body", {}).get("data")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
            inner = self._extract_body(part)
            if inner:
                return inner
        return ""

    @tool(
        "Create a draft email (does NOT send). Returns the draft id. "
        "Use this so the user can review on phone before sending."
    )
    async def create_draft(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
    ) -> dict:
        self._require()

        def _create():
            msg = MIMEText(body)
            msg["to"] = to
            msg["subject"] = subject
            if cc:
                msg["cc"] = cc
            if bcc:
                msg["bcc"] = bcc
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            draft = self._service.users().drafts().create(
                userId="me", body={"message": {"raw": raw}}
            ).execute()
            return {"draft_id": draft["id"], "to": to, "subject": subject}

        return await asyncio.get_event_loop().run_in_executor(None, _create)

    @tool(
        "Send an email immediately. Only use when the user has explicitly said "
        "'send it' — otherwise prefer create_draft.",
        safety=Safety.APPROVE_EXTERNAL,
    )
    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
    ) -> dict:
        self._require()

        def _send():
            msg = MIMEText(body)
            msg["to"] = to
            msg["subject"] = subject
            if cc:
                msg["cc"] = cc
            if bcc:
                msg["bcc"] = bcc
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            sent = self._service.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()
            return {"message_id": sent["id"], "to": to, "subject": subject, "sent": True}

        return await asyncio.get_event_loop().run_in_executor(None, _send)

    @tool("Mark a message as read (removes UNREAD label).")
    async def mark_read(self, message_id: str) -> dict:
        self._require()

        def _mark():
            self._service.users().messages().modify(
                userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            return {"message_id": message_id, "marked_read": True}

        return await asyncio.get_event_loop().run_in_executor(None, _mark)
