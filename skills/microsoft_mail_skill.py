"""MicrosoftMailSkill — read, draft, and send Outlook mail via Microsoft Graph.

Mirrors GmailSkill so the agent picks whichever provider is connected.
Uses delegated permissions (Mail.ReadWrite, Mail.Send) — sends as the
authenticated user.
"""
from __future__ import annotations

from typing import Any

import httpx

from ..core.skill_base import Skill, tool
from ..interfaces.microsoft_auth import get_valid_access_token

GRAPH = "https://graph.microsoft.com/v1.0"


class MicrosoftMailSkill(Skill):
    name = "ms_mail"
    description = (
        "Read, draft, and send Outlook / Microsoft 365 mail. Use this for "
        "Microsoft shops; GmailSkill handles Google."
    )

    async def _token(self) -> str | None:
        return await get_valid_access_token()

    async def _request(self, method: str, path: str, **kw) -> dict[str, Any]:
        token = await self._token()
        if not token:
            return {"error": "Microsoft not connected. Visit /settings/integrations."}
        async with httpx.AsyncClient(timeout=15) as c:
            kw.setdefault("headers", {})["Authorization"] = f"Bearer {token}"
            r = await c.request(method, f"{GRAPH}{path}", **kw)
            if r.status_code >= 400:
                return {"error": f"Graph error {r.status_code}: {r.text[:200]}"}
            if r.status_code == 204 or not r.text:
                return {"ok": True}
            return r.json()

    @tool("List recent unread Outlook emails (newest first). max_results caps; default 10.")
    async def list_unread(self, max_results: int = 10) -> list[dict]:
        data = await self._request(
            "GET",
            "/me/mailFolders/inbox/messages",
            params={
                "$filter": "isRead eq false",
                "$top": max(1, min(max_results, 50)),
                "$orderby": "receivedDateTime desc",
                "$select": "id,subject,from,receivedDateTime,bodyPreview",
            },
        )
        if "error" in data:
            return [data]
        out = []
        for m in data.get("value", []):
            out.append({
                "id": m["id"],
                "from": m.get("from", {}).get("emailAddress", {}).get("address", ""),
                "from_name": m.get("from", {}).get("emailAddress", {}).get("name", ""),
                "subject": m.get("subject", ""),
                "received": m.get("receivedDateTime", ""),
                "snippet": (m.get("bodyPreview", "") or "")[:200],
            })
        return out

    @tool("Get the full body of an Outlook email by message id.")
    async def get_message(self, message_id: str) -> dict:
        data = await self._request(
            "GET", f"/me/messages/{message_id}",
            params={"$select": "id,subject,from,toRecipients,receivedDateTime,body"},
        )
        if "error" in data:
            return data
        return {
            "id": data["id"], "subject": data.get("subject", ""),
            "from": data.get("from", {}).get("emailAddress", {}).get("address", ""),
            "to": ", ".join(
                r.get("emailAddress", {}).get("address", "")
                for r in data.get("toRecipients", [])
            ),
            "received": data.get("receivedDateTime", ""),
            "body": data.get("body", {}).get("content", ""),
        }

    @tool(
        "Create an Outlook draft email (does NOT send). Returns the draft id "
        "so the user can review on phone before sending."
    )
    async def create_draft(self, to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> dict:
        msg = {
            "subject": subject,
            "body": {"contentType": "HTML", "content": body},
            "toRecipients": _addrs(to),
        }
        if cc: msg["ccRecipients"] = _addrs(cc)
        if bcc: msg["bccRecipients"] = _addrs(bcc)
        data = await self._request("POST", "/me/messages", json=msg)
        if "error" in data:
            return data
        return {"id": data["id"], "subject": data.get("subject", ""), "web_link": data.get("webLink", "")}

    @tool(
        "Send an Outlook email immediately. ONLY use when the user has "
        "explicitly said 'send it' — otherwise prefer create_draft."
    )
    async def send(self, to: str, subject: str, body: str, cc: str = "", bcc: str = "") -> dict:
        msg = {
            "subject": subject,
            "body": {"contentType": "HTML", "content": body},
            "toRecipients": _addrs(to),
        }
        if cc: msg["ccRecipients"] = _addrs(cc)
        if bcc: msg["bccRecipients"] = _addrs(bcc)
        data = await self._request("POST", "/me/sendMail", json={"message": msg, "saveToSentItems": True})
        if "error" in data:
            return data
        return {"sent": True}

    @tool("Mark an Outlook message as read.")
    async def mark_read(self, message_id: str) -> dict:
        data = await self._request("PATCH", f"/me/messages/{message_id}", json={"isRead": True})
        return data


def _addrs(s: str) -> list[dict]:
    return [
        {"emailAddress": {"address": e.strip()}}
        for e in (s or "").split(",") if e.strip()
    ]
