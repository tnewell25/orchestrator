"""HTTP webhook receivers for Gmail Pub/Sub + Calendar push.

These are FastAPI sub-routers mounted under /webhooks. Each receives Google's
push notification, optionally verifies, and publishes an Event onto the bus.

Setup (out of band):
- Gmail: gmail.users.watch() with topicName, register Pub/Sub push subscription
  pointing at /webhooks/gmail
- Calendar: calendar.events.watch() with address=https://your-domain/webhooks/calendar

The webhooks stay TINY — they translate HTTP → bus.publish() and let rules
do the actual work.
"""
from __future__ import annotations

import base64
import json
import logging

from fastapi import APIRouter, Header, HTTPException, Request

from ..core.events import EventBus, EventType

logger = logging.getLogger(__name__)


def build_webhook_router(bus: EventBus, expected_token: str = "") -> APIRouter:
    """Construct the webhook router bound to a specific event bus.

    expected_token: if set, X-Webhook-Token header must match (cheap auth for
    when you can't easily configure mutual TLS). Leave empty for no check.
    """
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])

    def _verify(token_header: str | None):
        if expected_token and token_header != expected_token:
            raise HTTPException(401, "invalid webhook token")

    @router.post("/gmail")
    async def gmail_push(
        request: Request,
        x_webhook_token: str | None = Header(default=None),
    ):
        """Gmail Pub/Sub message envelope:
            {"message": {"data": <base64-json>, "messageId": "...", ...},
             "subscription": "..."}
        """
        _verify(x_webhook_token)
        try:
            envelope = await request.json()
        except Exception:
            raise HTTPException(400, "invalid json")

        msg = (envelope or {}).get("message", {})
        encoded = msg.get("data", "")
        decoded: dict = {}
        if encoded:
            try:
                decoded = json.loads(base64.b64decode(encoded).decode("utf-8"))
            except Exception as e:
                logger.warning("Could not decode Gmail push data: %s", e)

        await bus.publish(
            EventType.EMAIL_RECEIVED,
            payload={
                "history_id": decoded.get("historyId"),
                "email_address": decoded.get("emailAddress"),
                "raw": decoded,
            },
            source="webhook.gmail",
        )
        return {"ok": True}

    @router.post("/calendar")
    async def calendar_push(
        request: Request,
        x_goog_resource_state: str | None = Header(default=None),
        x_goog_channel_id: str | None = Header(default=None),
        x_goog_resource_id: str | None = Header(default=None),
        x_webhook_token: str | None = Header(default=None),
    ):
        """Google Calendar push notifications use response headers to convey
        state. Body is empty for most events."""
        _verify(x_webhook_token)

        # First notification on a new channel is a sync ping — ignore it.
        if x_goog_resource_state == "sync":
            return {"ok": True, "ack": "sync"}

        await bus.publish(
            EventType.CALENDAR_EVENT_UPDATED,
            payload={
                "channel_id": x_goog_channel_id,
                "resource_id": x_goog_resource_id,
                "state": x_goog_resource_state,
            },
            source="webhook.calendar",
        )
        return {"ok": True}

    return router
