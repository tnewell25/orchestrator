"""Microsoft Graph OAuth 2.0 authorization-code flow.

Handles:
- Building the authorize URL (user clicks → Microsoft consent screen)
- Exchanging the callback code for access + refresh tokens
- Refreshing tokens before they expire
- Storing tokens via the same OAuthToken row used by Google (provider="microsoft")

Tokens are scoped for delegated user access — Calendars.ReadWrite, Mail.ReadWrite,
Mail.Send, User.Read, offline_access (the last one is what gets us a refresh
token; without it, sessions expire after an hour).
"""
from __future__ import annotations

import logging
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth/microsoft", tags=["microsoft-auth"])

_sm = None
_settings = None

# In-memory state store — short-lived CSRF tokens for the OAuth dance. Mapping
# of state → created_at. Restart-safe-ish because a stale state just means the
# user re-clicks Connect.
_pending_state: dict[str, float] = {}

SCOPES = [
    "offline_access",      # required for refresh_token
    "User.Read",
    "Calendars.ReadWrite",
    "Mail.ReadWrite",
    "Mail.Send",
]


def mount_microsoft_auth(app, session_maker, settings):
    global _sm, _settings
    _sm = session_maker
    _settings = settings
    app.include_router(router)


def _redirect_uri() -> str:
    base = (_settings.app_base_url or "").rstrip("/")
    return f"{base}/auth/microsoft/callback"


def _authority() -> str:
    return f"https://login.microsoftonline.com/{_settings.microsoft_tenant or 'common'}"


def _is_configured() -> bool:
    return bool(_settings and _settings.microsoft_client_id and _settings.microsoft_client_secret)


@router.get("/login")
async def login(state: str = ""):
    """Build the Microsoft authorize URL and redirect the user to it."""
    if not _is_configured():
        raise HTTPException(503, "Microsoft Graph not configured. Set MICROSOFT_CLIENT_ID + MICROSOFT_CLIENT_SECRET.")
    s = state or secrets.token_urlsafe(24)
    _pending_state[s] = time.time()
    # Garbage-collect old states (>1h)
    cutoff = time.time() - 3600
    for k in list(_pending_state.keys()):
        if _pending_state[k] < cutoff:
            _pending_state.pop(k, None)

    params = {
        "client_id": _settings.microsoft_client_id,
        "response_type": "code",
        "redirect_uri": _redirect_uri(),
        "response_mode": "query",
        "scope": " ".join(SCOPES),
        "state": s,
        "prompt": "select_account",
    }
    url = f"{_authority()}/oauth2/v2.0/authorize?{urlencode(params)}"
    return RedirectResponse(url, status_code=302)


@router.get("/callback")
async def callback(code: str = Query(""), state: str = Query(""), error: str = Query(""), error_description: str = Query("")):
    """Microsoft redirects here after consent. Exchange code → tokens, store, return success page."""
    if error:
        return HTMLResponse(_html_result(False, f"Microsoft error: {error}", error_description), status_code=400)
    if not code or not state or state not in _pending_state:
        return HTMLResponse(_html_result(False, "Invalid state — try Connect again", ""), status_code=400)
    _pending_state.pop(state, None)

    try:
        tokens = await _exchange_code_for_tokens(code)
    except Exception as e:
        logger.exception("Microsoft token exchange failed")
        return HTMLResponse(_html_result(False, "Token exchange failed", str(e)[:500]), status_code=500)

    await _store_tokens(tokens)
    logger.info("Microsoft tokens stored — user %s connected", tokens.get("user_principal_name", "?"))
    return HTMLResponse(_html_result(True, "Microsoft connected", "You can close this tab."))


async def _exchange_code_for_tokens(code: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{_authority()}/oauth2/v2.0/token",
            data={
                "client_id": _settings.microsoft_client_id,
                "client_secret": _settings.microsoft_client_secret,
                "code": code,
                "redirect_uri": _redirect_uri(),
                "grant_type": "authorization_code",
                "scope": " ".join(SCOPES),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        body = resp.json()

    # Best-effort identity lookup so we can label which account is connected
    upn = ""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            me = await client.get(
                "https://graph.microsoft.com/v1.0/me",
                headers={"Authorization": f"Bearer {body['access_token']}"},
            )
            if me.status_code == 200:
                upn = me.json().get("userPrincipalName", "")
    except Exception:
        pass

    return {
        "access_token": body["access_token"],
        "refresh_token": body.get("refresh_token", ""),
        "expires_at": time.time() + int(body.get("expires_in", 3600)) - 60,  # 60s safety margin
        "user_principal_name": upn,
    }


async def _store_tokens(tokens: dict[str, Any]):
    from ..db.models import OAuthToken
    async with _sm() as s:
        existing = await s.get(OAuthToken, "microsoft")
        if existing:
            existing.access_token = tokens["access_token"]
            existing.refresh_token = tokens["refresh_token"] or existing.refresh_token
            existing.expires_at = tokens["expires_at"]
            existing.client_id = _settings.microsoft_client_id
            existing.source = "oauth_user"
        else:
            s.add(OAuthToken(
                provider="microsoft",
                access_token=tokens["access_token"],
                refresh_token=tokens["refresh_token"],
                expires_at=tokens["expires_at"],
                client_id=_settings.microsoft_client_id,
                source="oauth_user",
            ))
        await s.commit()


async def get_valid_access_token() -> str | None:
    """Return a non-expired access token, refreshing if needed. None if not connected."""
    if not _is_configured():
        return None
    from ..db.models import OAuthToken
    async with _sm() as s:
        row = await s.get(OAuthToken, "microsoft")
        if not row:
            return None
        if row.expires_at and row.expires_at > time.time():
            return row.access_token
        if not row.refresh_token:
            return None
    # Refresh
    try:
        new_tokens = await _refresh_tokens(row.refresh_token)
        await _store_tokens(new_tokens)
        return new_tokens["access_token"]
    except Exception as e:
        logger.error("Microsoft token refresh failed: %s", e)
        return None


async def _refresh_tokens(refresh_token: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{_authority()}/oauth2/v2.0/token",
            data={
                "client_id": _settings.microsoft_client_id,
                "client_secret": _settings.microsoft_client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
                "scope": " ".join(SCOPES),
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        body = resp.json()
    return {
        "access_token": body["access_token"],
        "refresh_token": body.get("refresh_token", refresh_token),
        "expires_at": time.time() + int(body.get("expires_in", 3600)) - 60,
        "user_principal_name": "",
    }


async def disconnect():
    """Drop the stored Microsoft tokens."""
    from ..db.models import OAuthToken
    async with _sm() as s:
        row = await s.get(OAuthToken, "microsoft")
        if row:
            await s.delete(row)
            await s.commit()


async def status() -> dict[str, Any]:
    """Connection status for the integrations UI."""
    from ..db.models import OAuthToken
    if not _is_configured():
        return {
            "configured": False,
            "connected": False,
            "message": "Set MICROSOFT_CLIENT_ID + MICROSOFT_CLIENT_SECRET to enable.",
        }
    async with _sm() as s:
        row = await s.get(OAuthToken, "microsoft")
        if not row:
            return {"configured": True, "connected": False}
        return {
            "configured": True,
            "connected": True,
            "expires_at": row.expires_at,
            "needs_refresh": bool(row.expires_at and row.expires_at < time.time()),
            "redirect_uri": _redirect_uri(),
        }


def _html_result(ok: bool, title: str, detail: str) -> str:
    color = "#10b981" if ok else "#ef4444"
    icon = "✓" if ok else "✗"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; background: #f5f6f8;
       display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }}
.card {{ background: white; border: 1px solid #e2e5ea; border-radius: 8px;
        padding: 32px 40px; max-width: 420px; text-align: center;
        box-shadow: 0 4px 12px rgba(15,23,42,0.08); }}
.icon {{ width: 56px; height: 56px; border-radius: 28px; background: {color};
        color: white; font-size: 28px; line-height: 56px; display: inline-block; margin-bottom: 12px; }}
h1 {{ font-size: 16px; color: #0a0e1a; margin: 0 0 8px; }}
p {{ font-size: 13px; color: #6b7280; margin: 0; line-height: 1.5; }}
</style></head><body><div class="card">
<div class="icon">{icon}</div><h1>{title}</h1><p>{detail}</p>
</div></body></html>"""
