"""Claude Max OAuth token lifecycle.

Port of Tethyr's token_manager. Uses the Anthropic public OAuth client_id
for Claude Max subscriptions — refreshes access tokens automatically and
persists to DB so restarts stay authenticated.
"""
import asyncio
import logging
import time

import aiohttp

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_REFRESH_BUFFER_S = 300


class TokenManager:
    def __init__(self, memory, settings):
        self.memory = memory
        self.settings = settings

        self.access_token: str = ""
        self.refresh_token: str = ""
        self.expires_at: float = 0
        self.api_key: str = settings.anthropic_api_key
        self.source: str = ""

        self._lock = asyncio.Lock()

    @property
    def mode(self) -> str:
        if self.access_token:
            return "oauth"
        if self.api_key:
            return "api_key"
        return "none"

    async def initialize(self):
        """Load tokens from DB, seeding from env vars on first boot.

        If ANTHROPIC_API_KEY is set AND no OAuth env vars are set, we skip
        loading DB OAuth tokens — API key always wins. This avoids a stale
        OAuth token in the DB forcing the agent into a broken OAuth path.
        """
        env_access = self.settings.anthropic_auth_token
        if self.api_key and not env_access:
            logger.info("API key present, no OAuth env → API key mode (skipping DB OAuth)")
            return

        row = await self.memory.get_oauth_token("anthropic")
        if row and row["access_token"]:
            self.access_token = row["access_token"]
            self.refresh_token = row["refresh_token"]
            self.expires_at = row["expires_at"] or 0
            self.source = row["source"]
            logger.info(
                "Loaded OAuth tokens from DB (source=%s, tail=...%s)",
                self.source,
                self.access_token[-8:] if self.access_token else "none",
            )
            return

        env_refresh = self.settings.anthropic_refresh_token
        if env_access:
            self.access_token = env_access
            self.refresh_token = env_refresh
            self.expires_at = 0
            self.source = "env"
            await self.memory.upsert_oauth_token(
                provider="anthropic",
                access_token=env_access,
                refresh_token=env_refresh,
                expires_at=None,
                client_id=_CLIENT_ID,
                source="env",
            )
            logger.info("Seeded OAuth tokens from env vars → DB")
        else:
            logger.info("No OAuth tokens found — using API key mode")

    async def ensure_token(self):
        """Refresh token if expired or about to expire."""
        if not self.refresh_token:
            return
        now = time.time()
        if self.expires_at > 0 and now < (self.expires_at - _REFRESH_BUFFER_S):
            return
        if self.expires_at == 0 and self.access_token:
            return  # first use; let 401 trigger refresh
        await self._refresh()

    async def _refresh(self) -> bool:
        async with self._lock:
            now = time.time()
            if self.expires_at > 0 and now < (self.expires_at - _REFRESH_BUFFER_S):
                return True

            logger.info("Refreshing OAuth access token...")
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        _TOKEN_URL,
                        data={
                            "grant_type": "refresh_token",
                            "refresh_token": self.refresh_token,
                            "client_id": _CLIENT_ID,
                        },
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                    ) as resp:
                        body = await resp.json()
                        if resp.status >= 400:
                            error = body.get("error", body.get("message", str(body)))
                            logger.error(
                                "Token refresh failed (%d): %s", resp.status, error
                            )
                            return False

                        new_access = body["access_token"]
                        new_refresh = body.get("refresh_token", self.refresh_token)
                        expires_in = body.get("expires_in", 3600)
                        new_expires_at = time.time() + expires_in

                        await self.memory.upsert_oauth_token(
                            provider="anthropic",
                            access_token=new_access,
                            refresh_token=new_refresh,
                            expires_at=new_expires_at,
                            client_id=_CLIENT_ID,
                            source="refresh",
                        )

                        self.access_token = new_access
                        self.refresh_token = new_refresh
                        self.expires_at = new_expires_at
                        self.source = "refresh"
                        logger.info(
                            "OAuth token refreshed (expires in %ds)", expires_in
                        )
                        return True
            except Exception as e:
                logger.error("Token refresh error: %s", e)
                return False

    async def handle_auth_error(self) -> bool:
        """Recovery chain on 401: refresh → API key fallback."""
        if self.refresh_token and await self._refresh():
            return True
        if self.api_key:
            logger.warning("OAuth recovery failed — falling back to API key")
            self.access_token = ""
            self.refresh_token = ""
            self.expires_at = 0
            self.source = "api_key_fallback"
            return True
        logger.error("All token recovery methods exhausted")
        return False

    async def inject_tokens(
        self, access_token: str, refresh_token: str, source: str = "manual"
    ):
        async with self._lock:
            await self.memory.upsert_oauth_token(
                provider="anthropic",
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=None,
                client_id=_CLIENT_ID,
                source=source,
            )
            self.access_token = access_token
            self.refresh_token = refresh_token
            self.expires_at = 0
            self.source = source

    def get_token_status(self) -> dict:
        now = time.time()
        status = {"mode": self.mode, "source": self.source}
        if self.access_token:
            status["access_token_tail"] = "..." + self.access_token[-8:]
            if self.expires_at > 0:
                status["expires_in_seconds"] = int(self.expires_at - now)
        return status
