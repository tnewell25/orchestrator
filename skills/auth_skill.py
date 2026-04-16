"""AuthSkill — manage the Telegram allow-list from chat.

Flow:
1. Unauthorized user messages the bot; bot replies with their Telegram user_id.
2. Owner says "add Brian, id 123456789" → agent calls auth-add_user.
3. Brian messages again — now authorized.

The env TELEGRAM_ALLOWED_USERS list is the root trust (owners). This skill
manages the DB-backed extension. Owners in env cannot be revoked via chat.
"""
from datetime import datetime, timezone

from sqlalchemy import select

from ..core.skill_base import Safety, Skill, tool
from ..db.models import AuthorizedUser


class AuthSkill(Skill):
    name = "auth"
    description = "Manage the Telegram allow-list (add/list/revoke collaborators)."

    def __init__(self, session_maker):
        super().__init__()
        self.session_maker = session_maker

    @tool(
        "Grant a Telegram user access to the bot. telegram_user_id is a "
        "numeric string (from the 'Not authorized' reply the user sees). "
        "name is a short label so the allow-list stays human-readable.",
        safety=Safety.CONFIRM,
    )
    async def add_user(self, telegram_user_id: str, name: str = "") -> dict:
        tid = str(telegram_user_id).strip()
        if not tid.isdigit():
            return {"error": f"telegram_user_id must be numeric, got '{tid}'"}

        async with self.session_maker() as s:
            existing = await s.get(AuthorizedUser, tid)
            if existing is not None:
                existing.active = "yes"
                existing.revoked_at = None
                if name:
                    existing.name = name
                await s.commit()
                return {
                    "telegram_user_id": tid,
                    "name": existing.name,
                    "status": "reactivated" if existing.revoked_at else "updated",
                }
            s.add(AuthorizedUser(
                telegram_user_id=tid,
                name=name or f"user-{tid[-4:]}",
                role="member",
                active="yes",
            ))
            await s.commit()
            return {"telegram_user_id": tid, "name": name or f"user-{tid[-4:]}",
                    "status": "added"}

    @tool("List currently authorized users (both env-owners and DB-added members).")
    async def list_users(self) -> dict:
        async with self.session_maker() as s:
            rows = (
                await s.execute(
                    select(AuthorizedUser)
                    .where(AuthorizedUser.active == "yes")
                    .order_by(AuthorizedUser.added_at)
                )
            ).scalars().all()
        return {
            "members": [
                {
                    "telegram_user_id": r.telegram_user_id,
                    "name": r.name,
                    "role": r.role,
                    "added_at": str(r.added_at),
                }
                for r in rows
            ],
            "note": "Env TELEGRAM_ALLOWED_USERS owners are not shown here but remain authorized.",
        }

    @tool(
        "Revoke a user's access. telegram_user_id is the numeric string. "
        "Does not remove env-var owners; only DB-added members.",
        safety=Safety.CONFIRM,
    )
    async def revoke_user(self, telegram_user_id: str) -> dict:
        tid = str(telegram_user_id).strip()
        async with self.session_maker() as s:
            row = await s.get(AuthorizedUser, tid)
            if row is None:
                return {"error": f"No DB-authorized user with id {tid}"}
            row.active = "no"
            row.revoked_at = datetime.now(timezone.utc)
            await s.commit()
            return {"telegram_user_id": tid, "name": row.name, "status": "revoked"}
