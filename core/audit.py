"""AuditLogger — records every tool call with status/duration for debugging and trust."""
import json
import logging

from ..db.models import AuditLog

logger = logging.getLogger(__name__)


class AuditLogger:
    def __init__(self, session_maker):
        self.session_maker = session_maker

    async def log(
        self,
        tool_name: str,
        args: dict,
        result_status: str,
        result_summary: str,
        session_id: str = "",
        duration_ms: int = 0,
    ):
        try:
            args_str = json.dumps(args, default=str)[:500]
            async with self.session_maker() as session:
                session.add(
                    AuditLog(
                        tool_name=tool_name,
                        args_summary=args_str,
                        result_status=result_status,
                        result_summary=result_summary[:500],
                        session_id=session_id,
                        duration_ms=duration_ms,
                    )
                )
                await session.commit()
        except Exception as e:
            logger.error("Audit log failed: %s", e)
