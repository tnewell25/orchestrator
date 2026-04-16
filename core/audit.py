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
        safety: str = "auto",
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
                        safety=safety,
                    )
                )
                await session.commit()
        except Exception as e:
            logger.error("Audit log failed: %s", e)

    async def log_usage(
        self,
        session_id: str,
        model: str,
        usage,                # anthropic.types.Usage or dict-like
        duration_ms: int = 0,
        iteration: int = 0,
    ):
        """Record one messages.create round-trip's token cost.

        `usage` can be an Anthropic SDK Usage object or a dict with the same
        field names. Missing fields default to 0 so partial data still logs."""
        try:
            def _get(k: str) -> int:
                if usage is None:
                    return 0
                v = getattr(usage, k, None)
                if v is None and isinstance(usage, dict):
                    v = usage.get(k)
                return int(v or 0)

            async with self.session_maker() as session:
                session.add(
                    AuditLog(
                        tool_name="_turn",
                        args_summary=f"iter={iteration}",
                        result_status="ok",
                        result_summary="",
                        session_id=session_id,
                        duration_ms=duration_ms,
                        model=model,
                        input_tokens=_get("input_tokens"),
                        output_tokens=_get("output_tokens"),
                        cache_read_tokens=_get("cache_read_input_tokens"),
                        cache_creation_tokens=_get("cache_creation_input_tokens"),
                    )
                )
                await session.commit()
        except Exception as e:
            logger.error("Usage log failed: %s", e)
