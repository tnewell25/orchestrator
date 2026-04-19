"""FastAPI entrypoint — lifespan wires memory, token manager, agent, skills, bot."""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import get_settings
from .core.agent import Agent
from .core.audit import AuditLogger
from .core.calendar_sync import CalendarAutoSync
from .core.action_gate import ActionGate
from .core.compactor import Compactor
from .core.entity_extractor import EntityExtractor
from .core.events import EventBus
from .core.graph import GraphStore
from .core.job_queue import JobQueue
from .core.memory import MemoryStore
from .core.pipeline_watcher import PipelineWatcher
from .core.planner import Planner
from .core.rule_engine import ActionDispatcher, RuleEngine
from .core.scheduler_tick import SchedulerTick
from .core.strategy_fanout import StrategyFanout
from .interfaces.dashboard_api import mount_dashboard_api
from .interfaces.microsoft_auth import mount_microsoft_auth
from .interfaces.webhooks import build_webhook_router
from .core.proactive_monitor import ProactiveMonitor
from .core.reminder_service import ReminderService
from .core.token_manager import TokenManager
from .interfaces.telegram_bot import TelegramBot
from .skills.bid_skill import BidSkill
from .skills.briefing_skill import BriefingSkill
from .skills.calendar_skill import CalendarSkill
from .skills.company_skill import CompanySkill
from .skills.competitor_skill import CompetitorSkill
from .skills.contact_skill import ContactSkill
from .skills.deal_health_skill import DealHealthSkill
from .skills.deal_skill import DealSkill
from .skills.email_triage_skill import EmailTriageSkill
from .skills.gmail_skill import GmailSkill
from .skills.graph_skill import GraphSkill
from .skills.microsoft_calendar_skill import MicrosoftCalendarSkill
from .skills.microsoft_mail_skill import MicrosoftMailSkill
from .skills.job_skill import JobSkill
from .skills.meeting_skill import MeetingSkill
from .skills.proposal_skill import ProposalSkill
from .skills.reminder_skill import ReminderSkill
from .skills.research_skill import ResearchSkill
from .skills.stakeholder_skill import StakeholderSkill
from .skills.auth_skill import AuthSkill
from .skills.task_skill import TaskSkill
from .skills.usage_skill import UsageSkill
from .skills.weekly_review_skill import WeeklyReviewSkill

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

memory: MemoryStore | None = None
agent: Agent | None = None
token_manager: TokenManager | None = None
audit_logger: AuditLogger | None = None
telegram_bot: TelegramBot | None = None
reminder_service: ReminderService | None = None
calendar_sync: CalendarAutoSync | None = None
proactive_monitor: ProactiveMonitor | None = None
bot_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory, agent, token_manager, audit_logger, telegram_bot, reminder_service, calendar_sync, proactive_monitor, bot_task

    logger.info("=== Orchestrator startup ===")
    warnings = settings.validate_critical()
    for w in warnings:
        logger.warning("CONFIG: %s", w)

    # Degraded-mode fallback — /health still responds so Railway healthcheck passes.
    if not settings.database_url:
        logger.error("DATABASE_URL missing — DEGRADED mode (agent disabled)")
        yield
        return

    # 1. Memory / DB
    try:
        memory = MemoryStore(
            database_url=settings.database_url,
            embedding_model=settings.embedding_model,
            embedding_dim=settings.embedding_dim,
        )
        await memory.initialize()
        logger.info("Memory initialized")
    except Exception as e:
        logger.exception("Memory init failed — DEGRADED: %s", e)
        yield
        return

    audit_logger = AuditLogger(memory.session_maker)

    # 2. OAuth token manager
    token_manager = TokenManager(memory, settings)
    await token_manager.initialize()
    logger.info("Token manager initialized (mode=%s)", token_manager.mode)

    # 2b. Knowledge graph + write-time entity extractor.
    # Built before skills because skills (future) may want to query the graph;
    # extractor's LLM client uses the same auth path as the main agent.
    graph = GraphStore(memory.session_maker)
    extractor_client = None
    if token_manager.access_token:
        import anthropic
        extractor_client = anthropic.AsyncAnthropic(
            auth_token=token_manager.access_token,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
            max_retries=3, timeout=30.0,
        )
    elif token_manager.api_key or settings.anthropic_api_key:
        import anthropic
        extractor_client = anthropic.AsyncAnthropic(
            api_key=token_manager.api_key or settings.anthropic_api_key,
            max_retries=3, timeout=30.0,
        )
    extractor = EntityExtractor(
        memory.session_maker, graph,
        anthropic_client=extractor_client,
        fast_model=settings.fast_model,
    )
    n_indexed = await extractor.refresh_index()
    memory.attach_extractor(extractor, llm_background=bool(extractor_client))
    memory.attach_graph(graph)  # enables hybrid recall with proximity
    logger.info("Entity extractor ready — %d names indexed (llm=%s)",
                n_indexed, bool(extractor_client))

    # 3. Default chat id for reminders — first allowed user (owner)
    default_chat_id = (
        str(settings.allowed_user_ids[0]) if settings.allowed_user_ids else ""
    )

    # 4. Skills — CRM + productivity + field ops + intelligence
    sm = memory.session_maker
    gmail_skill = GmailSkill(settings.google_credentials_path)
    calendar_skill = CalendarSkill(settings.google_credentials_path)

    user_tz = getattr(settings, "user_timezone", "UTC")
    skills = [
        # CRM core
        CompanySkill(sm),
        ContactSkill(sm),
        DealSkill(sm),
        TaskSkill(sm),
        MeetingSkill(sm, memory=memory),
        BidSkill(sm, default_chat_id=default_chat_id, user_timezone=user_tz),
        # Knowledge graph queries
        GraphSkill(sm, graph),
        # Productivity
        ReminderSkill(sm, default_chat_id=default_chat_id, user_timezone=user_tz),
        BriefingSkill(sm),
        WeeklyReviewSkill(sm),
        UsageSkill(sm),
        AuthSkill(sm),
        # Pipeline intelligence
        StakeholderSkill(sm),
        DealHealthSkill(sm),
        CompetitorSkill(sm, memory),
        ProposalSkill(sm, memory),
        # External
        ResearchSkill(settings.serper_api_key),
        gmail_skill,
        calendar_skill,
        EmailTriageSkill(sm, gmail_skill=gmail_skill),
        # Microsoft Graph (Outlook + Calendar). The skills check for a
        # connected token at call time and return a helpful message when
        # not connected, so registering them unconditionally is safe.
        MicrosoftCalendarSkill(),
        MicrosoftMailSkill(),
        # Field ops (v2 direction — slotted in)
        JobSkill(sm),
    ]
    for s in skills:
        await s.setup()

    # 5. Agent — with planner if we have an LLM client; lazy tool loading
    # so the always-loaded prompt prefix stays small and cache-stable.
    planner = Planner(extractor_client, fast_model=settings.fast_model) if extractor_client else None
    # Durable job queue + compactor wired through it so a crash mid-Haiku-call
    # doesn't silently drop summarization work.
    job_queue = JobQueue(memory.session_maker)
    recovered = await job_queue.recover_stuck()
    if recovered:
        logger.warning("Reset %d orphaned background jobs from previous run", recovered)
    compactor = (
        Compactor(
            memory.session_maker, extractor_client,
            fast_model=settings.fast_model, job_queue=job_queue,
        ) if extractor_client else None
    )
    app.state.job_queue = job_queue
    action_gate = ActionGate(memory.session_maker)
    # Create event bus BEFORE agent so cost alerts can publish from the first turn.
    event_bus = EventBus()
    strategy_fanout = (
        StrategyFanout(memory.session_maker, extractor_client, fast_model=settings.fast_model)
        if extractor_client else None
    )
    agent = Agent(
        memory=memory,
        skills=skills,
        settings=settings,
        token_manager=token_manager,
        audit_logger=audit_logger,
        planner=planner,
        entity_extractor=extractor,
        lazy_tools=True,
        compactor=compactor,
        action_gate=action_gate,
        event_bus=event_bus,
        strategy_fanout=strategy_fanout,
    )
    app.state.action_gate = action_gate
    logger.info(
        "Agent initialized — full catalog %d tools, lazy mode on (planner=%s)",
        len(agent.tools), bool(planner),
    )

    # 5b. Rule engine — reuses the EventBus created above so cost alerts
    # published from the agent feed into declarative rules.
    dispatcher = ActionDispatcher(sm, event_bus)
    rule_engine = RuleEngine(event_bus, sm, dispatcher)
    rule_engine.register_builtins()
    app.state.event_bus = event_bus
    app.state.rule_engine = rule_engine

    # Mount webhook receivers so Gmail Pub/Sub and Calendar push translate
    # into bus events. expected_token comes from settings if you want auth;
    # safer to use mutual TLS or a JWT verifier in front of this.
    app.include_router(build_webhook_router(event_bus))
    mount_dashboard_api(app, sm)
    mount_microsoft_auth(app, sm, settings)
    logger.info("EventBus + rule engine ready, webhooks + dashboard API mounted")

    # 5c. Pipeline Watcher sub-agent — runs nightly via DAILY_SWEEP.
    if extractor_client is not None:
        watcher = PipelineWatcher(
            sm, extractor_client, dispatcher,
            fast_model=settings.fast_model, default_chat_id=default_chat_id,
        )
        watcher.attach_to_bus(event_bus)
        app.state.pipeline_watcher = watcher
        logger.info("PipelineWatcher attached to DAILY_SWEEP")

    # 6. Telegram bot (optional — fail soft)
    if settings.telegram_bot_token:
        try:
            telegram_bot = TelegramBot(agent, settings, session_maker=sm)
            await telegram_bot.start()
        except Exception as e:
            logger.exception("Telegram bot failed to start: %s", e)
            telegram_bot = None
    else:
        logger.warning("No TELEGRAM_BOT_TOKEN — bot not started")

    # 7. Background services (fail soft)
    try:
        reminder_service = ReminderService(sm, telegram_bot, agent=agent)
        await reminder_service.start()
    except Exception as e:
        logger.exception("ReminderService failed: %s", e)

    try:
        calendar_sync = CalendarAutoSync(
            sm, calendar_skill, default_chat_id=default_chat_id
        )
        await calendar_sync.start()
    except Exception as e:
        logger.exception("CalendarAutoSync failed: %s", e)

    try:
        proactive_monitor = ProactiveMonitor(sm, default_chat_id, settings)
        await proactive_monitor.start()
    except Exception as e:
        logger.exception("ProactiveMonitor failed: %s", e)

    # Scheduler tick — fires HOURLY/DAILY events the rule engine reacts to.
    scheduler_tick: SchedulerTick | None = None
    try:
        scheduler_tick = SchedulerTick(
            event_bus, default_chat_id=default_chat_id,
            sweep_payload_extra={
                "stalled_deal_days": settings.stalled_deal_days,
            },
        )
        await scheduler_tick.start()
        app.state.scheduler_tick = scheduler_tick
    except Exception as e:
        logger.exception("SchedulerTick failed: %s", e)

    logger.info("=== Orchestrator ready ===")
    yield

    # Shutdown — reverse order
    if scheduler_tick:
        await scheduler_tick.stop()
    if proactive_monitor:
        await proactive_monitor.stop()
    if calendar_sync:
        await calendar_sync.stop()
    if reminder_service:
        await reminder_service.stop()
    if telegram_bot:
        await telegram_bot.stop()
    if memory:
        await memory.close()


app = FastAPI(title="Orchestrator", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_name": settings.agent_name,
        "llm_mode": token_manager.mode if token_manager else "uninit",
        "tools": len(agent.tools) if agent else 0,
    }


@app.get("/status")
async def status():
    if not token_manager:
        raise HTTPException(503, "not ready")
    return {
        "token": token_manager.get_token_status(),
        "tools": [t["name"] for t in (agent.tools if agent else [])],
    }


class TokenPush(BaseModel):
    access_token: str
    refresh_token: str


@app.post("/tokens")
async def push_tokens(body: TokenPush):
    """Inject fresh OAuth tokens (e.g. synced from ~/.claude/.credentials.json)."""
    if not token_manager:
        raise HTTPException(503, "not ready")
    await token_manager.inject_tokens(
        body.access_token, body.refresh_token, source="manual"
    )
    if agent:
        agent._init_client()
    return {"ok": True, "mode": token_manager.mode}


class Prompt(BaseModel):
    message: str
    session_id: str = "http-default"


# Approximate per-model pricing in USD per 1M tokens. Source: Anthropic's
# public pricing page — update if rates change. The /usage endpoint is for
# rough visibility; console.anthropic.com/usage is the authoritative bill.
PRICING = {
    "claude-sonnet-4-5-20250929": {
        "input": 3.00, "output": 15.00,
        "cache_read": 0.30, "cache_creation": 3.75,
    },
    "claude-sonnet-4-5": {  # alias
        "input": 3.00, "output": 15.00,
        "cache_read": 0.30, "cache_creation": 3.75,
    },
    "claude-haiku-4-5-20251001": {
        "input": 1.00, "output": 5.00,
        "cache_read": 0.10, "cache_creation": 1.25,
    },
    "claude-haiku-4-5": {
        "input": 1.00, "output": 5.00,
        "cache_read": 0.10, "cache_creation": 1.25,
    },
}
_DEFAULT_PRICING = {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_creation": 3.75}


def _cost_for(model: str, tokens: dict) -> float:
    p = PRICING.get(model, _DEFAULT_PRICING)
    return (
        tokens.get("input_tokens", 0) * p["input"] / 1_000_000
        + tokens.get("output_tokens", 0) * p["output"] / 1_000_000
        + tokens.get("cache_read_tokens", 0) * p["cache_read"] / 1_000_000
        + tokens.get("cache_creation_tokens", 0) * p["cache_creation"] / 1_000_000
    )


@app.get("/usage")
async def usage(hours: int = 24):
    """Aggregated token + cost usage from AuditLog over a rolling window.

    hours=24 (default): last 24 hours. Use hours=168 for a week, hours=0 for
    all time. Cost is estimated using public pricing; authoritative source is
    the Anthropic console.
    """
    if not memory:
        raise HTTPException(503, "not ready")
    from datetime import datetime, timedelta, timezone
    from sqlalchemy import select, func
    from .db.models import AuditLog

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours) if hours > 0 else None

    async with memory.session_maker() as s:
        q = select(
            AuditLog.model,
            func.count().label("turns"),
            func.sum(AuditLog.input_tokens).label("input_tokens"),
            func.sum(AuditLog.output_tokens).label("output_tokens"),
            func.sum(AuditLog.cache_read_tokens).label("cache_read_tokens"),
            func.sum(AuditLog.cache_creation_tokens).label("cache_creation_tokens"),
        ).where(AuditLog.tool_name == "_turn")
        if cutoff:
            q = q.where(AuditLog.timestamp >= cutoff)
        q = q.group_by(AuditLog.model)

        rows = (await s.execute(q)).all()

    by_model: dict[str, dict] = {}
    total_turns = 0
    total_cost = 0.0
    total_input = 0
    total_cache_read = 0

    for row in rows:
        model = row.model or "(unknown)"
        tokens = {
            "input_tokens": int(row.input_tokens or 0),
            "output_tokens": int(row.output_tokens or 0),
            "cache_read_tokens": int(row.cache_read_tokens or 0),
            "cache_creation_tokens": int(row.cache_creation_tokens or 0),
        }
        cost = _cost_for(model, tokens)
        denom = tokens["input_tokens"] + tokens["cache_read_tokens"] + tokens["cache_creation_tokens"]
        hit_ratio = (tokens["cache_read_tokens"] / denom) if denom else 0.0

        by_model[model] = {
            "turns": int(row.turns or 0),
            **tokens,
            "cache_hit_ratio": round(hit_ratio, 3),
            "estimated_cost_usd": round(cost, 4),
        }
        total_turns += int(row.turns or 0)
        total_cost += cost
        total_input += tokens["input_tokens"]
        total_cache_read += tokens["cache_read_tokens"]

    # Cache savings: what the cache_read tokens would have cost at the full
    # input rate, minus what they actually cost.
    savings = 0.0
    for model, data in by_model.items():
        p = PRICING.get(model, _DEFAULT_PRICING)
        savings += data["cache_read_tokens"] * (p["input"] - p["cache_read"]) / 1_000_000

    return {
        "window_hours": hours if hours > 0 else "all",
        "total_turns": total_turns,
        "total_estimated_cost_usd": round(total_cost, 4),
        "cache_savings_vs_uncached_usd": round(savings, 4),
        "by_model": by_model,
        "note": "Authoritative billing: https://console.anthropic.com/usage",
    }


@app.get("/usage/recent")
async def usage_recent(limit: int = 20):
    """Raw per-turn rows, newest first. Useful for spot-checking what happened
    in the last few interactions."""
    if not memory:
        raise HTTPException(503, "not ready")
    from sqlalchemy import select
    from .db.models import AuditLog

    async with memory.session_maker() as s:
        q = (
            select(AuditLog)
            .where(AuditLog.tool_name == "_turn")
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )
        rows = (await s.execute(q)).scalars().all()
    return {
        "items": [
            {
                "timestamp": str(r.timestamp),
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cache_read_tokens": r.cache_read_tokens,
                "cache_creation_tokens": r.cache_creation_tokens,
                "duration_ms": r.duration_ms,
                "session_id": r.session_id,
                "iteration": r.args_summary,
            }
            for r in rows
        ],
    }


@app.post("/chat")
async def chat(body: Prompt):
    """HTTP entrypoint for testing without Telegram."""
    if not agent:
        raise HTTPException(503, "not ready")
    response = await agent.run(
        body.message, session_id=body.session_id, interface="http"
    )
    return {"response": response}


# ---- Pending action approval API --------------------------------------


@app.get("/pending-actions")
async def list_pending_actions(session_id: str = ""):
    gate = getattr(app.state, "action_gate", None)
    if gate is None:
        raise HTTPException(503, "action gate not initialized")
    rows = await gate.list_pending(session_id=session_id)
    return {
        "items": [
            {
                "id": r.id, "session_id": r.session_id, "tool_name": r.tool_name,
                "summary": r.summary, "created_at": str(r.created_at),
                "expires_at": str(r.expires_at) if r.expires_at else None,
            }
            for r in rows
        ],
    }


@app.post("/pending-actions/{action_id}/approve")
async def approve_pending_action(action_id: str):
    gate = getattr(app.state, "action_gate", None)
    if gate is None or agent is None:
        raise HTTPException(503, "not ready")
    approved = await gate.approve(action_id)
    if approved is None:
        raise HTTPException(404, "action not found, already decided, or expired")

    # Execute the underlying tool now.
    import json as _json
    tool_input = _json.loads(approved.tool_input or "{}")
    try:
        result = await agent._execute_tool_bypass_gate(
            approved.tool_name, tool_input, session_id=approved.session_id,
        )
        await gate.mark_executed(action_id, str(result)[:500])
        return {"ok": True, "result": result}
    except Exception as e:
        await gate.mark_failed(action_id, str(e))
        raise HTTPException(500, f"execution failed: {e}")


@app.post("/pending-actions/{action_id}/reject")
async def reject_pending_action(action_id: str):
    gate = getattr(app.state, "action_gate", None)
    if gate is None:
        raise HTTPException(503, "action gate not initialized")
    rejected = await gate.reject(action_id)
    if rejected is None:
        raise HTTPException(404, "action not found or already decided")
    return {"ok": True, "id": action_id}
