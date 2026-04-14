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
from .core.memory import MemoryStore
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
from .skills.job_skill import JobSkill
from .skills.meeting_skill import MeetingSkill
from .skills.proposal_skill import ProposalSkill
from .skills.reminder_skill import ReminderSkill
from .skills.research_skill import ResearchSkill
from .skills.stakeholder_skill import StakeholderSkill
from .skills.task_skill import TaskSkill
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
    global memory, agent, token_manager, audit_logger, telegram_bot, reminder_service, bot_task

    warnings = settings.validate_critical()
    for w in warnings:
        logger.warning("CONFIG: %s", w)

    # 1. Memory / DB
    memory = MemoryStore(
        database_url=settings.database_url,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )
    await memory.initialize()
    logger.info("Memory initialized")

    audit_logger = AuditLogger(memory.session_maker)

    # 2. OAuth token manager
    token_manager = TokenManager(memory, settings)
    await token_manager.initialize()
    logger.info("Token manager initialized (mode=%s)", token_manager.mode)

    # 3. Default chat id for reminders — first allowed user (owner)
    default_chat_id = (
        str(settings.allowed_user_ids[0]) if settings.allowed_user_ids else ""
    )

    # 4. Skills — CRM + productivity + field ops + intelligence
    sm = memory.session_maker
    gmail_skill = GmailSkill(settings.google_credentials_path)
    calendar_skill = CalendarSkill(settings.google_credentials_path)

    skills = [
        # CRM core
        CompanySkill(sm),
        ContactSkill(sm),
        DealSkill(sm),
        TaskSkill(sm),
        MeetingSkill(sm),
        BidSkill(sm, default_chat_id=default_chat_id),
        # Productivity
        ReminderSkill(sm, default_chat_id=default_chat_id),
        BriefingSkill(sm),
        WeeklyReviewSkill(sm),
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
        # Field ops (v2 direction — slotted in)
        JobSkill(sm),
    ]
    for s in skills:
        await s.setup()

    # 5. Agent
    agent = Agent(
        memory=memory,
        skills=skills,
        settings=settings,
        token_manager=token_manager,
        audit_logger=audit_logger,
    )
    logger.info("Agent initialized with %d tools", len(agent.tools))

    # 6. Telegram bot
    if settings.telegram_bot_token:
        telegram_bot = TelegramBot(agent, settings)
        await telegram_bot.start()
    else:
        logger.warning("No TELEGRAM_BOT_TOKEN — bot not started")

    # 7. Background services
    reminder_service = ReminderService(sm, telegram_bot, agent=agent)
    await reminder_service.start()

    calendar_sync = CalendarAutoSync(
        sm, calendar_skill, default_chat_id=default_chat_id
    )
    await calendar_sync.start()

    proactive_monitor = ProactiveMonitor(sm, default_chat_id, settings)
    await proactive_monitor.start()

    yield

    # Shutdown — reverse order
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


@app.post("/chat")
async def chat(body: Prompt):
    """HTTP entrypoint for testing without Telegram."""
    if not agent:
        raise HTTPException(503, "not ready")
    response = await agent.run(
        body.message, session_id=body.session_id, interface="http"
    )
    return {"response": response}
