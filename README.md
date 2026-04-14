# Orchestrator

AI chief of staff for sales engineers. Telegram bot that captures meetings (by voice), manages pipeline, tracks follow-ups, and never forgets a commitment.

Forked conceptually from [Tethyr](https://github.com/tnewell25/Tethyr) — same Anthropic tool-use loop + Claude Max OAuth + Postgres/pgvector memory, but domain-focused on CRM rather than ops automation.

## What it does (v1)

- **Voice capture** — send a voice note after a meeting; it gets transcribed via Whisper, summarized, filed to the right deal, and action items are auto-created with due dates.
- **Pipeline memory** — ask "what's going on with Honeywell?" and get recent meetings, open follow-ups, next step.
- **Follow-up engine** — every commitment ("I'll send him the spec by Friday") becomes a tracked action item.
- **Relationship memory** — personal notes (kids at MIT, hobbies) are stored with contacts and surfaced before meetings.
- **Daily brief** — `/brief` returns overdue items, deals closing this week, recent meetings.

## Architecture

```
Telegram (long-poll) ─┐
                      ├─► Agent (Anthropic tool-use loop, Claude Max OAuth)
HTTP /chat endpoint ──┘          │
                                 ├─► Skills (Company, Contact, Deal, Task, Meeting, Briefing)
                                 └─► MemoryStore (Postgres + pgvector, local fastembed)
```

All skills register their `@tool`-decorated methods into Claude's tool schema via reflection. Add a new skill by subclassing `Skill` and dropping it into `main.py`.

## Setup — local

```bash
# 1. Postgres with pgvector (docker)
docker run -d --name orchestrator-pg \
  -e POSTGRES_USER=orchestrator -e POSTGRES_PASSWORD=orchestrator \
  -e POSTGRES_DB=orchestrator -p 5432:5432 \
  pgvector/pgvector:pg16

# 2. Python env
python -m venv .venv
.venv\Scripts\activate   # Windows; use source .venv/bin/activate on mac/linux
pip install -r requirements.txt

# 3. .env
copy .env.example .env   # Windows; use cp on mac/linux
# Fill in TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS, DATABASE_URL, and either
# ANTHROPIC_AUTH_TOKEN + ANTHROPIC_REFRESH_TOKEN (Claude Max) or ANTHROPIC_API_KEY

# 4. Run (from the PARENT directory of the repo, since `orchestrator` is the package name)
cd ..
uvicorn orchestrator.main:app --reload
```

Then DM your Telegram bot `/start`.

## Claude Max OAuth

If you have a Claude Max subscription, you can use its OAuth tokens instead of an API key (no per-token billing). The tokens live in `~/.claude/.credentials.json` after you log into Claude Code. Extract `accessToken` and `refreshToken` and put them in `.env` as `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_REFRESH_TOKEN`. The token manager will refresh them automatically.

Alternatively, POST to `/tokens` at runtime to inject fresh tokens without a restart.

## Deploy — Railway

1. Create a Railway project + add Postgres plugin (it injects `DATABASE_URL`).
2. Enable pgvector on the database (one-time): `CREATE EXTENSION vector;` (the app also runs this at init).
3. Push this repo; Railway picks up `railway.toml` + `Dockerfile`.
4. Set env vars in Railway dashboard: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USERS`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_REFRESH_TOKEN`, `OPENAI_API_KEY` (for voice).

## Reminders (v1.1)

The agent can schedule proactive pings that arrive on Telegram at the right time, even after a restart (they're stored in Postgres and polled every 30s).

**User patterns:**
- "remind me in 2 hours to call Markus" → natural-language time parsing via `dateparser`
- "ping me 30 min before the Bosch meeting" → `reminder-set_pre_meeting` auto-triggers a full context brief when it fires (last meetings, open actions, personal notes on the attendee)
- Bid deadlines auto-schedule T-7d/T-3d/T-1d pings when created via `bid-create`

**Kinds:** `custom`, `pre_meeting` (agent-enriched brief), `bid_deadline`, `commitment`.

## MEDDIC discipline (v1.1)

Deals track the full MEDDIC stack: Metrics, Economic buyer, Decision Criteria, Decision Process, Paper Process, Identified pain, Champion, Competition. When you ask the agent about a deal, it flags the gaps — "economic buyer unknown" — so you know what to dig for next conversation.

## Bid/RFP tracking (v1.1)

Separate entity from Deal. Tracks submission + Q&A deadlines, deliverables, value. Every new bid auto-schedules the full reminder sequence. Mark submitted → remaining pings cancel automatically.

## Roadmap — elite features

### v1.1 (shipped now)
- Reminders with natural-language time + pre-meeting briefs
- Bid/RFP tracking with auto-countdown
- MEDDIC fields on Deals
- CRM + voice capture + daily brief
- Claude Max OAuth

### v1.2 (next)
- **Google Calendar integration** — reads calendar, auto-creates 30-min pre-meeting briefs for every meeting with a known contact
- **Email draft skill** — "follow up with Markus about dinner, attach CM whitepaper" → Gmail draft ready to send
- **Commitment drift detection** — "you told Markus you'd send the spec Fri. It's Mon." (via reminders + pending action items)
- **Photo/vision** — business cards → contact record, whiteboards → notes

### v1.5
- **Competitor battle-card library** — surfaces win-playbook when Siemens MindSphere / GE Smart Signal / PTC ThingWorx / Rockwell FactoryTalk mentioned
- **LinkedIn / company news intel** — auto-monitors pipeline companies for job changes, press releases, layoffs
- **Travel intelligence** — "you're in Stuttgart next week, 4 Bosch contacts haven't been touched in 30+ days"
- **Proposal drafting from precedent** — pulls best sections from past proposals via vector recall
- **TTS replies** — hands-free morning brief via Telegram voice note
- **Expense capture** — dinner receipt photo → categorized per deal

### v2 (enterprise SKU for Bosch/Honeywell)
- **Microsoft Teams interface** (Bot Framework + Azure AD SSO)
- **Multi-tenant** — org-level isolation, RBAC, admin console
- **On-prem LLM** — Azure OpenAI in tenant or Claude via Bedrock PrivateLink
- **Compliance** — SOC 2 Type II, ISO 27001, GDPR DPA, IEC 62443 alignment
- **Integrations** — SAP, Salesforce, ServiceNow, SharePoint
- **Data residency** — EU (Bosch) / US (Honeywell)
- **Multi-agent research swarm** — "investigate Honeywell Forge strategy" spawns parallel LinkedIn + news + Gartner workers
- **Win/loss pattern mining** — vector-indexed deal history surfaces "we usually lose to Siemens when decision criterion X dominates"
- **SMS bridge (Twilio)** — loop in clients who don't use Telegram

