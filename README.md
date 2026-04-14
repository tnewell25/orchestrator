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

## Roadmap

- v1 (shipping now): Telegram, CRM skills, voice capture, daily brief
- v1.1: Vision (photos of business cards, whiteboards), LinkedIn research tool, proposal drafting
- v1.5: Read-only web dashboard (Next.js) for pipeline view
- v2: Microsoft Teams interface, multi-tenant auth, SSO, admin console (enterprise SKU)
