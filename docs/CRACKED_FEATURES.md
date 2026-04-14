# Cracked Features — Master Vision

Comprehensive feature inventory for the sales-engineer persona. Organized by category, marked with status:

- ✅ **Shipped** — in the current codebase
- 🚧 **Building** — next 1-2 sessions
- 📋 **Roadmap** — known-wanted, not yet started
- 🤔 **Idea** — worth validating before building

---

## 1. Inbox Intelligence

| # | Feature | Status |
|---|---------|--------|
| 1 | List unread emails | ✅ |
| 2 | Create drafts / send | ✅ |
| 3 | Rank unread by importance (sender × urgency × deal-relevance) | ✅ |
| 4 | Track sent emails, nudge if no reply in N days | ✅ |
| 5 | Suggest 3 reply drafts in different tones (direct/diplomatic/close-question) | 🚧 |
| 6 | Thread summarization (long chains → 5 bullets) | 🚧 |
| 7 | Auto-extract commitments from email → action items | 🚧 |
| 8 | BCC-to-log — bcc bot@ on any email → auto-files to deal | 📋 |
| 9 | Attachment processing — RFP PDF → extract requirements | 📋 |
| 10 | Voice-to-email dictation with draft review | 📋 |
| 11 | Multi-inbox (personal + Bosch Outlook + Honeywell Outlook) | 📋 |

## 2. Calendar Intelligence

| # | Feature | Status |
|---|---------|--------|
| 12 | List today / upcoming / create events | ✅ |
| 13 | Auto-pre-meeting briefs for every event with known contacts (hourly sync) | ✅ |
| 14 | Pre-meeting brief generator (last meetings + actions + personal notes) | ✅ |
| 15 | Daily combined prep pack ("tomorrow: 3 Bosch meetings") | 🚧 |
| 16 | Smart scheduling (propose 3 slots matching constraints) | 📋 |
| 17 | Travel-aware scheduling (timezone + jet lag) | 📋 |
| 18 | Conflict detection + resolution suggestions | 📋 |
| 19 | Post-meeting prompt — meeting ends → "voice-note me" | 📋 |
| 20 | Conference/event tracker (Hannover Messe, ISA, IoT World) | 🤔 |
| 21 | Recurring pattern suggestions ("4 meetings with Anja — standing cadence?") | 🤔 |

## 3. Pipeline Intelligence

| # | Feature | Status |
|---|---------|--------|
| 22 | MEDDIC fields + gap analysis | ✅ |
| 23 | Stakeholder map — 5 roles (champion/EC/tech/blocker/coach/user) | ✅ |
| 24 | Deal temperature scoring (recency × MEDDIC × coverage × activity × stage) | ✅ |
| 25 | Stalled deal alerts (proactive, no activity 14d+) | ✅ |
| 26 | Pipeline health overview (hot/warm/cold/stalled buckets) | ✅ |
| 27 | Deal velocity comparison vs your historical average | 📋 |
| 28 | Win/loss pattern mining ("you win when EC is CTO-level") | 🚧 (log_win_loss built; mining view coming) |
| 29 | Multithread coverage alerts | ✅ (coverage_score in StakeholderSkill) |
| 30 | Champion health score (gone dark?) | 📋 |

## 4. Contact Intelligence

| # | Feature | Status |
|---|---------|--------|
| 31 | CRUD contacts with personal relationship notes | ✅ |
| 32 | LinkedIn job change alerts | 📋 (ResearchSkill can look up on demand; proactive later) |
| 33 | Company news monitoring (layoffs, acquisitions, leadership) | 📋 |
| 34 | Auto-enrichment (name + company → LinkedIn profile) | 🚧 (via research-exec_bio) |
| 35 | Intro path finder ("who I know at Honeywell Aerospace") | 📋 |
| 36 | Personal ammo refresh before meetings (surfaces in pre_meeting brief) | ✅ |
| 37 | Dormant relationship revival (3 people 90d+ dormant) | 📋 |
| 38 | Birthday/anniversary reminders | 🤔 |
| 39 | Relationship strength indicator | 📋 |

## 5. Proposal / Document Generation

| # | Feature | Status |
|---|---------|--------|
| 40 | Save precedent sections (vector-indexed) | ✅ |
| 41 | Find precedent by semantic search | ✅ |
| 42 | Draft full proposal markdown from deal context + precedent | ✅ |
| 43 | RFP response generator (upload PDF → section-by-section draft) | 📋 |
| 44 | SOW generator | 📋 |
| 45 | Pricing calculator with margin/volume tiers | 📋 |
| 46 | One-pager exec summary generator | 📋 |
| 47 | Redline tracker for customer-returned docs | 📋 |

## 6. Research / Due Diligence

| # | Feature | Status |
|---|---------|--------|
| 48 | Web search (Serper or DuckDuckGo fallback) | ✅ |
| 49 | Company deep dive (news + leadership + strategy) | ✅ |
| 50 | Exec bio lookup (LinkedIn + background + speaking) | ✅ |
| 51 | Competitive analysis (features + criticism + complaints) | ✅ |
| 52 | Vertical intelligence (industry-wide concerns) | 🚧 (reachable via research-search) |
| 53 | Regulatory tracking (IEC 62443, NIST, CMMC) | 📋 |
| 54 | Patent landscape | 🤔 |
| 55 | Tech stack inference from job postings | 📋 |
| 56 | M&A / funding / hiring signals as buying-mode indicators | 📋 |

## 7. Competitor Intelligence

| # | Feature | Status |
|---|---------|--------|
| 57 | Competitor profiles (strengths/weaknesses/pricing) | ✅ |
| 58 | Battle cards (situation-specific, vector-indexed) | ✅ |
| 59 | Semantic battle card retrieval | ✅ |
| 60 | Win/loss recording per deal | ✅ |
| 61 | Win/loss summary (by competitor) | ✅ |
| 62 | Pricing benchmarks from past deals | 📋 |
| 63 | Feature gap tracking ("MindSphere does X, we don't") | 📋 |

## 8. Bid / RFP

| # | Feature | Status |
|---|---------|--------|
| 64 | Bid CRUD with submission + Q&A deadlines | ✅ |
| 65 | Auto-scheduled T-7d/T-3d/T-1d reminders | ✅ |
| 66 | Mark submitted (cancels remaining reminders) | ✅ |
| 67 | RFP Q&A tracker | 📋 |

## 9. Meeting Capture

| # | Feature | Status |
|---|---------|--------|
| 68 | Log meeting (summary, attendees, decisions, transcript) | ✅ |
| 69 | Voice-note capture (Telegram → Whisper → agent) | ✅ |
| 70 | Auto-extract action items from meeting | ✅ (via system-prompt instruction) |
| 71 | Auto-extract MEDDIC signals from meeting | ✅ (via system-prompt instruction) |
| 72 | Auto-save personal notes about contacts from meeting | ✅ (via system-prompt instruction) |
| 73 | Meeting recording transcription (Zoom/Teams integration) | 📋 |

## 10. Reminders / Tasks

| # | Feature | Status |
|---|---------|--------|
| 74 | Natural-language time parsing ("in 2 hours", "tomorrow 3pm") | ✅ |
| 75 | Pre-meeting brief reminders (agent-enriched) | ✅ |
| 76 | Bid deadline reminders (auto) | ✅ |
| 77 | Commitment reminders (with due-date context) | ✅ |
| 78 | Snooze / cancel / list pending | ✅ |
| 79 | Persistent across restart (DB-backed) | ✅ |
| 80 | Smart recurring ("every Mon 9am") | 📋 |

## 11. Productivity / Personal Ops

| # | Feature | Status |
|---|---------|--------|
| 81 | Daily brief (overdue + today meetings + closing deals) | ✅ |
| 82 | Morning brief (TTS-friendly narrative form) | ✅ |
| 83 | Weekly review auto-draft (wins, losses, movement, next week) | ✅ |
| 84 | End-of-day journal prompt → files to deals | 📋 |
| 85 | Commission tracker | 📋 |
| 86 | Goal / quota tracking against pipeline | 📋 |
| 87 | Focus mode (block 90min, no notifications) | 🤔 |
| 88 | Energy awareness ("7 calls today, async tomorrow") | 🤔 |

## 12. Proactive Behavior (Background Services)

| # | Feature | Status |
|---|---------|--------|
| 89 | Reminder polling service (30s) | ✅ |
| 90 | Calendar auto-sync (hourly, creates pre-meeting reminders) | ✅ |
| 91 | Proactive monitor (4h sweep: stalled deals, unanswered emails, overdue actions) | ✅ |
| 92 | Automatic weekly review on Friday | 📋 (skill exists; schedule wiring next) |
| 93 | Morning brief auto-push at configurable time | 📋 |

## 13. Travel Intelligence

| # | Feature | Status |
|---|---------|--------|
| 94 | Geo-aware contact surfacing ("in Stuttgart — 4 dormant contacts nearby") | 📋 |
| 95 | Flight tracker + leave-by reminders | 📋 |
| 96 | Expense receipt OCR | 📋 |
| 97 | Timezone math in scheduling | 📋 |
| 98 | Auto-expense filing per deal | 📋 |

## 14. Advanced AI Features

| # | Feature | Status |
|---|---------|--------|
| 99 | Multi-agent research swarm (parallel workers) | 📋 |
| 100 | Meeting role-play practice | 🤔 |
| 101 | Objection simulator (10 likely objections) | 🤔 |
| 102 | Cold email generator trained on your past winners | 📋 |
| 103 | Sentiment analysis on client comms | 📋 |
| 104 | Linguistic style matching (drafts in your voice) | 📋 |
| 105 | Technical-to-non-technical translation | 🤔 |

## 15. Voice / Mobile UX

| # | Feature | Status |
|---|---------|--------|
| 106 | Voice note capture (Whisper) | ✅ |
| 107 | Photo capture placeholder | ✅ (handler wired; full vision pending) |
| 108 | Business card OCR → contact | 📋 |
| 109 | Whiteboard photo → meeting note | 📋 |
| 110 | Receipt OCR → expense | 📋 |
| 111 | Hands-free morning brief via TTS | 📋 (morning_brief data shipped; TTS wiring pending) |
| 112 | Voice search ("what did I promise Markus") | ✅ (via memory-recall tool, Telegram voice input) |
| 113 | Apple Watch deadline notifications | 🤔 |

## 16. Integrations

| # | Feature | Status |
|---|---------|--------|
| 114 | Gmail | ✅ |
| 115 | Google Calendar | ✅ |
| 116 | Claude Max OAuth (no API billing) | ✅ |
| 117 | CRM sync (Salesforce / HubSpot two-way) | 📋 |
| 118 | ERP read (SAP / Oracle at client) | 🤔 (needs per-client IT) |
| 119 | Google Drive / OneDrive auto-tagging | 📋 |
| 120 | DocuSign send + track | 📋 |
| 121 | Zoom / Teams — fetch recordings → transcribe | 📋 |
| 122 | Slack / Teams bridge | 📋 |

## 17. Strategic Synthesis

| # | Feature | Status |
|---|---------|--------|
| 123 | Relationship graph viz | 📋 |
| 124 | Ex-founder network tool (BSH/HON alumni leverage) | 🤔 |
| 125 | Proactive intros | 🤔 |
| 126 | Deal chess (pivot resources between deals) | 🤔 |
| 127 | Market map (top 50 industrial IoT players, auto-updated) | 🤔 |
| 128 | Pattern memos ("German auto buys due to CBAM, US aero due to FAA") | 🤔 |

---

## Prioritization Rationale

**What got built first:** anything that captures the user's reality with zero tax (voice capture, ubiquitous reminders, automatic pre-meeting briefs, proactive stalled-deal nudges). The bot is useless if he has to think about it — it has to earn its place by being effortless.

**What's next:** reply drafting, thread summarization, weekly-review auto-schedule, commitment extraction from email. This turns the inbox from an obligation into a managed queue.

**What's after that:** multi-agent research (feels like magic when he says "research Honeywell Forge strategy" and gets a briefing 90 seconds later), auto-push of morning brief at configurable time with TTS.

**Enterprise differentiators (v2):** Teams bot, SSO, multi-tenant, on-prem LLM, SOC 2 — all required to sell into Bosch/Honeywell procurement, but useless for a single user test.
