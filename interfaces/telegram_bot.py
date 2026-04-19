"""Telegram interface — text, voice notes (via Whisper), and photo handlers.

Long-polling for zero-ops deployment. Voice notes are downloaded, transcribed
with OpenAI Whisper, prefixed with [VOICE NOTE]:, and sent to the agent. Photos
are forwarded as-is with a [PHOTO] marker so the agent can use vision on them
(in a later iteration — v1 stores a placeholder).
"""
import logging
import os
import tempfile
from datetime import datetime, timezone

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)


def _smart_split(text: str, limit: int = 4096) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


class TelegramBot:
    def __init__(self, agent, settings, session_maker=None):
        self.agent = agent
        self.settings = settings
        # Optional — when provided, _allowed() also checks the DB allow-list
        # so owner can add collaborators via the auth-add_user skill without
        # editing env vars. Env list remains the root trust (owners).
        self.session_maker = session_maker
        self.app: Application | None = None
        self.sessions: dict[str, str] = {}
        self.owner_chat_id: int | None = None

    def _session_id(self, user_id: int) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"{user_id}:{today}"
        if key not in self.sessions:
            self.sessions[key] = f"tg-{user_id}-{today}"
        return self.sessions[key]

    async def _allowed(self, user_id: int) -> bool:
        env_ids = self.settings.allowed_user_ids
        if env_ids and user_id in env_ids:
            return True
        if self.session_maker is None:
            # No DB wired — open bot (env_ids empty) or locked to env only
            return not env_ids
        # DB allow-list (runtime-managed via AuthSkill)
        try:
            from sqlalchemy import select
            from ..db.models import AuthorizedUser
            async with self.session_maker() as s:
                row = await s.get(AuthorizedUser, str(user_id))
                return bool(row and row.active == "yes")
        except Exception as e:
            logger.warning("auth DB check failed, falling back to env-only: %s", e)
            return not env_ids  # fail-soft: open if no env, closed if env list present

    async def _deny(self, update: Update) -> None:
        """Tell the user their Telegram ID so they can share it with the owner."""
        uid = update.effective_user.id
        name = (update.effective_user.full_name or "").strip()
        msg = (
            f"Not authorized. Ask the owner to add you with this ID:\n"
            f"\n"
            f"  {uid}  ({name})\n"
        )
        try:
            await update.message.reply_text(msg)
        except Exception:
            pass

    async def start(self):
        self.app = (
            Application.builder().token(self.settings.telegram_bot_token).build()
        )
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("new", self._cmd_new))
        self.app.add_handler(CommandHandler("brief", self._cmd_brief))
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )
        self.app.add_handler(MessageHandler(filters.VOICE, self._on_voice))
        self.app.add_handler(
            MessageHandler(filters.AUDIO, self._on_voice)
        )
        self.app.add_handler(MessageHandler(filters.PHOTO, self._on_photo))

        await self.app.initialize()
        try:
            await self.app.bot.delete_webhook(drop_pending_updates=True)
        except Exception as e:
            logger.warning(f"delete_webhook failed: {e}")
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        logger.info("Telegram bot started")

    async def stop(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

    async def send_to_owner(self, text: str):
        if not self.owner_chat_id and self.settings.allowed_user_ids:
            self.owner_chat_id = self.settings.allowed_user_ids[0]
        if self.owner_chat_id and self.app:
            await self._send(self.owner_chat_id, text)

    async def _send(self, chat_id: int, text: str):
        for chunk in _smart_split(text, 4096):
            await self.app.bot.send_message(chat_id=chat_id, text=chunk)

    # -- handlers --

    async def _cmd_start(self, update: Update, context):
        if not await self._allowed(update.effective_user.id):
            await self._deny(update)
            return
        self.owner_chat_id = update.effective_chat.id
        await update.message.reply_text(
            f"{self.settings.agent_name} online.\n\n"
            "Send me text, voice notes, or photos. Try:\n"
            "• 'Just met with Markus at Bosch about the condition-monitoring pilot'\n"
            "• 'What's going on with Honeywell?'\n"
            "• /brief — today's brief\n"
            "• /new — start a fresh session"
        )

    async def _cmd_new(self, update: Update, context):
        if not await self._allowed(update.effective_user.id):
            await self._deny(update)
            return
        uid = update.effective_user.id
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        self.sessions[f"{uid}:{today}"] = f"tg-{uid}-{today}-{ts}"
        await update.message.reply_text("Fresh session started.")

    async def _cmd_brief(self, update: Update, context):
        if not await self._allowed(update.effective_user.id):
            await self._deny(update)
            return
        await update.message.chat.send_action(ChatAction.TYPING)
        session_id = self._session_id(update.effective_user.id)
        try:
            response = await self.agent.run(
                "Give me today's brief.", session_id=session_id, interface="telegram"
            )
            await self._send(update.effective_chat.id, response)
        except Exception as e:
            logger.error(f"Brief error: {e}", exc_info=True)
            await update.message.reply_text(f"Error: {str(e)[:200]}")

    async def _on_text(self, update: Update, context):
        uid = update.effective_user.id
        if not await self._allowed(uid):
            await self._deny(update)
            return
        self.owner_chat_id = update.effective_chat.id
        session_id = self._session_id(uid)
        await update.message.chat.send_action(ChatAction.TYPING)
        try:
            response = await self.agent.run(
                update.message.text, session_id=session_id, interface="telegram"
            )
            await self._send(update.effective_chat.id, response)
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            await update.message.reply_text(f"Error: {str(e)[:200]}")

    async def _on_voice(self, update: Update, context):
        uid = update.effective_user.id
        if not await self._allowed(uid):
            await self._deny(update)
            return
        self.owner_chat_id = update.effective_chat.id
        session_id = self._session_id(uid)

        has_stt = self.settings.deepgram_api_key or self.settings.openai_api_key
        if not has_stt:
            await update.message.reply_text(
                "Voice transcription disabled — set DEEPGRAM_API_KEY (preferred) or OPENAI_API_KEY."
            )
            return

        await update.message.chat.send_action(ChatAction.TYPING)

        voice = update.message.voice or update.message.audio
        file = await context.bot.get_file(voice.file_id)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            await file.download_to_drive(tmp_path)
            transcript = await self._transcribe(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if not transcript:
            await update.message.reply_text("Couldn't transcribe that.")
            return

        # For longer recordings (meeting-length, ≥600 chars ≈ 60+s), pre-run
        # the structured categorizer so the agent receives pre-extracted
        # meeting_type / decisions / action items / MEDDIC deltas / competitor
        # mentions instead of having to extract them itself. Big quality jump
        # on long transcripts; matches the "Jamie" pattern users expect.
        is_long = len(transcript) >= 600
        structured_block = ""
        if is_long:
            try:
                from ..core.audio_processor import categorize_transcript
                cat = await categorize_transcript(
                    transcript=transcript,
                    deal_context="(Telegram voice note — caller must identify deal from transcript content)",
                    llm_client=self.agent.client,
                    model=self.settings.fast_model,
                )
                if cat:
                    structured_block = (
                        "\n\n[PRE-EXTRACTED CATEGORIZATION]\n"
                        f"meeting_type: {cat.get('meeting_type', 'other')}\n"
                        f"sentiment: {cat.get('sentiment', 'unknown')}\n"
                        f"summary: {cat.get('summary', '')}\n"
                        f"attendees_mentioned: {', '.join(cat.get('attendees_mentioned', []))}\n"
                        f"key_decisions: {'; '.join(cat.get('key_decisions', []))}\n"
                        f"action_items: {'; '.join(a.get('description', '') for a in cat.get('action_items', []))}\n"
                        f"competitors_mentioned: {', '.join(cat.get('competitors_mentioned', []))}\n"
                        f"pricing_mentioned: {cat.get('pricing_mentioned', '')}\n"
                        f"meddic_deltas: {cat.get('meddic_deltas', {})}\n"
                    )
            except Exception as e:
                logger.warning("Pre-categorization failed (continuing with raw transcript): %s", e)

        prompt = (
            f"[VOICE NOTE TRANSCRIPT]\n{transcript}\n"
            f"{structured_block}\n"
            "File this. Identify the deal from the transcript content (call DealSkill.find), "
            "then log a Meeting with the summary/attendees/decisions, create action items "
            "for each commitment with due dates, append any MEDDIC field updates, and "
            "update personal_notes on contacts mentioned. Confirm what you captured "
            "in 1-2 sentences."
        )
        try:
            response = await self.agent.run(
                prompt, session_id=session_id, interface="telegram"
            )
            await self._send(update.effective_chat.id, response)
        except Exception as e:
            logger.error(f"Voice agent error: {e}", exc_info=True)
            await update.message.reply_text(f"Error: {str(e)[:200]}")

    async def _transcribe(self, path: str) -> str:
        """Transcribe via Deepgram (preferred) or Whisper (fallback).

        Deepgram has speaker diarization (knows who said who), no file-size
        cap, and is ~30% cheaper. Whisper is the legacy path when only
        OPENAI_API_KEY is set.
        """
        try:
            with open(path, "rb") as f:
                audio_bytes = f.read()
            from ..core.audio_processor import transcribe_audio
            return await transcribe_audio(
                audio_bytes,
                filename=os.path.basename(path),
                openai_api_key=self.settings.openai_api_key,
                deepgram_api_key=self.settings.deepgram_api_key,
                deepgram_model=getattr(self.settings, "deepgram_model", "nova-3"),
            )
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""

    async def _on_photo(self, update: Update, context):
        uid = update.effective_user.id
        if not await self._allowed(uid):
            await self._deny(update)
            return
        # v1: acknowledge only. Vision handling (business cards, whiteboards)
        # added in v1.1 — forward caption + placeholder to the agent.
        caption = update.message.caption or ""
        session_id = self._session_id(uid)
        prompt = f"[PHOTO received with caption: '{caption}'] (vision not yet wired — note the event)"
        try:
            response = await self.agent.run(
                prompt, session_id=session_id, interface="telegram"
            )
            await self._send(update.effective_chat.id, response)
        except Exception as e:
            logger.error(f"Photo handler error: {e}", exc_info=True)
