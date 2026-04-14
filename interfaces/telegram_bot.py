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
    def __init__(self, agent, settings):
        self.agent = agent
        self.settings = settings
        self.app: Application | None = None
        self.sessions: dict[str, str] = {}
        self.owner_chat_id: int | None = None

    def _session_id(self, user_id: int) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"{user_id}:{today}"
        if key not in self.sessions:
            self.sessions[key] = f"tg-{user_id}-{today}"
        return self.sessions[key]

    def _allowed(self, user_id: int) -> bool:
        ids = self.settings.allowed_user_ids
        return not ids or user_id in ids

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
        if not self._allowed(update.effective_user.id):
            await update.message.reply_text("Not authorized.")
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
        if not self._allowed(update.effective_user.id):
            return
        uid = update.effective_user.id
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        self.sessions[f"{uid}:{today}"] = f"tg-{uid}-{today}-{ts}"
        await update.message.reply_text("Fresh session started.")

    async def _cmd_brief(self, update: Update, context):
        if not self._allowed(update.effective_user.id):
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
        if not self._allowed(uid):
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
        if not self._allowed(uid):
            return
        self.owner_chat_id = update.effective_chat.id
        session_id = self._session_id(uid)

        if not self.settings.openai_api_key:
            await update.message.reply_text(
                "Voice transcription disabled — set OPENAI_API_KEY to enable."
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

        prompt = (
            f"[VOICE NOTE TRANSCRIPT]\n{transcript}\n\n"
            "Extract and file: meeting summary, attendees, decisions, commitments "
            "(as action items with due dates), competitor mentions, any personal "
            "details about contacts. Then confirm what you captured."
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
        """Whisper via OpenAI. Uses a thread executor to avoid blocking."""
        import asyncio

        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        try:
            with open(path, "rb") as f:
                result = await client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
            return result.text or ""
        except Exception as e:
            logger.error(f"Whisper error: {e}", exc_info=True)
            return ""

    async def _on_photo(self, update: Update, context):
        uid = update.effective_user.id
        if not self._allowed(uid):
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
