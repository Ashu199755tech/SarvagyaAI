"""MS Teams bot activity handler — routes user messages to the RAG chain."""

from __future__ import annotations

import logging

from botbuilder.core import TurnContext
from botbuilder.core.teams import TeamsActivityHandler
from botbuilder.schema import ChannelAccount

from src.rag.chain import ask

logger = logging.getLogger(__name__)

WELCOME_MESSAGE = (
    "👋 Hi! I'm **ResoAI**, your HR assistant.\n\n"
    "Ask me about employees, projects, clients, team allocations, "
    "or project timelines and I'll look it up for you.\n\n"
    "For example:\n"
    "- *Who is working on Project X?*\n"
    "- *What projects is John assigned to?*\n"
    "- *When does the Acme project end?*"
)


class ResoAIBot(TeamsActivityHandler):
    """Teams bot that answers HR questions via RAG."""

    # ── Messages ─────────────────────────────────────────

    async def on_message_activity(self, turn_context: TurnContext) -> None:
        """Handle incoming user messages."""
        question = turn_context.activity.text
        if not question or not question.strip():
            await turn_context.send_activity("Please type a question and I'll help you out!")
            return

        logger.info("User question: %s", question)

        # Send typing indicator while we work
        await turn_context.send_activity("🔍 Looking that up for you …")

        try:
            result = await ask(question.strip())
            answer = result["answer"]
        except Exception:
            logger.exception("RAG chain error")
            answer = "Sorry, I ran into an issue fetching that information. Please try again."

        await turn_context.send_activity(answer)

    # ── Member Events ────────────────────────────────────

    async def on_members_added_activity(
        self,
        members_added: list[ChannelAccount],
        turn_context: TurnContext,
    ) -> None:
        """Send a welcome message when the bot is added to a conversation."""
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(WELCOME_MESSAGE)
