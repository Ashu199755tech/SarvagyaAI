"""Bot Framework adapter configuration for MS Teams."""

from __future__ import annotations

from botbuilder.core import (
    BotFrameworkAdapter,
    BotFrameworkAdapterSettings,
)

from src.config import settings


def create_adapter() -> BotFrameworkAdapter:
    """Create a BotFrameworkAdapter with Teams app credentials."""
    adapter_settings = BotFrameworkAdapterSettings(
        app_id=settings.teams_app_id,
        app_password=settings.teams_app_password,
    )
    adapter = BotFrameworkAdapter(adapter_settings)

    # Global error handler
    async def on_error(context, error):
        import logging

        logging.getLogger(__name__).exception("Bot error: %s", error)
        await context.send_activity(
            "Sorry, something went wrong on my end. Please try again later."
        )

    adapter.on_turn_error = on_error
    return adapter
