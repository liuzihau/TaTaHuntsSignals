"""Runtime config loader with env-variable support."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    app_mode: str = os.getenv("APP_MODE", "demo")
    default_ticker: str = os.getenv("DEFAULT_TICKER", "NVDA")


def get_settings() -> Settings:
    return Settings()
