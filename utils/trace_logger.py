"""Structured JSONL logging for workflow runs and LLM interactions."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _ROOT / "logs"
_WORKFLOW_LOG = _LOG_DIR / "workflow_runs.jsonl"
_LLM_LOG = _LOG_DIR / "llm_responses.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload))
            f.write("\n")
    except Exception as exc:  # pragma: no cover - depends on filesystem/runtime
        LOGGER.warning("Could not write trace log '%s': %s", path, exc)


def log_workflow_run(payload: dict[str, Any]) -> None:
    entry = {"timestamp": _utc_now(), **payload}
    _append_jsonl(_WORKFLOW_LOG, entry)


def log_llm_response(payload: dict[str, Any]) -> None:
    entry = {"timestamp": _utc_now(), **payload}
    _append_jsonl(_LLM_LOG, entry)


def get_log_paths() -> dict[str, str]:
    return {
        "workflow_runs": str(_WORKFLOW_LOG),
        "llm_responses": str(_LLM_LOG),
    }
