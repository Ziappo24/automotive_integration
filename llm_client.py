"""
llm_client.py
=============
Thin wrapper around the local Ollama REST API.

Features:
  - SQLite-backed cache: identical (model, prompt) pairs are never sent twice.
  - JSONL log of every LLM call with timestamp, latency, and outcome.
  - Strict JSON validation: returns None on malformed output (triggers fallback).
  - Configurable retry logic before giving up.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from config import (
    CACHE_DB_FILE,
    LLM_LOG_FILE,
    LOGS_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_MAX_RETRIES,
    OLLAMA_MAX_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
)

logger = logging.getLogger("integration.llm_client")


# ===========================================================================
# SQLite Cache
# ===========================================================================

class LLMCache:
    """Persistent key-value cache backed by SQLite."""

    def __init__(self, db_path: Path = CACHE_DB_FILE) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                ts    TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        """Return cached JSON string or None."""
        row = self._conn.execute(
            "SELECT value FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        """Insert or replace a cache entry."""
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, ts) VALUES (?, ?, ?)",
            (key, value, ts),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


_cache = LLMCache()


# ===========================================================================
# JSONL Logger
# ===========================================================================

def _log_llm_call(entry: Dict[str, Any]) -> None:
    """Append a structured log entry to the JSONL log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LLM_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ===========================================================================
# Core LLM Call
# ===========================================================================

def call_llm(
    system_prompt: str,
    user_message: str,
    required_keys: Optional[List[str]] = None,
    call_tag: str = "generic",
) -> Optional[Dict[str, Any]]:
    """
    Send a prompt to the local Ollama model and return parsed JSON.

    The function:
      1. Checks the SQLite cache first.
      2. Sends the request to Ollama with retries.
      3. Attempts to parse the response as JSON.
      4. Validates required_keys if provided.
      5. Returns None (triggering caller's fallback) on any failure.
      6. Logs every attempt to the JSONL log.

    Parameters
    ----------
    system_prompt : str
        System-level context / instructions for the LLM.
    user_message : str
        The user-facing part of the prompt (the actual task).
    required_keys : list[str], optional
        If provided, the parsed JSON must contain all listed keys.
    call_tag : str
        Label for the log entry (e.g. 'schema_alignment', 'record_linkage').

    Returns
    -------
    dict or None
        Parsed JSON dict, or None on any failure.
    """
    cache_key = f"{OLLAMA_MODEL}|{system_prompt}|{user_message}"
    cached = _cache.get(cache_key)

    if cached is not None:
        logger.debug("[%s] Cache HIT", call_tag)
        try:
            result = json.loads(cached)
            _log_llm_call({
                "ts": datetime.now(timezone.utc).isoformat(),
                "tag": call_tag,
                "cached": True,
                "success": True,
                "latency_s": 0.0,
                "result": result,
            })
            return result
        except json.JSONDecodeError:
            logger.warning("[%s] Cached value is not valid JSON; calling model", call_tag)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_MAX_TOKENS,
        },
        "stream": False,
        "format": "json",   # Ollama JSON mode
    }

    last_error: Optional[str] = None
    for attempt in range(1, OLLAMA_MAX_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            response = requests.post(
                OLLAMA_BASE_URL,
                json=payload,
                timeout=OLLAMA_TIMEOUT_SECONDS,
            )
            latency = round(time.perf_counter() - t0, 3)
            response.raise_for_status()
            raw_text = response.json()["message"]["content"]

            # Parse JSON
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                last_error = f"JSONDecodeError: {exc}"
                logger.warning("[%s] attempt %d: invalid JSON — %s", call_tag, attempt, exc)
                continue

            # Validate required keys
            if required_keys:
                missing = [k for k in required_keys if k not in parsed]
                if missing:
                    last_error = f"Missing keys: {missing}"
                    logger.warning(
                        "[%s] attempt %d: missing keys %s in response",
                        call_tag, attempt, missing,
                    )
                    continue

            # Success — cache and log
            _cache.set(cache_key, json.dumps(parsed))
            _log_llm_call({
                "ts": datetime.now(timezone.utc).isoformat(),
                "tag": call_tag,
                "cached": False,
                "attempt": attempt,
                "success": True,
                "latency_s": latency,
                "result": parsed,
            })
            return parsed

        except requests.exceptions.Timeout:
            last_error = "Timeout"
            logger.warning("[%s] attempt %d: request timed out", call_tag, attempt)
        except requests.exceptions.ConnectionError:
            last_error = "ConnectionError (is Ollama running?)"
            logger.error("[%s] Cannot connect to Ollama at %s", call_tag, OLLAMA_BASE_URL)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            logger.warning("[%s] attempt %d: unexpected error — %s", call_tag, attempt, exc)

    # All retries exhausted → return None (triggers fallback)
    _log_llm_call({
        "ts": datetime.now(timezone.utc).isoformat(),
        "tag": call_tag,
        "cached": False,
        "success": False,
        "error": last_error,
        "fallback_triggered": True,
    })
    logger.error(
        "[%s] All %d retries failed. Last error: %s. Triggering fallback.",
        call_tag, OLLAMA_MAX_RETRIES, last_error,
    )
    return None
