"""Chat model wrapper with a disk cache and a simple rate limiter.

The cache makes reruns free and reproducible: the same model, prompt and
settings always return the stored response.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CACHE = Path(".cache/llm")


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / max(rpm, 1)
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                time.sleep(self.next_time - now)
            self.next_time = max(now, self.next_time) + self.interval


class GeminiChat:
    def __init__(self, model: str, temperature: float | None = None, rpm: int = 30):
        from dotenv import load_dotenv
        from langchain_google_genai import ChatGoogleGenerativeAI

        load_dotenv()
        kwargs = {"model": model, "max_retries": 0}  # retries handled in _invoke_with_retry
        if temperature is not None:
            kwargs["temperature"] = temperature
        self.model = model
        self.temperature = temperature
        self.client = ChatGoogleGenerativeAI(**kwargs)
        self.limiter = RateLimiter(rpm)

    def _key(self, system: str, user: str) -> Path:
        h = hashlib.sha256(json.dumps([self.model, self.temperature, system, user]).encode()).hexdigest()
        return CACHE / f"{h}.json"

    def _invoke_with_retry(self, messages, attempts: int = 10):
        for i in range(attempts):
            self.limiter.wait()
            start = time.perf_counter()
            try:
                msg = self.client.invoke(messages)
                return msg, time.perf_counter() - start
            except Exception as e:  # noqa: BLE001
                kind = _rate_limit_kind(e)
                if kind == "day":
                    raise SystemExit(
                        f"Daily free-tier quota used up for {self.model}. Rerun tomorrow (saved work is reused) "
                        "or add billing at aistudio.google.com/apikey."
                    ) from e
                if kind != "minute" or i == attempts - 1:
                    raise
                wait = _retry_delay(e)
                print(f"  {self.model} rate limited, waiting {wait:.0f}s (attempt {i + 1}/{attempts})")
                time.sleep(wait)

    def complete(self, system: str, user: str) -> LLMResult:
        path = self._key(system, user)
        if path.exists():
            data = json.loads(path.read_text())
            return LLMResult(**{**data, "cached": True})

        from langchain_core.messages import HumanMessage, SystemMessage

        msg, latency = self._invoke_with_retry([SystemMessage(content=system), HumanMessage(content=user)])
        usage = getattr(msg, "usage_metadata", None) or {}
        text = msg.content if isinstance(msg.content, str) else "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in msg.content
        )
        result = LLMResult(
            text=text.strip(),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_s=round(latency, 3),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(result)))
        return result


def _rate_limit_kind(err: Exception) -> str | None:
    """'minute' for per-minute limits (worth waiting), 'day' for daily quotas (stop), else None."""
    msg = str(err)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "day" if "PerDay" in msg else "minute"
    if any(code in msg for code in ("503", "UNAVAILABLE", "500 INTERNAL", "DEADLINE_EXCEEDED")):
        return "minute"  # transient server errors: wait and retry
    return None


def _retry_delay(err: Exception, default: float = 30.0) -> float:
    match = re.search(r"retry in ([0-9.]+)s", str(err))
    return float(match.group(1)) + 2 if match else default


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model response (handles code fences)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    return json.loads(match.group(0))
