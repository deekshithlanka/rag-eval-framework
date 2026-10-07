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
        kwargs = {"model": model, "max_retries": 6}
        if temperature is not None:
            kwargs["temperature"] = temperature
        self.model = model
        self.temperature = temperature
        self.client = ChatGoogleGenerativeAI(**kwargs)
        self.limiter = RateLimiter(rpm)

    def _key(self, system: str, user: str) -> Path:
        h = hashlib.sha256(json.dumps([self.model, self.temperature, system, user]).encode()).hexdigest()
        return CACHE / f"{h}.json"

    def complete(self, system: str, user: str) -> LLMResult:
        path = self._key(system, user)
        if path.exists():
            data = json.loads(path.read_text())
            return LLMResult(**{**data, "cached": True})

        from langchain_core.messages import HumanMessage, SystemMessage

        self.limiter.wait()
        start = time.perf_counter()
        msg = self.client.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        latency = time.perf_counter() - start
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


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model response (handles code fences)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:200]}")
    return json.loads(match.group(0))
