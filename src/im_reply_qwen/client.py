from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class EndpointConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout_seconds: int = 90


class OpenAICompatibleClient:
    """Small dependency-light client for vLLM and other OpenAI-compatible servers."""

    def __init__(self, config: EndpointConfig) -> None:
        self.config = config

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 160,
    ) -> str:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return str(payload["choices"][0]["message"]["content"]).strip()


def demo_baseline(goal: str, max_chars: int) -> str:
    """Clearly labelled offline fallback; never used as an experimental result."""
    text = f"收到。关于“{goal}”，我会尽快确认后回复您，谢谢。"
    return text[:max_chars]

