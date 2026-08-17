"""OpenAI-compatible LLM client with dual-mode / adaptive request pacing."""

from __future__ import annotations

import threading
import time
from typing import Any, Literal

from openai import OpenAI

from .config import llm_settings

PaceMode = Literal["auto", "fast", "stable"]


class RequestPacer:
    """Pace LLM calls using fast/stable intervals; auto mode upgrades after 429."""

    def __init__(
        self,
        fast_interval_s: float,
        stable_interval_s: float,
        mode: PaceMode = "auto",
    ):
        self.fast_interval_s = max(0.0, float(fast_interval_s))
        self.stable_interval_s = max(0.0, float(stable_interval_s))
        self.mode: PaceMode = mode
        self._effective = self._interval_for_mode(mode)
        self._lock = threading.Lock()
        self._last_call_at = 0.0
        self._rate_limited_once = False

    def _interval_for_mode(self, mode: PaceMode) -> float:
        if mode == "stable":
            return self.stable_interval_s
        if mode == "fast":
            return self.fast_interval_s
        # auto: start fast, may promote to stable after 429
        return self.fast_interval_s

    @property
    def min_interval_s(self) -> float:
        return self._effective

    def set_mode(self, mode: PaceMode) -> None:
        with self._lock:
            self.mode = mode
            if mode == "auto" and self._rate_limited_once:
                self._effective = self.stable_interval_s
            else:
                self._effective = self._interval_for_mode(mode)

    def on_rate_limit(self) -> None:
        with self._lock:
            self._rate_limited_once = True
            if self.mode in ("auto", "stable"):
                self._effective = max(self._effective, self.stable_interval_s)
            elif self.mode == "fast":
                # still briefly slow down even in fast mode after 429
                self._effective = max(self._effective, self.stable_interval_s)

    def wait(self) -> None:
        with self._lock:
            interval = self._effective
            now = time.monotonic()
            gap = interval - (now - self._last_call_at)
            if gap > 0:
                time.sleep(gap)
            self._last_call_at = time.monotonic()


class ChatLLM:
    def __init__(self, pace_mode: PaceMode | None = None):
        settings = llm_settings()
        self.client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])
        self.model = settings["model"]
        self.provider = settings["provider"]
        mode = pace_mode or settings["pace_mode"]
        self.pacer = RequestPacer(
            fast_interval_s=settings["fast_interval_s"],
            stable_interval_s=settings["stable_interval_s"],
            mode=mode,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_retries: int = 5,
    ) -> Any:
        last_err: Exception | None = None
        for attempt in range(max_retries):
            self.pacer.wait()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc).lower()
                if "rate" in msg or "429" in msg:
                    self.pacer.on_rate_limit()
                    time.sleep(max(20.0, self.pacer.stable_interval_s * (attempt + 2)))
                else:
                    time.sleep(min(2 ** attempt, 8) + self.pacer.min_interval_s * 0.5)
        raise RuntimeError(f"LLM request failed after retries: {last_err}")
