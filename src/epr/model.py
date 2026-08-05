"""Provider-agnostic model client: pinned version, retries, cost accounting.

Two backends, one interface. The study is about a prompting intervention, not
about one vendor, so the harness must not be welded to either — and being able
to run a second family is the only real answer to the single-model-family threat
named in the report.

The model id is recorded in every result record. `make reproduce` never calls
this module; tables are rebuilt from `results/raw/` with no network access.
"""

from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, at import, so a key set there works everywhere without the
# caller remembering to do it. Real environment variables always win.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

OPENAI, ANTHROPIC = "openai", "anthropic"

# Dated snapshots only. Undated aliases move under you, and a moved alias makes
# a "reproducible" run silently unreproducible. See HYPOTHESIS.md §8 + Amendment 1.
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"

# USD per million tokens, (input, output). Used ONLY for the projection printed
# before each phase, never for a reported result. Provider pricing changes;
# `make auth-check` prints a reminder to confirm these before a large run.
PRICING = {
    # OpenAI
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-4.1-2025-04-14": (2.00, 8.00),
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "o4-mini-2025-04-16": (1.10, 4.40),
    # Anthropic
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
}

RETRYABLE = (
    "rate_limit",
    "overloaded",
    "api_error",
    "timeout",
    "connection",
    "503",
    "502",
    "500",
    "429",
    "apiconnection",
    "internalserver",
)

# Reasoning models bill hidden thinking tokens and reject `temperature`. Getting
# either wrong silently corrupts both the cost projection and the pinned
# decoding config, so they are detected rather than assumed.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def provider_for(model: str) -> str:
    return ANTHROPIC if model.startswith("claude") else OPENAI


def is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_PREFIXES)


class MissingCredentials(RuntimeError):
    """No usable credential for the selected provider."""


def resolve_credentials(provider: str) -> dict:
    """Find a credential for `provider`, or explain precisely what is missing.

    Supports a plain API key and, for an OAuth-fronted gateway, a bearer token
    plus base URL. Nothing is ever read out of a keychain or a credential store
    belonging to another application.
    """
    if provider == OPENAI:
        key = os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("OPENAI_BASE_URL")
        if key:
            return {"api_key": key} | ({"base_url": base} if base else {})
        raise MissingCredentials(
            "No OpenAI credential found.\n"
            "  Set OPENAI_API_KEY in your environment or in a .env file at the repo root.\n"
            "  For an OAuth-fronted gateway, set OPENAI_API_KEY to the issued bearer\n"
            "  token and OPENAI_BASE_URL to the gateway origin.\n"
            "  Analysis (`make reproduce`) needs no credential at all."
        )

    key = os.environ.get("ANTHROPIC_API_KEY")
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    base = os.environ.get("ANTHROPIC_BASE_URL")
    if key:
        return {"api_key": key} | ({"base_url": base} if base else {})
    if token:
        # Bearer path, for an OAuth-secured gateway in front of the API.
        return {"auth_token": token} | ({"base_url": base} if base else {})
    raise MissingCredentials(
        "No Anthropic credential found.\n"
        "  Set ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN (+ ANTHROPIC_BASE_URL)\n"
        "  for an OAuth-secured gateway. Analysis (`make reproduce`) needs neither."
    )


@dataclass
class Usage:
    """Token and cost accounting. Thread-safe: the runner calls this from a pool,
    and an unsynchronised += would undercount spend — which, for a hard budget
    ceiling, means overspending."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    retries: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, i: int, o: int, r: int = 0) -> None:
        with self._lock:
            self.input_tokens += i
            self.output_tokens += o
            self.reasoning_tokens += r
            self.calls += 1

    def note_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def note_failure(self) -> None:
        with self._lock:
            self.failures += 1

    def cost(self, model: str) -> float:
        cin, cout = PRICING.get(model, (0.0, 0.0))
        # Reasoning tokens bill at the output rate and are already included in
        # output_tokens by both providers' accounting; tracked only for reporting.
        return self.input_tokens / 1e6 * cin + self.output_tokens / 1e6 * cout

    def summary(self, model: str) -> str:
        extra = f" ({self.reasoning_tokens:,} reasoning)" if self.reasoning_tokens else ""
        return (
            f"{self.calls} calls | {self.input_tokens:,} in + {self.output_tokens:,} out{extra} "
            f"| ${self.cost(model):.2f} | {self.retries} retries, {self.failures} failed"
        )


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0
    stop_reason: str = ""
    error: str = ""


@dataclass
class Client:
    """One interface over both providers.

    A call that ultimately fails returns a Response carrying the error rather
    than raising, so one bad item cannot abort a multi-hour run — and the
    failure lands in the raw output instead of vanishing from the denominator.
    """

    model: str = DEFAULT_MODEL
    max_tokens: int = 1400
    temperature: float = 0.0
    max_retries: int = 6
    usage: Usage = field(default_factory=Usage)
    provider: str = ""
    _client: object = None

    def __post_init__(self) -> None:
        self.provider = self.provider or provider_for(self.model)
        if self._client is not None:
            return
        creds = resolve_credentials(self.provider)
        if self.provider == OPENAI:
            from openai import OpenAI

            self._client = OpenAI(**creds)
        else:
            from anthropic import Anthropic

            self._client = Anthropic(**creds)

    # ---------------------------------------------------------------- calls

    def _call_openai(self, system: str, user: str) -> Response:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if is_reasoning_model(self.model):
            # Reasoning models reject temperature and use a different cap that
            # must also cover hidden thinking tokens.
            kwargs["max_completion_tokens"] = self.max_tokens * 3
        else:
            kwargs["temperature"] = self.temperature
            kwargs["max_tokens"] = self.max_tokens

        r = self._client.chat.completions.create(**kwargs)
        choice = r.choices[0]
        u = r.usage
        reasoning = 0
        if u and getattr(u, "completion_tokens_details", None):
            reasoning = getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
        return Response(
            text=choice.message.content or "",
            input_tokens=u.prompt_tokens if u else 0,
            output_tokens=u.completion_tokens if u else 0,
            reasoning_tokens=reasoning,
            stop_reason=str(choice.finish_reason or ""),
        )

    def _call_anthropic(self, system: str, user: str) -> Response:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return Response(
            text=text,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            stop_reason=str(msg.stop_reason or ""),
        )

    def complete(self, system: str, user: str) -> Response:
        """One call, with exponential backoff on transient failures."""
        delay = 2.0
        last = ""
        for attempt in range(self.max_retries):
            try:
                resp = (
                    self._call_openai(system, user)
                    if self.provider == OPENAI
                    else self._call_anthropic(system, user)
                )
                self.usage.add(resp.input_tokens, resp.output_tokens, resp.reasoning_tokens)
                return resp
            except Exception as e:  # noqa: BLE001 - recorded in the record, not swallowed
                last = f"{type(e).__name__}: {e}"
                if not any(k in last.lower() for k in RETRYABLE) or attempt == self.max_retries - 1:
                    break
                self.usage.note_retry()
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        self.usage.note_failure()
        return Response(text="", input_tokens=0, output_tokens=0, error=last)


def estimate_tokens(text: str) -> int:
    """Offline token estimate. ~3.6 chars/token is close enough for a budget check."""
    return int(len(text) / 3.6) + 8


def estimate_cost(
    prompts: list[tuple[str, str]], model: str, expected_output_tokens: int
) -> tuple[int, int, float]:
    """Project (input, output, USD) for a phase before spending anything."""
    tin = sum(estimate_tokens(s) + estimate_tokens(u) for s, u in prompts)
    tout = expected_output_tokens * len(prompts)
    cin, cout = PRICING.get(model, (0.0, 0.0))
    return tin, tout, tin / 1e6 * cin + tout / 1e6 * cout
