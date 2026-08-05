"""Model client: pinned version, retries, and cost accounting.

The model id is recorded in every result record. `make reproduce` never calls
this module — tables are rebuilt from `results/raw/` with no network access.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

# Dated snapshot, not a moving alias. See HYPOTHESIS.md §8.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# USD per million tokens. Used ONLY for the pre-run estimate printed before each
# phase, never for any reported result. Verify against current pricing before
# trusting a projection.
PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
}

RETRYABLE = ("rate_limit", "overloaded", "api_error", "timeout", "connection")


class MissingAPIKey(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    retries: int = 0

    def add(self, i: int, o: int) -> None:
        self.input_tokens += i
        self.output_tokens += o
        self.calls += 1

    def cost(self, model: str) -> float:
        cin, cout = PRICING.get(model, (0.0, 0.0))
        return self.input_tokens / 1e6 * cin + self.output_tokens / 1e6 * cout

    def summary(self, model: str) -> str:
        return (
            f"{self.calls} calls | {self.input_tokens:,} in + {self.output_tokens:,} out "
            f"| ${self.cost(model):.2f} | {self.retries} retries"
        )


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str = ""
    error: str = ""


@dataclass
class Client:
    model: str = DEFAULT_MODEL
    max_tokens: int = 1400
    temperature: float = 0.0
    max_retries: int = 6
    usage: Usage = field(default_factory=Usage)
    _client: object = None

    def __post_init__(self) -> None:
        if self._client is None:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise MissingAPIKey(
                    "ANTHROPIC_API_KEY is not set. This harness makes real API calls; "
                    "set the key in your environment or a .env file. "
                    "Analysis (`make reproduce`) needs no key."
                )
            from anthropic import Anthropic

            self._client = Anthropic(api_key=key)

    def complete(self, system: str, user: str) -> Response:
        """One call, with backoff on transient failures.

        A call that ultimately fails returns a Response carrying the error
        rather than raising, so one bad item cannot abort a multi-hour run —
        and the failure is recorded in the raw output rather than vanishing.
        """
        delay = 2.0
        last = ""
        for attempt in range(self.max_retries):
            try:
                msg = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
                self.usage.add(msg.usage.input_tokens, msg.usage.output_tokens)
                return Response(
                    text=text,
                    input_tokens=msg.usage.input_tokens,
                    output_tokens=msg.usage.output_tokens,
                    stop_reason=str(msg.stop_reason or ""),
                )
            except Exception as e:  # noqa: BLE001 - surfaced in the record, not swallowed
                last = f"{type(e).__name__}: {e}"
                if not any(k in last.lower() for k in RETRYABLE) or attempt == self.max_retries - 1:
                    break
                self.usage.retries += 1
                time.sleep(delay + random.random())
                delay = min(delay * 2, 60)
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
