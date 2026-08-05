#!/usr/bin/env python3
"""Verify credentials and confirm the pinned model exists — before spending.

    python scripts/auth_check.py --model gpt-4.1-mini-2025-04-14

Makes exactly ONE tiny call. Lists the dated snapshots the account can actually
reach, so the model is chosen from real availability rather than from memory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from epr.model import (
    DEFAULT_MODEL,
    OPENAI,
    PRICING,
    Client,
    MissingCredentials,
    is_reasoning_model,
    provider_for,
)

ROOT = Path(__file__).resolve().parents[1]


def list_models(client: Client, contains: str) -> list[str]:
    """Ask the provider what this account can reach. Best effort."""
    try:
        if client.provider == OPENAI:
            names = [m.id for m in client._client.models.list().data]
        else:
            names = [m.id for m in client._client.models.list(limit=100).data]
    except Exception as e:  # noqa: BLE001
        print(f"  (could not list models: {type(e).__name__}: {e})")
        return []
    return sorted(n for n in names if contains in n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--filter", default="", help="substring to filter the model listing")
    args = ap.parse_args()

    provider = provider_for(args.model)
    print(f"\n=== auth check: provider={provider} model={args.model} ===\n")

    seen = [
        k
        for k in (
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
        )
        if os.environ.get(k)
    ]
    print(f"  credentials visible : {', '.join(seen) if seen else 'NONE'}")
    if (ROOT / ".env").exists():
        print("  .env                : present (loaded automatically)")

    try:
        client = Client(model=args.model)
    except MissingCredentials as e:
        print(f"\n{e}\n")
        return 3

    print("\n  reachable dated snapshots:")
    for name in list_models(client, args.filter or ("gpt-" if provider == OPENAI else "claude-")):
        mark = "  <-- selected" if name == args.model else ""
        price = PRICING.get(name)
        cost = f"  ${price[0]:.2f}/${price[1]:.2f} per Mtok" if price else "  (pricing unknown)"
        print(f"    {name}{cost}{mark}")

    print("\n  making one tiny call...")
    resp = client.complete(
        "You are a test harness probe. Answer with exactly one word.",
        "Reply with the single word: ready",
    )
    if resp.error:
        print(f"\n  FAILED: {resp.error}\n")
        return 4

    print(f"    response      : {resp.text.strip()[:60]!r}")
    print(f"    tokens        : {resp.input_tokens} in / {resp.output_tokens} out")
    if resp.reasoning_tokens:
        print(f"    reasoning     : {resp.reasoning_tokens} (billed at the output rate)")
    if is_reasoning_model(args.model):
        print("    NOTE          : reasoning model — temperature is not applied.")
        print("                    HYPOTHESIS.md pins temperature=0.0; a reasoning")
        print("                    model cannot honour that. Prefer a non-reasoning")
        print("                    snapshot, or record the deviation as an amendment.")

    if args.model not in PRICING:
        print(f"\n  WARNING: no pricing entry for {args.model}. Cost projections will read $0.00.")
        print("           Add it to PRICING in src/epr/model.py before running a phase.")
    else:
        cin, cout = PRICING[args.model]
        print(f"\n  pricing used for projections: ${cin:.2f} in / ${cout:.2f} out per Mtok")
        print("  Confirm against current provider pricing before a large run.")

    print("\n  OK — credentials work and the model responds.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
