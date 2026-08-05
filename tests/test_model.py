"""Client tests: provider routing, credentials, cost, and the budget ceiling.

No network. The provider SDKs are injected as fakes, because the thing worth
testing is our routing and accounting, not theirs.
"""

from __future__ import annotations

import pytest

from epr.model import (
    ANTHROPIC,
    OPENAI,
    Client,
    MissingCredentials,
    Usage,
    is_reasoning_model,
    provider_for,
    resolve_credentials,
)
from epr.runner import BudgetExceeded, run_condition

# ------------------------------------------------------------ provider routing


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4.1-mini-2025-04-14", OPENAI),
        ("o4-mini-2025-04-16", OPENAI),
        ("claude-haiku-4-5-20251001", ANTHROPIC),
        ("claude-opus-5", ANTHROPIC),
    ],
)
def test_provider_is_inferred_from_the_model_id(model, expected):
    assert provider_for(model) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("o4-mini-2025-04-16", True),
        ("o3-mini-2025-01-31", True),
        ("gpt-5-foo", True),
        ("gpt-4.1-mini-2025-04-14", False),
        ("claude-haiku-4-5-20251001", False),
    ],
)
def test_reasoning_models_are_detected(model, expected):
    """They reject `temperature` and bill hidden tokens; both must be handled."""
    assert is_reasoning_model(model) is expected


# ---------------------------------------------------------------- credentials


def test_openai_key_is_picked_up(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert resolve_credentials(OPENAI) == {"api_key": "sk-test"}


def test_openai_gateway_base_url_is_honoured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.internal/v1")
    creds = resolve_credentials(OPENAI)
    assert creds["base_url"] == "https://gateway.internal/v1"


def test_anthropic_bearer_token_path_for_oauth_gateways(monkeypatch):
    """An OAuth-secured gateway issues a bearer, not an api key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "oauth-bearer")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gw.example/v1")
    creds = resolve_credentials(ANTHROPIC)
    assert creds["auth_token"] == "oauth-bearer"
    assert creds["base_url"] == "https://gw.example/v1"
    assert "api_key" not in creds


def test_missing_credentials_names_the_variable_to_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(MissingCredentials, match="OPENAI_API_KEY"):
        resolve_credentials(OPENAI)


# ---------------------------------------------------------------------- usage


def test_cost_uses_the_per_million_rates():
    u = Usage()
    u.add(1_000_000, 1_000_000)
    # gpt-4.1-mini: $0.40 in, $1.60 out
    assert u.cost("gpt-4.1-mini-2025-04-14") == pytest.approx(2.00)


def test_unknown_model_costs_zero_rather_than_guessing():
    u = Usage()
    u.add(1_000_000, 1_000_000)
    assert u.cost("some-unlisted-model") == 0.0


def test_failed_calls_are_counted_not_hidden():
    u = Usage()
    u.failures += 1
    assert "1 failed" in u.summary("gpt-4.1-mini-2025-04-14")


# ------------------------------------------------------------- budget ceiling


class _FakeItem:
    """Minimal stand-in for datasets.Item, enough for run_condition."""

    def __init__(self, uid):
        self.uid = uid
        self.dataset = "prontoqa"
        self.depth = 1
        self.theory = None
        self.gold_steps = []
        self.gold_answer = "True"
        self.answer_space = ("True", "False")
        self.question = "q"
        self.meta = {}

    @property
    def premise_block(self):
        return "fact1: x"

    @property
    def supports_step_metrics(self):
        return False


class _FakeSDK:
    """Returns a canned response and bills a fixed number of tokens."""

    def __init__(self, per_call_in=1_000_000):
        self.per_call_in = per_call_in
        self.calls = 0

        class _Completions:
            def create(inner, **kw):
                self.calls += 1
                return type(
                    "R",
                    (),
                    {
                        "choices": [
                            type(
                                "C",
                                (),
                                {
                                    "message": type("M", (), {"content": "ANSWER: True"})(),
                                    "finish_reason": "stop",
                                },
                            )()
                        ],
                        "usage": type(
                            "U",
                            (),
                            {
                                "prompt_tokens": self.per_call_in,
                                "completion_tokens": 0,
                                "completion_tokens_details": None,
                            },
                        )(),
                    },
                )()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_run_stops_when_real_spend_crosses_the_ceiling(tmp_path, monkeypatch):
    """A pre-run projection is an estimate; only real usage enforces a ceiling."""
    from epr import runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS", tmp_path)
    fake = _FakeSDK(per_call_in=1_000_000)  # $0.40 of input per call
    client = Client(model="gpt-4.1-mini-2025-04-14", _client=fake)

    items = [_FakeItem(f"i{i}") for i in range(20)]
    ceiling = 1.00  # ~2.5 calls' worth

    with pytest.raises(BudgetExceeded):
        run_condition(
            client,
            items,
            "direct_zs",
            [],
            seed=1,
            phase="test",
            stop_check=lambda: client.usage.cost(client.model) >= ceiling,
        )

    assert client.usage.cost(client.model) >= ceiling
    assert fake.calls < len(items), "must stop early, not run the whole condition"
    written = (tmp_path / "test" / "prontoqa" / "direct_zs_seed1.jsonl").read_text().splitlines()
    assert len(written) == fake.calls, "every paid-for call must already be on disk"


def test_records_are_flushed_per_item_so_a_stop_loses_nothing(tmp_path, monkeypatch):
    from epr import runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS", tmp_path)
    fake = _FakeSDK(per_call_in=10)
    client = Client(model="gpt-4.1-mini-2025-04-14", _client=fake)
    items = [_FakeItem(f"i{i}") for i in range(5)]

    run_condition(client, items, "direct_zs", [], seed=1, phase="test")
    path = tmp_path / "test" / "prontoqa" / "direct_zs_seed1.jsonl"
    assert len(path.read_text().splitlines()) == 5

    # resume: a second pass must not re-pay for completed items
    before = fake.calls
    run_condition(client, items, "direct_zs", [], seed=1, phase="test")
    assert fake.calls == before


def test_concurrent_run_writes_every_record_exactly_once(tmp_path, monkeypatch):
    """Parallel workers share one file handle; interleaved writes would corrupt it."""
    import json

    from epr import runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS", tmp_path)
    fake = _FakeSDK(per_call_in=10)
    client = Client(model="gpt-4.1-mini-2025-04-14", _client=fake)
    items = [_FakeItem(f"i{i}") for i in range(60)]

    run_condition(client, items, "direct_zs", [], seed=1, phase="test", concurrency=8)

    lines = (tmp_path / "test" / "prontoqa" / "direct_zs_seed1.jsonl").read_text().splitlines()
    assert len(lines) == 60
    uids = [json.loads(ln)["uid"] for ln in lines]  # every line must be valid JSON
    assert sorted(uids) == sorted(i.uid for i in items), "no dupes, no losses"


def test_concurrent_run_resumes_without_repaying(tmp_path, monkeypatch):
    from epr import runner as runner_mod

    monkeypatch.setattr(runner_mod, "RESULTS", tmp_path)
    fake = _FakeSDK(per_call_in=10)
    client = Client(model="gpt-4.1-mini-2025-04-14", _client=fake)
    items = [_FakeItem(f"i{i}") for i in range(30)]

    run_condition(client, items, "direct_zs", [], seed=1, phase="test", concurrency=8)
    after_first = fake.calls
    run_condition(client, items, "direct_zs", [], seed=1, phase="test", concurrency=8)
    assert fake.calls == after_first
