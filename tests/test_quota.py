"""Prior Labs rate limits, token budget and cost quotes around hosted calls. No network."""

import pandas as pd
import pytest

from fm26scout import quota, settings

RATE_LIMITED = RuntimeError("Fail to call predict: [HTTP 429] Rate limit exceeded.")
USAGE = {"daily_usage": 566_762, "daily_limit": 10_000_000,
         "daily_reset_time": "2026-10-06 00:00:00 UTC", "monthly_usage": 686_762,
         "monthly_limit": 20_000_000, "monthly_reset_time": "2026-11-01 00:00:00 UTC",
         "thinking_fit_count": 0, "thinking_fit_limit": 0}


class Response:
    def __init__(self, status, wait=None):
        self.status_code, self.headers = status, {"retry-after": wait} if wait else {}


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "a-token-from-the-shell-123")
    monkeypatch.setenv("FM26_TABPFN", "hosted")
    monkeypatch.setattr(quota, "_hook", lambda: None)
    waits = []
    monkeypatch.setattr(quota, "_sleep", waits.append)
    return waits


def flaky(times, wait="7"):
    """Rate-limited `times` times (the response hook saw Retry-After), then answers 42."""
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= times:
            quota._remember(Response(429, wait))
            raise RATE_LIMITED
        return 42
    return fn, calls


def test_rate_limit_waits_retry_after_then_retries(cloud, monkeypatch):
    monkeypatch.setattr(quota, "_fetch_usage", lambda token: USAGE)
    fn, calls = flaky(2)
    assert quota.call(fn) == 42 and len(calls) == 3 and cloud == [7.0, 7.0]


def test_rate_limit_gives_up_with_words(cloud):
    fn, calls = flaky(5, wait="60")  # 60 + 60 s would pass MAX_WAIT
    with pytest.raises(RuntimeError, match=r"rate limit\): try again in 60 seconds"):
        quota.call(fn)
    assert cloud == [60.0] and len(calls) == 2


def test_other_errors_pass_unchanged(cloud):
    def broken():
        raise RuntimeError("Fail to call fit: [HTTP 400] bad data.")
    with pytest.raises(RuntimeError, match="HTTP 400"):
        quota.call(broken)
    assert cloud == []


def test_spent_budget_says_when_it_resets(cloud, monkeypatch):
    monkeypatch.setattr(quota, "_fetch_usage",
                        lambda token: USAGE | {"daily_usage": 9_995_000})
    fn, calls = flaky(1)
    with pytest.raises(RuntimeError, match=r"5,000 of the 10,000,000 daily tokens remain.*"
                                           r"resets at 2026-10-06 00:00 UTC"):
        quota.call(fn)
    assert cloud == [] and len(calls) == 1  # no retry: waiting won't help


def test_budget_429_without_usage_still_says_budget(cloud):
    def spent():
        raise RuntimeError("Fail to call fit: [HTTP 429] Monthly quota exceeded.")
    with pytest.raises(RuntimeError, match="token budget is used up"):
        quota.call(spent)


def test_thinking_fits_are_counted_and_paced(cloud, monkeypatch):
    now = [10_000.0]
    monkeypatch.setattr(quota, "_clock", lambda: now[0])
    players = pd.DataFrame({"potential_ability": [1.0]})
    quota._thinking[:] = [now[0] - 3000 + i for i in range(25)]  # 25 this hour
    quota.plan(players, ["potential_ability"], "fast")  # other models: no thinking limit
    with pytest.raises(RuntimeError, match="needs 10, but only 5 are left"):
        quota.plan(players, ["potential_ability", "market_value"], "thinking")
    quota.plan(players, ["potential_ability"], "thinking")  # 5 fits fit in
    quota._thinking[:] = [now[0] - 30 + i for i in range(10)]  # 10 in the last minute
    quota.pace()
    assert cloud == [30.0] and len(quota._thinking) == 11
    quota._thinking[:] = [now[0] - 3000 + i for i in range(30)]
    with pytest.raises(RuntimeError, match="30 thinking fits of this hour are used up"):
        quota.pace()


def test_plan_refuses_more_tokens_than_are_left(cloud, monkeypatch):
    monkeypatch.setattr(quota, "_fetch_usage",
                        lambda token: USAGE | {"monthly_usage": 19_960_000})
    monkeypatch.setattr(quota, "_quote", lambda *a: 10_000)
    players = pd.DataFrame({"potential_ability": range(100)}, dtype=float)
    with pytest.raises(RuntimeError, match="this needs about 50,000"):
        quota.plan(players, ["potential_ability"], "default")


def test_cost_quote_per_model_and_fail_soft(cloud, monkeypatch):
    seen = []
    monkeypatch.setattr(quota, "_quote", lambda *a: seen.append(a) or 10_000)
    players = pd.DataFrame({"potential_ability": [float(i) for i in range(100)]})
    assert quota.estimate(players, ["potential_ability"], "default") == 50_000
    assert quota.estimate(players, ["potential_ability"], "thinking") == 150_000
    assert {a[3:] for a in seen} == {("predict", "v3.5"), ("thinking_fit", "v3.5"),
                                     ("thinking_predict", "v3.5")}
    assert seen[0][:3] == (80, 20, 171)  # train and predicted rows of a fold, features
    monkeypatch.setattr(quota, "_quote", lambda *a: 1 / 0)
    assert quota.estimate(players, ["potential_ability"], "fast") is None
    monkeypatch.setenv("FM26_TABPFN", "local")
    assert quota.estimate(players, ["potential_ability"], "fast") is None


def test_usage_in_settings_and_fail_soft(cloud, monkeypatch):
    calls = []
    monkeypatch.setattr(quota, "_fetch_usage", lambda token: calls.append(1) or USAGE)
    usage = settings.runtime()["usage"]
    assert usage["daily"] == {"used": 566_762, "limit": 10_000_000,
                              "reset": "2026-10-06T00:00:00Z"}
    assert usage["monthly"]["limit"] == 20_000_000 and usage["thinking"] is None
    settings.runtime()
    assert len(calls) == 1  # cached: the Settings page polls
    monkeypatch.setattr(quota, "_usage", (-quota.USAGE_TTL, None))
    monkeypatch.setattr(quota, "_fetch_usage", lambda token: 1 / 0)
    assert settings.runtime()["usage"] is None
    monkeypatch.delenv("TABPFN_TOKEN")
    monkeypatch.setattr(quota, "_usage", (-quota.USAGE_TTL, USAGE))
    assert settings.runtime()["usage"] is None  # no token: nothing asked
