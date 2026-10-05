"""Prior Labs metering and rate limits around every hosted TabPFN call.

Metering (quota_v3): tokens per day and per month (get_api_usage), at least MIN_TOKENS per
operation; over budget the API answers HTTP 429. Rate limits: fit and predict 60 a minute,
thinking fits 10 a minute and 30 an hour; a rate-limit 429 carries Retry-After.
tabpfn-client turns an API 429 into RuntimeError("... [HTTP 429] ...") and drops the headers,
so a response hook keeps the last Retry-After of this thread.

call       one hosted fit or predict: waits out a rate limit (RETRIES tries, MAX_WAIT seconds),
           a spent budget raises RuntimeError with its reset time
plan       before a run of fits: refuses what the token budget or the thinking hour can't hold
pace       before each thinking fit: at most 10 a minute, refuses the 31st in an hour
usage      the account's tokens used / limit and reset times, cached, None when unknown
estimate   Prior Labs' quote for fitting targets on a save (free, sizes only), None when unknown
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from functools import lru_cache

import numpy as np

MIN_TOKENS = 10_000  # the smallest charge of one operation
THINKING_MINUTE, THINKING_HOUR = 10, 30
RETRIES, MAX_WAIT, WAIT = 3, 90.0, 20.0  # WAIT when a 429 says nothing about how long
USAGE_TTL = 30.0  # seconds a usage answer is reused (the Settings page polls)
BUDGET_WORDS = ("quota", "budget", "token limit", "usage limit", "credit")

_last = threading.local()  # Retry-After of this thread's last 429
_hooked = False
_thinking: list[float] = []  # start times of this process's thinking fits
_pacing = threading.Lock()
_usage: tuple[float, dict | None] = (-USAGE_TTL, None)
_clock, _sleep = time.time, time.sleep  # tests replace these


def _remember(response) -> None:
    if response.status_code == 429:
        _last.retry_after = response.headers.get("retry-after")


def _hook() -> None:
    global _hooked
    if not _hooked:
        from tabpfn_client.client import ServiceClient

        ServiceClient.httpx_client.event_hooks["response"].append(_remember)
        _hooked = True


def _seconds(value) -> float | None:
    try:
        return max(float(value), 1.0)
    except (TypeError, ValueError):
        return None


def call(fn, *args, **kwargs):
    """fn(*args, **kwargs) against the Prior Labs API, through its 429s."""
    _hook()
    waited = 0.0
    for attempt in range(RETRIES):
        _last.retry_after = None
        try:
            return fn(*args, **kwargs)
        except RuntimeError as error:
            if "[HTTP 429]" not in str(error):
                raise
            spent = over_budget(usage(fresh=True))
            if spent or any(w in str(error).lower() for w in BUDGET_WORDS):
                raise RuntimeError(spent or "Your Prior Labs token budget is used up: it "
                                   "resets daily and monthly (see Settings)") from None
            wait = _seconds(getattr(_last, "retry_after", None)) or WAIT
            if attempt == RETRIES - 1 or waited + wait > MAX_WAIT:
                raise RuntimeError(f"Prior Labs is limiting how often we call it (rate limit): "
                                   f"try again in {round(wait)} seconds") from None
            _sleep(wait)
            waited += wait


def _span(u: dict, span: str) -> dict | None:
    used, limit = u.get(f"{span}_usage"), u.get(f"{span}_limit")
    if used is None or limit is None:
        return None
    return {"used": int(used), "limit": int(limit), "reset": _iso(u.get(f"{span}_reset_time"))}


def _words(iso: str | None) -> str:
    return iso.replace("T", " ").replace(":00Z", " UTC") if iso else "the end of the period"


def _iso(stamp: str | None) -> str | None:
    """'2026-10-06 00:00:00 UTC' (the API's form) -> '2026-10-06T00:00:00Z' for the page."""
    return stamp.replace(" UTC", "Z").replace(" ", "T") if isinstance(stamp, str) else stamp


def summary(u: dict | None) -> dict | None:
    """The page's view of get_api_usage: daily and monthly tokens, thinking fits."""
    if not u:
        return None
    out = {span: _span(u, span) for span in ("daily", "monthly")}
    if out["monthly"] is None and "usage_limit" in u:  # older servers: one budget
        out["monthly"] = {"used": int(u["current_usage"]), "limit": int(u["usage_limit"]),
                          "reset": _iso(u.get("reset_time"))}
    limit = int(u.get("thinking_fit_limit") or 0)
    out["thinking"] = {"used": int(u.get("thinking_fit_count") or 0), "limit": limit} \
        if limit > 0 else None
    out["thinking_hour"] = len(_recent(3600))
    return out


def over_budget(u: dict | None, need: int = MIN_TOKENS) -> str | None:
    """Why `need` more tokens can't be spent now, or None (also when usage is unknown)."""
    s = summary(u) or {}
    for span in ("daily", "monthly"):
        b = s.get(span)
        if b and b["limit"] >= 0 and b["used"] + need > b["limit"]:
            left = max(b["limit"] - b["used"], 0)
            return (f"Not enough Prior Labs tokens left: {left:,} of the {b['limit']:,} "
                    f"{span} tokens remain and this needs about {need:,}. The {span} budget "
                    f"resets at {_words(b['reset'])}")
    return None


def _fetch_usage(token: str) -> dict:
    """get_api_usage as the raw dict (the client only returns a sentence), 5 s at most."""
    from tabpfn_client.client import ServiceClient

    response = ServiceClient.httpx_client.post(
        ServiceClient.server_endpoints.get_api_usage.path,
        headers={"Authorization": f"Bearer {token}"}, timeout=5)
    response.raise_for_status()
    return response.json()


def usage(fresh: bool = False) -> dict | None:
    """The account's raw usage, or None when local, without a token or unreachable."""
    global _usage
    from .models import hosted

    token = os.environ.get("TABPFN_TOKEN")
    if not hosted() or not token:
        return None
    if fresh or _clock() - _usage[0] > USAGE_TTL:
        try:
            found = _fetch_usage(token)
        except Exception:
            found = None
        _usage = (_clock(), found)
    return _usage[1]


@lru_cache(maxsize=256)
def _quote(train: int, test: int, columns: int, operation: str, version: str) -> int:
    """One operation's quote: only the sizes of these empty arrays leave this computer."""
    import tabpfn_client

    tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
    X = np.empty((train, columns))
    if operation == "thinking_fit":
        quote = tabpfn_client.estimate_cost(X, operation=operation, model_version=version,
                                            thinking_effort="medium")
    else:
        quote = tabpfn_client.estimate_cost(X, np.empty((test, columns)), operation=operation,
                                            model_version=version)
    return quote.estimated_cost


def _operations(tier: str) -> list[tuple[str, str]]:
    """What one fold costs: the predict of its quantiles, plus thinking's own fit and mean."""
    if tier == "fast":
        return [("predict", "v3.5-fast")]
    if tier == "thinking":
        return [("predict", "v3.5"), ("thinking_fit", "v3.5"), ("thinking_predict", "v3.5")]
    return [("predict", "v3.5")]


def estimate(players, targets, tier: str, n_train: int = 10_000) -> int | None:
    """Tokens to fit `targets` for every player of a save with model `tier` (cross-fitted,
    predict.FOLDS folds each). None when local, without a token or unreachable."""
    from .models import hosted
    from .predict import FOLDS, folds
    from .schema import feature_columns

    if not hosted() or not os.environ.get("TABPFN_TOKEN") or not len(targets):
        return 0 if len(targets) == 0 else None
    fold, total = folds(players), 0
    try:
        for target in targets:
            known = players[target].notna().to_numpy()
            columns = len(feature_columns(target))
            for k in range(FOLDS):
                train = int(min(n_train, ((fold != k) & known).sum()))
                test = int((fold == k).sum())
                total += sum(_quote(train, test, columns, op, version)
                             for op, version in _operations(tier))
    except Exception:
        return None
    return total


def _recent(seconds: float) -> list[float]:
    now = _clock()
    return [t for t in _thinking if now - t < seconds]


def _at(stamp: float) -> str:
    return datetime.fromtimestamp(stamp).strftime("%H:%M")


def plan(players, targets, tier: str, n_train: int = 10_000) -> None:
    """Refuse up front a run of fits the account can't finish: more thinking fits than this
    hour has left, or more tokens than the budget has left (skipped when unknown)."""
    from .models import hosted
    from .predict import FOLDS

    if not hosted() or not targets:
        return
    if tier == "thinking":
        need, hour = len(targets) * FOLDS, sorted(_recent(3600))
        server = (summary(usage()) or {}).get("thinking")
        left = THINKING_HOUR - len(hour)
        if server:
            left = min(left, server["limit"] - server["used"])
        if need > left:
            free = f" The next one frees up at {_at(hour[0] + 3600)}." if hour else ""
            raise RuntimeError(f"TabPFN-3.5 Thinking: Prior Labs allows {THINKING_HOUR} "
                               f"thinking fits an hour and this needs {need}, but only "
                               f"{max(left, 0)} are left.{free} Or pick another model.")
    u = usage()
    if u:
        spent = over_budget(u, estimate(players, targets, tier, n_train) or MIN_TOKENS)
        if spent:
            raise RuntimeError(spent)


def pace() -> None:
    """Before each thinking fit: wait for a free slot in the minute, refuse past the hour."""
    with _pacing:
        hour = sorted(_recent(3600))
        if len(hour) >= THINKING_HOUR:
            raise RuntimeError(f"TabPFN-3.5 Thinking: the {THINKING_HOUR} thinking fits of "
                               f"this hour are used up. Try again after {_at(hour[0] + 3600)}, "
                               "or pick another model.")
        minute = [t for t in hour if _clock() - t < 60]
        if len(minute) >= THINKING_MINUTE:
            _sleep(60 - (_clock() - minute[-THINKING_MINUTE]))
        _thinking[:] = [*hour, _clock()]
