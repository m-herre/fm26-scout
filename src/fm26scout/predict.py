"""The predicted player DB: every player, every target, a TabPFN estimate with an 80% range.

Cross-fitted: players are cut into folds and each fold is predicted by a model whose training
rows come only from the other folds, so no player is ever predicted by a model that saw his own
hidden values. The trust report then checks whether the ranges deserve to be shown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import TabPFN
from .schema import TARGETS, feature_columns

LEVELS = [0.1, 0.5, 0.9]  # the 80% range and the median estimate
FOLDS = 5
POINTS = 81  # where each player's CDF is kept: 2.5 PA, 0.25 of a 1-20 value, x1.28 in price


def folds(players: pd.DataFrame, k: int = FOLDS, seed: int = 0) -> np.ndarray:
    order = np.random.default_rng(seed).permutation(len(players))
    fold = np.empty(len(players), dtype=int)
    fold[order] = np.arange(len(players)) % k
    return fold


def grid(name: str) -> np.ndarray:
    """The values a target's CDF is kept at, on the model's scale (log1p for prices)."""
    target = TARGETS[name]
    return np.linspace(0, 20 if target.log else target.high, POINTS)


def predict_target(players: pd.DataFrame, name: str, n_train: int = 10_000,
                   seed: int = 0, on_fold=None, tier: str = "default"
                   ) -> tuple[pd.DataFrame, np.ndarray]:
    """Out-of-fold median and 80% range of `name` for every player, and his full CDF at
    grid(name) (players x points). `on_fold(k)` after each fold, for progress bars."""
    target, features = TARGETS[name], feature_columns(name)
    fold, known = folds(players, seed=seed), players[name].notna().to_numpy()
    q = np.full((len(LEVELS), len(players)), np.nan)
    cdf = np.full((len(players), POINTS), np.nan, dtype=np.float32)
    for k in range(FOLDS):
        context = players[(fold != k) & known]
        context = context.sample(min(n_train, len(context)), random_state=seed + k)
        y = np.log1p(context[name]) if target.log else context[name]
        rows = fold == k
        q[:, rows], cdf[rows] = TabPFN(context[features], y, seed, tier=tier).distribution(
            players.loc[rows, features], LEVELS, grid(name))
        if on_fold:
            on_fold(k)
    q = np.sort(q, axis=0)  # quantiles must not cross
    q = np.expm1(q) if target.log else np.clip(q, target.low, target.high)
    return pd.DataFrame({f"lo_{name}": q[0], f"pred_{name}": q[1], f"hi_{name}": q[2]}), cdf


def chance(lo, median, hi, at_least: float, dist=None) -> np.ndarray:
    """P(value >= at_least). With `dist` (grid in the target's units, CDF rows): read off
    TabPFN's full distribution. Without (saves fitted before it was kept): from the 10/50/90%
    points, the CDF running linearly between them and the outer slopes continuing into the
    tails. Checked against the truth in the trust report (chance_reliability)."""
    if dist is not None:
        x, cdf = dist
        i = int(np.clip(np.searchsorted(x, at_least), 1, len(x) - 1))
        w = np.clip((at_least - x[i - 1]) / (x[i] - x[i - 1]), 0, 1)
        cdf = np.asarray(cdf[:, i - 1 : i + 1], dtype=float)
        return np.clip(1 - (cdf[:, 0] * (1 - w) + cdf[:, 1] * w), 0, 1)
    lo, median, hi = (np.asarray(v, dtype=float) for v in (lo, median, hi))
    below = 0.5 - (median - at_least) * 0.4 / np.maximum(median - lo, 1e-6)
    above = 0.5 + (at_least - median) * 0.4 / np.maximum(hi - median, 1e-6)
    return 1 - np.clip(np.where(at_least <= median, below, above), 0, 1)


def ranking(c, real, top=(50, 100)) -> dict:
    """How well a chance ranks a rare event: base rate, ROC AUC, average precision (PR-AUC)
    and how many of the `top` players by chance really reach it."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    c, real = np.asarray(c, float), np.asarray(real, float)
    both = 0 < real.sum() < len(real)
    order = np.argsort(-c, kind="stable")
    return {"n": len(real), "base_rate": float(real.mean()),
            "auc": float(roc_auc_score(real, c)) if both else None,
            "ap": float(average_precision_score(real, c)) if both else None,
            **{f"top{k}": int(real[order[:k]].sum()) for k in top}}


def chance_reliability(db: pd.DataFrame, players: pd.DataFrame, target: str,
                       at_least: float, dist=None) -> pd.DataFrame:
    """Quoted chance vs how often it really happened, in bins; attrs: Brier score, AUC, where
    the chance came from, and the ranking checks (`ranking`) for all players and those aged
    21 or under."""
    from sklearn.metrics import roc_auc_score

    c = chance(db[f"lo_{target}"], db[f"pred_{target}"], db[f"hi_{target}"], at_least, dist)
    ok = players[target].notna().to_numpy()
    young = (players["age"] <= 21).to_numpy()[ok]
    c, real = c[ok], (players[target][ok] >= at_least).to_numpy(float)
    bins = pd.cut(c, [0, 0.05, 0.2, 0.4, 0.6, 0.8, 1], include_lowest=True)
    table = pd.DataFrame({"quoted": c, "real": real}).groupby(bins, observed=True).agg(
        players=("real", "size"), quoted=("quoted", "mean"), real=("real", "mean"))
    table.attrs = {"brier": float(np.mean((c - real) ** 2)),
                   "auc": float(roc_auc_score(real, c)) if 0 < real.sum() < len(real) else None,
                   "source": "full distribution" if dist is not None else "10/50/90% points",
                   "ranking": {"all players": ranking(c, real),
                               "age 21 or under": ranking(c[young], real[young])}}
    return table


def trust(db: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Per target: skill, range coverage, and whether narrow ranges really have smaller
    errors. A target passes only if confidence tracks the actual error."""
    rows = []
    for name, target in TARGETS.items():
        if f"pred_{name}" not in db:
            continue
        f = (lambda v: np.log1p(v)) if target.log else (lambda v: v)
        y = f(players[name].to_numpy(float))
        pred, lo, hi = (f(db[f"{p}_{name}"].to_numpy(float)) for p in ("pred", "lo", "hi"))
        ok = ~np.isnan(y)
        err, width = np.abs(pred - y)[ok], (hi - lo)[ok]
        by_width = pd.Series(err).groupby(pd.qcut(width, 3, labels=False, duplicates="drop")).mean()
        skill = 1 - err.mean() / np.abs(y[ok] - np.median(y[ok])).mean()
        coverage = ((lo <= y) & (y <= hi))[ok].mean()
        tracks = by_width.iloc[0] < by_width.iloc[-1]
        rows.append({
            "target": name, "n": int(ok.sum()), "mae": err.mean(), "skill": skill,
            "coverage_80": coverage, "mae_narrow": by_width.iloc[0], "mae_wide": by_width.iloc[-1],
            "range_shown": bool(tracks and 0.7 <= coverage <= 0.9),
            "verdict": "reliable" if skill >= 0.4 else "rough" if skill >= 0.15
            else "not predictable",
        })
    return pd.DataFrame(rows)


def trust_markdown(table: pd.DataFrame, reliability: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# Trust report",
        "",
        "Out-of-fold TabPFN predictions for every player. Coverage: share of true values inside",
        "the 80% range (should be ~0.80). Narrow/wide: error for the third of players with the",
        "narrowest / widest ranges. Market value in log units.",
        "",
        "| target | verdict | skill | coverage | MAE | narrow | wide | range shown |",
        "|---" * 8 + "|",
    ]
    for r in table.itertuples():
        lines.append(f"| {r.target} | {r.verdict} | {r.skill:.0%} | {r.coverage_80:.2f} | "
                     f"{r.mae:.2f} | {r.mae_narrow:.2f} | {r.mae_wide:.2f} | "
                     f"{'yes' if r.range_shown else 'no'} |")
    for title, rel in reliability.items():
        a, auc = rel.attrs, rel.attrs.get("auc")
        lines += ["", f"## Chance of {title}: quoted vs real", "",
                  f"From the {a.get('source', '10/50/90% points')}: Brier "
                  f"{a.get('brier', float('nan')):.4f}, AUC "
                  + ("-" if auc is None else f"{auc:.3f}"), "",
                  "| players | quoted | real |", "|---|---|---|"]
        lines += [f"| {r.players:,} | {r.quoted:.0%} | {r.real:.0%} |" for r in rel.itertuples()]
        if a.get("ranking"):
            lines += ["", "A rare event: the base rate is the share of players who really reach "
                      "it, PR-AUC", "(average precision) the ranking quality against that, top "
                      "50/100 how many of the", "players with the highest chance really reach "
                      "it.", "", "| players | n | base rate | AUC | PR-AUC | top 50 | top 100 |",
                      "|---" * 7 + "|"]
            lines += [f"| {who} | {r['n']:,} | {r['base_rate']:.1%} | " + " | ".join(
                "-" if r[k] is None else f"{r[k]:.3f}" for k in ("auc", "ap"))
                + f" | {r['top50']} of {min(50, r['n'])} | {r['top100']} of {min(100, r['n'])} |"
                for who, r in a["ranking"].items()]
    return "\n".join(lines) + "\n"
