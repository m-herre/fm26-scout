"""evaluate(): every target x every model on the fixed split, plus error by player slice.

Held-out players' targets never reach a model: each model sees only train-split rows
(targets included) and the visible features of test-split rows.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .models import MODELS, fit_predict
from .schema import REPUTATION, TARGETS, feature_columns
from .split import AGE_LABELS, age_band, position_group

SLICE_ORDER = [*AGE_LABELS, "GK", "DEF", "MID", "ATT", *REPUTATION[1], "known", "missing"]
RIVALS = ("ridge", "random_forest", "catboost")


def predict_all(players: pd.DataFrame, targets=None, models=MODELS, n_train: int = 10_000,
                seed: int = 0, log=print) -> pd.DataFrame:
    """Long table of test-set predictions: target, model, player_id, truth, prediction."""
    rows = []
    for name in targets or TARGETS:
        target, features = TARGETS[name], feature_columns(name)
        known = players[players[name].notna()]
        train = known[known["split"] == "train"]
        train = train.sample(min(n_train, len(train)), random_state=seed)
        test = known[known["split"] == "test"]
        y = np.log1p(train[name]) if target.log else train[name]
        for model in models:
            start = time.perf_counter()
            pred = fit_predict(model, train[features], y, test[features], seed)
            pred = np.expm1(pred) if target.log else np.clip(pred, target.low, target.high)
            log(f"{name:<20} {model:<14} {time.perf_counter() - start:6.1f}s")
            rows.append(pd.DataFrame({"target": name, "model": model,
                                      "player_id": test["player_id"].to_numpy(),
                                      "truth": test[name].to_numpy(), "pred": pred}))
    return pd.concat(rows, ignore_index=True)


def _scores(group: pd.DataFrame) -> pd.Series:
    t, p = group["truth"].to_numpy(), group["pred"].to_numpy()
    if TARGETS[group.name[0]].log:  # group.name = (target, model, ...)
        t, p = np.log1p(t), np.log1p(p)  # errors of log value; exp(MAE) - 1 ~ typical % off
    err = p - t
    return pd.Series({
        "n": len(t),
        "mae": np.abs(err).mean(),
        "rmse": np.sqrt((err**2).mean()),
        "r2": 1 - (err**2).sum() / ((t - t.mean()) ** 2).sum(),
        "spearman": spearmanr(t, p).statistic if np.ptp(p) > 0 else np.nan,
    })


def _errors(frame: pd.DataFrame, target: str) -> np.ndarray:
    """Absolute errors in the scale MAE is reported on (log1p for prices)."""
    t, p = frame["truth"].to_numpy(float), frame["pred"].to_numpy(float)
    if TARGETS[target].log:
        t, p = np.log1p(t), np.log1p(p)
    return np.abs(p - t)


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1_000, seed: int = 0
              ) -> tuple[float, float, float]:
    """Mean of a - b (paired absolute errors) and its 95% bootstrap interval over players."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def verdict(lo: float, hi: float) -> str:
    """TabPFN minus rival: better if the whole interval is below 0, worse if above."""
    return "better" if hi < 0 else "worse" if lo > 0 else "tied"


def versus(mine: pd.DataFrame, theirs: pd.DataFrame, target: str, **kw) -> dict:
    """TabPFN (`mine`) vs one rival on the same test players: MAE difference with its CI."""
    pair = mine.merge(theirs, on="player_id", suffixes=("", "_rival"))
    rival = pair[["player_id", "truth_rival", "pred_rival"]].rename(
        columns=lambda c: c.removesuffix("_rival"))
    diff, lo, hi = paired_ci(_errors(pair, target), _errors(rival, target), **kw)
    return {"delta": diff, "lo": lo, "hi": hi, "verdict": verdict(lo, hi)}


def compare(predictions: pd.DataFrame, rivals=RIVALS, **kw) -> pd.DataFrame:
    """Per target: TabPFN against the best rival (lowest MAE) and against the median guess,
    each a paired difference in MAE (TabPFN minus other) with a 95% bootstrap CI."""
    rows = []
    for target, group in predictions.groupby("target", sort=False):
        by = {m: g for m, g in group.groupby("model")}
        if "tabpfn" not in by:
            continue
        mae = {m: _errors(g, target).mean() for m, g in by.items()}
        best = min((m for m in rivals if m in by), key=mae.get)
        row = {"target": target, "tabpfn": mae["tabpfn"], "best_other": best,
               "best_other_mae": mae[best], **versus(by["tabpfn"], by[best], target, **kw)}
        if "median" in by:
            vs = versus(by["tabpfn"], by["median"], target, **kw)
            row |= {"median": mae["median"], **{f"median_{k}": v for k, v in vs.items()}}
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per target x model, with the MAE removed relative to predicting the median."""
    table = (predictions.groupby(["target", "model"], sort=False)
             .apply(_scores, include_groups=False).reset_index())
    base = table[table["model"] == "median"].set_index("target")["mae"]
    table["skill"] = 1 - table["mae"] / table["target"].map(base)
    return table


def slices(predictions: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """MAE per target x model within each player slice."""
    info = pd.DataFrame({
        "player_id": players["player_id"],
        "age": age_band(players),
        "position": position_group(players),
        "reputation": players["reputation"].astype(str),
        "market_value": np.where(players["market_value"].notna(), "known", "missing"),
    })
    joined = predictions.merge(info, on="player_id")
    parts = []
    for by in ("age", "position", "reputation", "market_value"):
        part = joined.groupby(["target", "model", by], observed=True, sort=False).apply(
            _scores, include_groups=False).reset_index().rename(columns={by: "slice"})
        parts.append(part.assign(by=by))
    return pd.concat(parts, ignore_index=True)[
        ["target", "model", "by", "slice", "n", "mae", "r2"]]


def _ci(diff: float, lo: float, hi: float) -> str:
    return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def markdown(table: pd.DataFrame, by_slice: pd.DataFrame, meta: dict,
             versus_best: pd.DataFrame | None = None, more: pd.DataFrame | None = None,
             more_meta: dict | None = None) -> str:
    mae = table.pivot(index="target", columns="model", values="mae")
    mae = mae[[m for m in MODELS if m in mae.columns]].loc[table["target"].unique()]
    lines = [
        "# Hidden-attribute benchmark",
        "",
        " · ".join(f"{k}: {v}" for k, v in meta.items()),
        "",
        "Mean absolute error on held-out players (lower is better; market value on log scale).",
        "Skill = share of the median-guess error removed by the best model.",
        "TabPFN vs best other: MAE of TabPFN minus MAE of the best of ridge, random forest and",
        "CatBoost, with a 95% bootstrap interval (1,000 paired resamples of the test players,",
        "seed 0). Better: the whole interval is below 0; tied: it contains 0; worse: above 0.",
        "⚠ marks targets where TabPFN is not better than the median guess by the same test.",
        "",
        "| target | " + " | ".join(mae.columns) + " | best | skill | R² (best) "
        "| TabPFN vs best other (95% CI) | verdict |",
        "|---" * (len(mae.columns) + 6) + "|",
    ]
    vs = pd.DataFrame() if versus_best is None else versus_best.set_index("target")
    for target, row in mae.iterrows():
        best = row.drop("median").idxmin()
        cells = [f"**{v:.3f}**" if m == best else f"{v:.3f}" for m, v in row.items()]
        score = table[(table["target"] == target) & (table["model"] == best)].iloc[0]
        line = (f"| {target} | " + " | ".join(cells)
                + f" | {best} | {score.skill:.0%} | {score.r2:.3f} |")
        if target in vs.index:
            r = vs.loc[target]
            flag = "" if r.get("median_verdict", "better") == "better" else \
                f" ⚠ vs median: {r.median_verdict} {_ci(r.median_delta, r.median_lo, r.median_hi)}"
            line += f" {_ci(r.delta, r.lo, r.hi)} vs {r.best_other} | {r.verdict}{flag} |"
        else:
            line += " – | – |"
        lines.append(line)
    if len(vs):
        counts = vs["verdict"].value_counts()
        lines += ["", "Against the best other model TabPFN is "
                  + ", ".join(f"{v} on {counts[v]}" for v in ("better", "tied", "worse")
                              if v in counts) + f" of {len(vs)} targets."]
    if more is not None and len(more):
        lines += ["", "## Baselines with 4× the rows", "",
                  " · ".join(f"{k}: {v}" for k, v in (more_meta or {}).items()), "",
                  "The rivals refitted on more training rows (the whole train split when it is",
                  "smaller); TabPFN keeps its 10,000-row fit. Same test players, same paired CI.",
                  "", "| target | model | train rows | MAE (more rows) | MAE (10,000) | TabPFN "
                  "| TabPFN vs it (95% CI) | verdict |", "|---" * 8 + "|"]
        for r in more.itertuples():
            lines.append(f"| {r.target} | {r.model} | {r.n_train:,} | {r.mae:.3f} | "
                         f"{r.mae_10k:.3f} | {r.tabpfn:.3f} | {_ci(r.delta, r.lo, r.hi)} "
                         f"| {r.verdict} |")
    lines += ["", "## TabPFN error by slice (MAE)", ""]
    tab = by_slice[by_slice["model"] == "tabpfn"]
    if len(tab):
        wide = tab.pivot_table(index="target", columns="slice", values="mae")
        wide = wide[[s for s in SLICE_ORDER if s in wide.columns]]
        lines.append("| target | " + " | ".join(wide.columns) + " |")
        lines.append("|---" * (len(wide.columns) + 1) + "|")
        for target, row in wide.loc[table["target"].unique()].iterrows():
            lines.append(f"| {target} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    return "\n".join(lines) + "\n"


def more_rows(more: pd.DataFrame, predictions: pd.DataFrame, n_train: dict, **kw) -> pd.DataFrame:
    """Rivals fitted on more rows (`more`) vs TabPFN's own fit in `predictions`."""
    rows = []
    for (target, model), group in more.groupby(["target", "model"], sort=False):
        same = predictions[predictions["target"] == target]
        mine, before = (same[same["model"] == m] for m in ("tabpfn", model))
        rows.append({"target": target, "model": model, "n_train": n_train[target],
                     "mae": _errors(group, target).mean(),
                     "mae_10k": _errors(before, target).mean() if len(before) else np.nan,
                     "tabpfn": _errors(mine, target).mean(), **versus(mine, group, target, **kw)})
    return pd.DataFrame(rows)
