"""Benchmark: the why line's Shapley values against a high-budget reference.

One hosted TabPFN-3.5 fit (potential ability, 10,000 context players, as analytics.explain)
explains a 20-player shortlist three ways:
  shap_peer   official get_tabpfn_imputation_explainer, baseline = the peer-typical row,
              Shapley value per feature, summed per group (what analytics.explain uses)
  shap_inf    official get_tabpfn_inf_explainer (masked feature = missing), summed per group
  reference   grouped Shapley values (peer baseline) with a large OddSHAP budget, two seeds
Metrics: agreement with the reference, deletion test (swap the 4 named groups, how far does
the estimate move), completeness, predicted rows, seconds.

Runs on the Prior Labs API (TABPFN_TOKEN). The price is per call (~10k tokens for up to ~8k
rows), so each player is explained twice: a record pass collects every row the explainers ask
for, one API call answers them all, a replay pass reads the answers (coalition sampling is
seeded, so it asks for the same rows; a miss costs one more call and is counted).
Summary -> reports/<save>/shap.json. Needs the dev extra (shapiq).
    TABPFN_DISABLE_TELEMETRY=1 python scripts/bench_shap.py [--save sample] [--players 20]
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fm26scout import models
from fm26scout.analytics import GROUPS, _typical
from fm26scout.schema import feature_columns
from fm26scout.scout import NOT_REASONS
from fm26scout.settings import ROOT, read_env
from fm26scout.split import age_band, position_group

TARGET = "potential_ability"


def why(effects: dict[str, float], k: int = 4) -> list[str]:
    """The why line: biggest football reasons (as scout.py shows them)."""
    ranked = sorted(effects, key=lambda g: -abs(effects[g]))
    return [g for g in ranked if g not in NOT_REASONS][:k]


class Oracle:
    """Stands in for reg.predict: record rows, answer them with one real call, replay."""

    def __init__(self, real, columns: list[str]):
        self.real, self.columns, self.cache, self.asked = real, columns, {}, []
        self.recording, self.calls, self.rows, self.seconds = True, 0, 0, 0.0

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.recording:
            self.asked.extend(x)
            return np.zeros(len(x))
        keys = [r.tobytes() for r in x]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        if missing:
            self.fetch(x[missing])
        return np.array([self.cache[k] for k in keys])

    def fetch(self, x: np.ndarray) -> None:
        x = np.unique(np.asarray(x, dtype=float), axis=0)
        x = x[[r.tobytes() not in self.cache for r in x]]
        if not len(x):
            return
        t = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = self.real(pd.DataFrame(x, columns=self.columns))
        self.seconds += time.perf_counter() - t
        self.calls, self.rows = self.calls + 1, self.rows + len(x)
        self.cache.update({r.tobytes(): float(v) for r, v in zip(x, y, strict=True)})

    def run(self, work):
        """work() twice: record, one call, replay."""
        self.recording, self.asked = True, []
        work()
        self.fetch(np.array(self.asked))
        self.recording = False
        return work()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="sample", help="sample, or a prepared folder in data/")
    ap.add_argument("--players", type=int, default=20)
    ap.add_argument("--budget", type=int, default=2048, help="coalitions per player, SHAP")
    ap.add_argument("--ref-budget", type=int, default=8192, help="coalitions per player, ref")
    ap.add_argument("--context", type=int, default=10_000)
    args = ap.parse_args()
    for name, value in read_env().items():  # the project's .env; real environment wins
        os.environ.setdefault(name, value)

    import shapiq
    from tabpfn_extensions.interpretability import shapiq as tpe

    folder = ROOT / "sample" if args.save == "sample" else ROOT / "data" / args.save
    players = pd.read_parquet(folder / "players.parquet")
    pred = pd.read_parquet(folder / "predictions.parquet").set_index("player_id")
    features = feature_columns(TARGET)

    # shortlist: half "wonderkid" search (age <= 21, top predicted PA), half random
    rng = np.random.default_rng(0)
    p_pa = players["player_id"].map(pred[f"pred_{TARGET}"])
    young = players[(players["age"] <= 21)].assign(p=p_pa).nlargest(args.players // 2, "p")
    rest = players.drop(index=young.index)
    rows = [*young.index, *rng.choice(rest.index, args.players - len(young), replace=False)]

    others = players.drop(index=rows)
    ctx = others[others[TARGET].notna()].sample(args.context, random_state=0)

    # same hosted regressor as the app, plus the +inf passthrough the missingness SHAP needs
    base_regressor = models._regressor
    models._regressor = lambda *a, **k: base_regressor(*a, **k).set_params(
        inference_config={"PASSTHROUGH_INF": True})
    t = time.perf_counter()
    model = models.TabPFN(ctx[features], ctx[TARGET], seed=0, local=False)
    fit_s = time.perf_counter() - t
    reg = model.model
    oracle = Oracle(reg.predict, features)
    reg.predict = oracle  # shapiq calls model.predict(X)

    groups = {g: [c for c in cs if c in features] for g, cs in GROUPS.items()}
    names = [g for g, cs in groups.items() if cs]
    col = {c: i for i, c in enumerate(features)}
    member = {g: [col[c] for c in groups[g]] for g in names}
    feature_group = {i: g for g, idx in member.items() for i in idx}

    peers = age_band(players) + "|" + position_group(players)
    dtypes = players[features].dtypes.to_dict()
    dest = ROOT / "reports" / "bench_shap"
    dest.mkdir(parents=True, exist_ok=True)
    partial = dest / f"partial-{args.save}-{args.players}-{args.budget}-{args.ref_budget}.jsonl"
    results = [json.loads(line) for line in partial.read_text().splitlines()] \
        if partial.exists() else []
    done = {r["row"] for r in results}

    def explain_player(row):
        raw = players.loc[[row], features]
        group = players[peers == peers[row]]
        typical = raw.copy()
        for c in features:
            typical[c] = _typical(group, c)
        x, b = (model._numeric(f.astype(dtypes)).to_numpy(float)[0] for f in (raw, typical))
        f_x, f_b = oracle(np.vstack([x, b]))
        out = {"row": int(row), "player_id": int(players.at[row, "player_id"]),
               "age": int(players.at[row, "age"]), "estimate": float(f_x),
               "peer_estimate": float(f_b), "x": x.tolist(), "b": b.tolist()}

        def per_group(values: np.ndarray) -> dict[str, float]:
            agg = dict.fromkeys(names, 0.0)
            for i, v in enumerate(values):
                agg[feature_group[i]] += float(v)
            return agg

        # official imputation explainer, baseline = the peer row
        t = time.perf_counter()
        ex = tpe.get_tabpfn_imputation_explainer(reg, b[None, :], index="SV", max_order=1,
                                                 imputer="baseline", random_state=0)
        iv = ex.explain(x, budget=args.budget)
        out["shap_peer"] = {"effects": per_group(iv.get_n_order_values(1)),
                            "seconds": time.perf_counter() - t, "rows": args.budget}

        # official missingness explainer (masked feature -> +inf -> "missing")
        t = time.perf_counter()
        ex = tpe.get_tabpfn_inf_explainer(reg, ctx[features].head(1).pipe(model._numeric)
                                          .to_numpy(float), random_state=0)
        iv = ex.explain(x, budget=args.budget)
        out["shap_inf"] = {"effects": per_group(iv.get_n_order_values(1)),
                           "seconds": time.perf_counter() - t, "rows": args.budget,
                           "base": float(iv.baseline_value)}

        # grouped Shapley values, peer baseline, big budget, two seeds (noise check)
        class Grouped(shapiq.Game):
            def __init__(self):
                super().__init__(n_players=len(names), normalize=True,
                                 normalization_value=float(f_b), verbose=False)

            def value_function(self, coalitions: np.ndarray) -> np.ndarray:
                xs = np.tile(b, (len(coalitions), 1))
                for j, g in enumerate(names):
                    on = coalitions[:, j]
                    xs[np.ix_(on, member[g])] = x[member[g]]
                return oracle(xs)

        for seed in (0, 1):
            t = time.perf_counter()
            approx = shapiq.OddSHAP(n=len(names), random_state=seed)
            iv = approx.approximate(budget=args.ref_budget, game=Grouped())
            vals = iv.get_n_order_values(1)
            out[f"reference_{seed}"] = {"effects": dict(zip(names, map(float, vals), strict=True)),
                                        "seconds": time.perf_counter() - t,
                                        "rows": args.ref_budget}
        return out

    for n, row in enumerate(rows, 1):
        if int(row) in done:
            continue
        calls, rows_sent, secs = oracle.calls, oracle.rows, oracle.seconds
        out = oracle.run(lambda row=row: explain_player(row))
        out["api"] = {"calls": oracle.calls - calls, "rows": oracle.rows - rows_sent,
                      "seconds": round(oracle.seconds - secs, 1)}
        results.append(out)
        with partial.open("a") as fh:
            fh.write(json.dumps(out) + "\n")
        print(f"[{n}/{len(rows)}] player {out['player_id']} age {out['age']} "
              f"PA {out['estimate']:.0f} (peers {out['peer_estimate']:.0f})  "
              f"api {out['api']}", flush=True)

    # deletion test for every player in one call: swap each method's 4 why-line groups
    def deletion():
        for out in results:
            x, b = np.array(out["x"]), np.array(out["b"])
            for m in ("shap_peer", "shap_inf", "reference_0"):
                named = why(out[m]["effects"])
                s = x.copy()
                for g in named:
                    s[member[g]] = b[member[g]]
                out[m]["why"] = named
                out[m]["deletion_move"] = float(abs(out["estimate"] - oracle(s[None, :])[0]))

    oracle.run(deletion)
    api = {"calls": oracle.calls, "rows": oracle.rows, "seconds": round(oracle.seconds, 1)}

    report = summarise(results, names, fit_s, args) | {"api": api}
    (dest / f"results-{args.save}.json").write_text(
        json.dumps({"summary": report, "players": results}, indent=2))
    summary = ROOT / "reports" / folder.name / "shap.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"summary -> {summary}")


def summarise(results: list[dict], names: list[str], fit_s: float, args) -> dict:
    def agree(a: dict, r: dict) -> dict:
        wa, wr = set(why(a)), set(why(r))
        common = [g for g in names if g in a]
        top = why(r)
        return {"why_overlap_of_4": len(wa & wr),
                "same_top_reason": why(a)[:1] == top[:1],
                "spearman": spearmanr([a[g] for g in common], [r[g] for g in common])[0],
                "sign_agree_ref_top4": np.mean([np.sign(a.get(g, 0)) == np.sign(r[g])
                                                for g in top])}

    out = {"save": args.save, "target": TARGET, "players": len(results),
           "fit_seconds": round(fit_s, 1), "budget": args.budget,
           "ref_budget": args.ref_budget, "methods": {}}
    for m in ("shap_peer", "shap_inf", "reference_1"):
        rows = [agree(p[m]["effects"], p["reference_0"]["effects"]) for p in results]
        stats = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]}
        stats["seconds_per_player"] = round(float(np.mean([p[m]["seconds"]
                                                           for p in results])), 2)
        stats["rows_per_player"] = results[0][m]["rows"]
        if m != "reference_1":
            stats["deletion_move"] = round(float(np.mean([p[m]["deletion_move"]
                                                          for p in results])), 2)
        gap = [abs(sum(p[m]["effects"].values()) - (p["estimate"] - p["peer_estimate"]))
               for p in results]
        stats["completeness_error"] = round(float(np.mean(gap)), 2)
        out["methods"][m] = stats
    out["methods"]["reference_0"] = {"deletion_move": round(float(np.mean(
        [p["reference_0"]["deletion_move"] for p in results])), 2)}
    out["mean_gap_to_peers"] = round(float(np.mean([abs(p["estimate"] - p["peer_estimate"])
                                                    for p in results])), 2)
    return out


if __name__ == "__main__":
    main()
