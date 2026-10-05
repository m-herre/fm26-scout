import numpy as np
import pandas as pd
import pytest

from fm26scout.benchmark import compare, markdown, metrics, more_rows, paired_ci, verdict
from fm26scout.predict import chance_reliability, ranking, trust_markdown


def _predictions(n=400, seed=0):
    """Synthetic test predictions: TabPFN clearly better than CatBoost, ridge as good as the
    median guess, market value on a price scale."""
    rng = np.random.default_rng(seed)
    rows = []
    for target, truth in (("potential_ability", rng.uniform(80, 180, n)),
                          ("market_value", np.exp(rng.uniform(8, 16, n)))):
        noise = {"tabpfn": 0.05, "catboost": 0.2, "ridge": 0.4, "median": 0.4}
        for model, scale in noise.items():
            pred = truth * np.exp(rng.normal(0, scale, n))
            rows.append(pd.DataFrame({"target": target, "model": model, "player_id": range(n),
                                      "truth": truth, "pred": pred}))
    return pd.concat(rows, ignore_index=True)


def test_verdict_reads_the_whole_interval():
    assert verdict(-0.3, -0.01) == "better"
    assert verdict(-0.3, 0.01) == "tied"
    assert verdict(0.01, 0.3) == "worse"


def test_paired_ci_separates_a_real_gap_from_noise():
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 5, 2000)
    diff, lo, hi = paired_ci(base, base + 0.5)
    assert diff == pytest.approx(-0.5) and lo == pytest.approx(-0.5) and hi == pytest.approx(-0.5)
    a, b = base + rng.normal(0, 1, 2000), base + rng.normal(0, 1, 2000)
    diff, lo, hi = paired_ci(np.abs(a), np.abs(b))
    assert lo < diff < hi and verdict(lo, hi) == "tied"
    assert paired_ci(np.abs(a), np.abs(b)) == (diff, lo, hi)  # fixed seed


def test_compare_picks_the_best_rival_and_flags_the_median():
    preds = _predictions()
    vs = compare(preds).set_index("target")
    assert (vs["best_other"] == "catboost").all() and (vs["verdict"] == "better").all()
    assert (vs["median_verdict"] == "better").all()
    mae = metrics(preds).set_index(["target", "model"])["mae"]
    gap = mae[("market_value", "tabpfn")] - mae[("market_value", "catboost")]
    assert vs.loc["market_value", "delta"] == pytest.approx(gap)  # log scale, as metrics()
    swapped = preds.replace({"model": {"tabpfn": "median", "median": "tabpfn"}})
    vs = compare(swapped).set_index("target")
    assert (vs["verdict"] == "worse").all() and (vs["median_verdict"] == "worse").all()


def test_report_carries_the_ci_and_the_more_rows_section():
    preds = _predictions()
    more = preds[preds["model"] == "catboost"].assign(pred=lambda f: f["truth"])  # perfect
    extra = more_rows(more, preds, {"potential_ability": 40_000, "market_value": 21_000})
    assert (extra["verdict"] == "worse").all() and (extra["mae"] == 0).all()
    table = metrics(preds)
    by_slice = pd.DataFrame(columns=["model"])
    text = markdown(table, by_slice, {"data": "x"}, compare(preds), extra, {"rows asked": 1})
    assert "TabPFN vs best other (95% CI)" in text and "| better |" in text
    assert "## Baselines with 4× the rows" in text and "| 40,000 |" in text
    assert "⚠ vs median" not in text  # no target loses to the median guess
    assert "| – | – |" in markdown(table, by_slice, {"data": "x"})  # no CI asked: no claim


def test_ranking_on_a_tiny_example():
    r = ranking([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0], top=(1, 2, 3))
    assert r["base_rate"] == 0.5 and r["auc"] == 0.75
    assert r["ap"] == pytest.approx((1 + 2 / 3) / 2)  # hits at ranks 1 and 3
    assert (r["top1"], r["top2"], r["top3"]) == (1, 1, 2)
    assert ranking([0.2, 0.1], [0, 0])["ap"] is None


def test_chance_reliability_reports_the_young_separately():
    players = pd.DataFrame({"potential_ability": [170, 120, 165, 100, 155, 90.0],
                            "age": [19, 20, 30, 18, 21, 33]})
    db = pd.DataFrame({"lo_potential_ability": players["potential_ability"] - 10,
                       "pred_potential_ability": [150, 140, 175, 100, 150, 95.0],
                       "hi_potential_ability": players["potential_ability"] + 10})
    rel = chance_reliability(db, players, "potential_ability", 160)
    young = rel.attrs["ranking"]["age 21 or under"]
    assert rel.attrs["ranking"]["all players"]["n"] == 6 and young["n"] == 4
    assert young["base_rate"] == 0.25
    text = trust_markdown(pd.DataFrame(columns=["target"]), {"PA 160+": rel})
    assert "| age 21 or under | 4 | 25.0% |" in text and "of 4 | 1 of 4 |" in text
