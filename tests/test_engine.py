import json
import shutil

import numpy as np
import pandas as pd
import pytest

from fm26scout.predict import chance, folds, grid
from fm26scout.schema import HIDDEN
from fm26scout.scout import Scout
from fm26scout.settings import SAMPLE
from fm26scout.split import age_band, position_group


def test_engine_never_loads_hidden_values(folder):
    scout = Scout(folder)
    assert not set(HIDDEN) & set(scout.frame.columns)
    assert "injury_proneness" not in scout.targets and "potential_ability" in scout.targets
    with pytest.raises(ValueError, match="cannot predict"):
        scout.search({"rank": {"pred_injury_proneness": 1}})


def test_filters_and_ranking(folder):
    scout = Scout(folder)
    report = scout.search({"filters": {"age": [None, 21], "position": ["STC"],
                                       "reputation": ["Local", "Regional"]},
                           "rank": {"pred_potential_ability": 1}, "limit": 5})
    players = report["players"]
    assert [n for _, n in report["funnel"]] == sorted((n for _, n in report["funnel"]),
                                                      reverse=True)
    assert all(p["age"] <= 21 for p in players)
    estimates = [p["predictions"]["potential_ability"]["estimate"] for p in players]
    assert estimates == sorted(estimates, reverse=True)


@pytest.mark.parametrize("by, column", [("ceiling", "hi_potential_ability"),
                                        ("safe", "lo_potential_ability")])
def test_ranking_by_top_or_bottom_of_range(folder, by, column):
    scout = Scout(folder)
    report = scout.search({"filters": {"age": [None, 21]}, "limit": 5, "by": by,
                           "rank": {"pred_potential_ability": 1}, "risk": 0.5})
    ends = [scout.frame.loc[scout.row(p["player_id"]), column] for p in report["players"]]
    assert ends == sorted(ends, reverse=True)


def test_folds_cover_everyone_once():
    fold = folds(pd.DataFrame(index=range(1003)))
    assert np.bincount(fold).tolist() == [201, 201, 201, 200, 200]


def test_chance_from_the_full_distribution_or_the_three_points():
    x = np.array([140.0, 150, 160, 170])
    cdf = np.array([[0.1, 0.3, 0.7, 0.9], [0.0, 0.0, 0.2, 1.0]])
    assert np.allclose(chance(None, None, None, 155, (x, cdf)), [0.5, 0.9])
    assert np.allclose(chance(None, None, None, 200, (x, cdf)), [0.1, 0.0])
    # no CDF kept (older saves): linear between the 10/50/90% points
    assert np.allclose(chance([140], [150], [160], 150), [0.5])
    assert np.allclose(chance([140], [150], [160], 155), [0.3])


def test_scout_reads_the_kept_distribution(folder, tmp_path):
    save = tmp_path / "save"
    shutil.copytree(folder, save)
    trust = pd.read_csv(save / "trust.csv")
    trust.loc[trust["target"] == "potential_ability", "range_shown"] = True  # curve only then
    trust.to_csv(save / "trust.csv", index=False)
    old = Scout(save)
    spec = {"chance": {"target": "potential_ability", "at_least": 160, "min": 0.5},
            "rank": {"chance": 1}, "limit": 3}
    assert old.distribution("potential_ability") is None  # fallback: the 10/50/90% points
    assert "distribution" not in old.details(0)["predictions"]["potential_ability"]
    # every player sure to reach 160 except the first, sure to stay below
    x = grid("potential_ability")
    cdf = np.tile((x >= 165).astype(np.float16), (len(old.frame), 1))
    cdf[0] = (x >= 100).astype(np.float16)
    (save / "db").mkdir()
    np.save(save / "db" / "potential_ability_grid.npy", x)
    np.save(save / "db" / "potential_ability_cdf.npy", cdf[::-1])
    scout = Scout(save)
    report = scout.search(spec)
    last = scout.frame.loc[len(scout.frame) - 1, "player_id"]
    assert report["funnel"][-1][1] == len(scout.frame) - 1
    assert all(p["chance"] == 1 for p in report["players"])
    assert last not in [p["player_id"] for p in scout.search(spec | {"limit": 5000})["players"]]
    curve = scout.details(0)["predictions"]["potential_ability"]["distribution"]
    assert curve["x"][0] < 165 <= curve["x"][-1] and curve["cdf"][0] == 0 and curve["cdf"][-1] == 1


def test_explain_is_shapley_vs_peers_in_one_call(monkeypatch):
    """Prior Labs' explainer on a known model: the effects are the true Shapley values (for a
    linear model, weight x distance to the peer value), they add up to estimate minus peer
    estimate, and every coalition of every player goes to the model in one call."""
    from fm26scout import analytics
    from fm26scout.models import TabPFN
    from fm26scout.schema import CATEGORICAL

    calls, tiers = [], []

    class Linear(TabPFN):  # 3 x passing + age, no fit
        def __init__(self, train, y, seed=0, local=None, tier="default", cache=False):
            tiers.append(tier)
            self.columns, self.local = list(train.columns), True
            self.levels = {c: pd.Index(train[c].dropna().unique())
                           for c in self.columns if c in CATEGORICAL}

        def predict_encoded(self, rows):
            calls.append(len(rows))
            return 3 * rows[:, self.columns.index("passing")] + rows[:, self.columns.index("age")]

    monkeypatch.setattr(analytics, "TabPFN", Linear)
    players = pd.read_parquet(SAMPLE).reset_index(drop=True)
    rows = [0, 1, 2]
    effects = analytics.explain(players, rows, "potential_ability", context=len(players),
                                budget=512, tier="thinking")  # more context than players
    assert len(calls) == 1 and tiers == ["default"]  # Thinking is explained by the default
    for row in rows:
        e = dict(effects[row])
        peers = players[(age_band(players) == age_band(players)[row])
                        & (position_group(players) == position_group(players)[row])]
        gap_passing = players.at[row, "passing"] - analytics._typical(peers, "passing")
        gap_age = players.at[row, "age"] - analytics._typical(peers, "age")
        assert e["passing"] == pytest.approx(3 * gap_passing, abs=0.05)
        assert e["age"] == pytest.approx(gap_age, abs=0.05)
        assert sum(e.values()) == pytest.approx(3 * gap_passing + gap_age, abs=0.05)
        assert set(analytics.GROUPS) >= set(e)


def test_why_lines_use_the_saves_model(monkeypatch, folder, tmp_path):
    from fm26scout import scout

    seen = []
    monkeypatch.setattr(scout, "explain", lambda players, rows, target, tier: seen.append(
        tier) or {r: [("pace", 1.0)] for r in rows})
    shutil.copytree(folder, tmp_path / "save")
    s = Scout(tmp_path / "save")
    s.model = "fast"
    assert s.explain([0], "potential_ability") == {0: [["pace", 1.0]]} and seen == ["fast"]


def test_why_lines_are_kept_per_save_and_model(monkeypatch, folder, tmp_path):
    """Kept players cost no TabPFN call (no token needed); only the missing ones are asked,
    in one call; Thinking shares the default model's file, as it is explained by it."""
    from fm26scout import scout

    shutil.copytree(folder, tmp_path / "save")
    asked = []

    def fake(players, rows, target, tier):
        asked.append(list(rows))
        return {r: [("pace", r + 0.123), ("market_value", -1.0)] for r in rows}

    monkeypatch.setattr(scout, "explain", fake)
    s = Scout(tmp_path / "save")
    s.model = "thinking"
    first = s.explain([0, 1], "potential_ability")
    assert asked == [[0, 1]] and first[1] == [["pace", 1.12], ["market value", -1.0]]
    kept = json.loads((tmp_path / "save/why/default_potential_ability.json").read_text())
    assert kept[str(s.frame.loc[1, "player_id"])] == first[1]
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    assert s.explain([1, 0], "potential_ability") == first and len(asked) == 1  # all kept
    assert s.explain([2, 0], "potential_ability")[0] == first[0] and asked[-1] == [2]
    s.model = "fast"  # another model's estimates: explained afresh
    s.explain([0], "potential_ability")
    assert asked[-1] == [0] and (tmp_path / "save/why/fast_potential_ability.json").exists()


@pytest.mark.parametrize("column, rule, text", [
    ("value", [500_000, 20_000_000], "value 500k to 20M"),
    ("value", [1_500_000, None], "value 1.5M+"),
    ("wage", [None, 900], "wage up to 900"),
    ("value_gap", [np.log(2), None], "worth 2.0× his price"),
    ("age", [None, 21], "age up to 21"),
])
def test_filters_in_plain_words(column, rule, text):
    from fm26scout.scout import _label

    assert _label(column, rule) == text
