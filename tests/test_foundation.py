import numpy as np
import pandas as pd
import pytest

from fm26scout.benchmark import metrics, predict_all
from fm26scout.schema import BY_NAME, COLUMNS, HIDDEN, TARGETS, VISIBLE, feature_columns
from fm26scout.settings import SAMPLE
from fm26scout.split import age_band, assign


@pytest.fixture(scope="module")
def players():
    return pd.read_parquet(SAMPLE)


def test_sample_matches_the_registry(players):
    assert list(players.columns) == [c.name for c in COLUMNS]
    assert players["name"].is_unique and players["player_id"].is_unique


def test_features_are_visible_only():
    for name in TARGETS:
        features = feature_columns(name)
        assert name not in features
        assert all(BY_NAME[f].visibility == "visible" for f in features)
    assert set(HIDDEN) <= set(TARGETS) and "market_value" in VISIBLE


def test_split_is_fixed_and_stratified(players):
    again = assign(players.drop(columns="split"))
    assert again.set_index("player_id")["split"].equals(
        players.set_index("player_id")["split"].loc[again["player_id"]])
    share = (players["split"] == "test").groupby(age_band(players)).mean()
    assert share.between(0.18, 0.22).all()


def test_held_out_targets_never_reach_a_model(players):
    small = players.sample(3000, random_state=1)
    run = lambda frame: predict_all(frame, ["potential_ability", "market_value"],  # noqa: E731
                                    ["median", "ridge"], n_train=1500, log=lambda *_: None)
    before = run(small)
    corrupted = small.copy()
    test = corrupted["split"] == "test"
    corrupted.loc[test, HIDDEN] = np.random.default_rng(0).uniform(1, 20, (test.sum(), len(HIDDEN)))
    after = run(corrupted)
    assert np.allclose(before["pred"], after["pred"])
    assert (metrics(before).query("model == 'ridge'")["skill"] > 0.3).all()
