"""FM26 save -> canonical player table (one row per player, columns exactly as in schema.py)."""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import pandas as pd

from .schema import (
    ATTRIBUTES,
    COLUMNS,
    FAMILIARITY,
    FOOT,
    HIDDEN_ATTRIBUTES,
    PERSONALITY,
    POSITIONS,
    REPUTATION,
    SEASON_PER_90,
    STARS,
    TARGETS,
    TRAIT_CODES,
)


def read_save(path: Path, allow_warnings: bool = False) -> tuple[dict[str, pd.DataFrame], dict]:
    """fmsave's player, season-stat and injury frames plus metadata. Refuses builds fmsave
    can't vouch for."""
    import fmsave

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with fmsave.open(path, strict=False) as save:
            info = save.info
            raw = {"players": save.players().to_pandas(),
                   "season": save.player_season_stats().to_pandas(),
                   "injuries": save.injuries().to_pandas()}
    problems = [
        str(w.message)
        for w in caught
        if issubclass(w.category, (fmsave.UnknownBuildWarning, fmsave.ReaderCheckWarning))
    ]
    if problems and not allow_warnings:
        raise RuntimeError("fmsave cannot vouch for this save:\n" + "\n".join(problems))
    meta = {"game": info.game, "build": info.build, "save_date": info.game_date.isoformat()}
    return raw, meta


def _years(dates: pd.Series, since: pd.Timestamp) -> pd.Series:
    return (pd.to_datetime(dates, errors="coerce") - since).dt.days / 365.25


def _band(values: pd.Series, bands: tuple[list, list[str]]) -> pd.Series:
    """The in-game label for a stored number, as an ordered category."""
    edges, labels = bands
    return pd.cut(pd.to_numeric(values, errors="coerce").astype(float), [-float("inf"), *edges],
                  labels=labels, ordered=True)


def _season(stats: pd.DataFrame) -> pd.DataFrame:
    """This season, all competitions, summed over the teams a player played for."""
    rows = stats[stats["kind"] == "overall"].assign(
        saves=lambda f: f[["saves_held", "saves_parried", "saves_tipped"]].sum(axis=1),
        rating_x_apps=lambda f: f["average_rating"] * f["rated_appearances"],
    )
    sums = rows.groupby("player_uid").sum(numeric_only=True)
    minutes = sums["minutes"].where(sums["minutes"] >= 270)  # per 90 is noise below 3 games
    return pd.DataFrame({
        "season_apps": sums["starts"] + sums["substitute_appearances"],
        "season_minutes": sums["minutes"],
        "season_rating": sums["rating_x_apps"] / sums["rated_appearances"].where(
            sums["rated_appearances"] > 0),
        "season_potm": sums["player_of_the_match"],
        "season_pass_pct": sums["passes_completed"] / sums["passes_attempted"].where(
            sums["passes_attempted"] > 0),
        **{f"season_{s}_p90": sums[s] / minutes * 90 for s in SEASON_PER_90},
    })


def _injuries(records: pd.DataFrame) -> pd.DataFrame:
    """Injuries in the save's history (about the last two seasons)."""
    history = records[records["kind"] == "history"]
    by = history.groupby("player_uid")
    return pd.DataFrame({
        "injuries": by.size(),
        "injuries_major": by["severity"].apply(lambda s: (s == "major").sum()),
        "injuries_in_match": by["cause"].apply(lambda s: (s == "in_match").sum()),
    })


def _member(values: pd.Series, groups: pd.Series) -> list[float]:
    return [float(pd.notna(v) and v in g) for v, g in zip(values, groups, strict=True)]


def _release_clause(clauses) -> float:
    fees = [c["value"] for c in clauses or () if c["kind"] == "minimum_fee_release"]
    return float(max(fees)) if fees else float("nan")


def canonical(raw: dict[str, pd.DataFrame], save_date: str) -> pd.DataFrame:
    """Map fmsave's frames onto the registry. Unregistered columns are dropped."""
    today = pd.Timestamp(save_date)
    players = raw["players"]
    uid = players["uid"].astype("int64")
    season = _season(raw["season"]).reindex(uid).reset_index(drop=True)
    season[["season_apps", "season_minutes", "season_potm"]] = season[
        ["season_apps", "season_minutes", "season_potm"]].fillna(0)  # did not play
    injuries = _injuries(raw["injuries"]).reindex(uid).fillna(0).reset_index(drop=True)
    raw = players
    out = pd.DataFrame(
        {
            "player_id": raw["uid"].astype("int64"),
            "name": raw["name"],
            "club": raw["club_name"],
            "split": "unassigned",
            "age": raw["age"],
            "height_cm": raw["height_cm"],
            "left_foot": _band(raw["left_foot"], FOOT),
            "right_foot": _band(raw["right_foot"], FOOT),
            "nationalities": 1 + raw["second_nation_ids"].map(len),
            "reputation": _band(raw["reputation_current"], REPUTATION),
            "club_stars": _band(raw["club_reputation"], STARS),
            "club_league_position": raw["club_last_league_position"],
            "market_value": raw["transfer_value"].where(raw["transfer_value_state"] == "ok"),
            "wage": raw["contract_wage"],
            "contract_years_left": _years(raw["contract_end"], today),
            "years_at_club": -_years(raw["club_join_date"], today),
            "on_loan": raw["on_loan"].astype(float),
            "nation_id": raw["nation_id"],
            "club_id": raw["club_uid"],
            "club_nation_id": raw["club_nation_id"],
            "squad_status": raw["contract_squad_status"],
            "contract_kind": raw["contract_kind"],
            "natural_positions": raw["natural_positions"].map(
                lambda p: "|".join(sorted(p)) or None),
            "home_grown_club": _member(raw["club_uid"], raw["home_grown_club_uids"]),
            "home_grown_nation": _member(raw["club_nation_id"], raw["home_grown_nation_ids"]),
            "release_clause": raw["contract_clauses"].map(_release_clause),
            **season,
            **injuries,
            **{a: raw[f"attributes_{a}"] for a in ATTRIBUTES},
            **{f"pos_{p}": _band(raw[f"positions_{p}"], FAMILIARITY) for p in POSITIONS},
            **{
                f"trait_{c}": raw["traits_code"].map(lambda codes, c=c: float(c in codes))
                for c in TRAIT_CODES
            },
            "potential_ability": raw["ability_potential"],
            "current_ability": raw["ability_current"],
            **{a: raw[f"attributes_{a}"] for a in HIDDEN_ATTRIBUTES},
            **{p: raw[f"personality_{p}"] for p in PERSONALITY},
        }
    )
    for name, target in TARGETS.items():  # values outside the scale are unreadable, not real
        value = pd.to_numeric(out[name], errors="coerce").astype(float)
        out[name] = value.where(value.between(target.low, target.high))
    assert list(out.columns) == [c.name for c in COLUMNS]
    below = (out["potential_ability"] < out["current_ability"]).mean()
    if below > 0.01:
        raise RuntimeError(f"PA looks misread: {below:.1%} of players have PA below CA")
    return out


def anonymise(players: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Replace player names with unique made-up ones (for the public sample)."""
    first = "Adrian Bruno Carlos Diego Elias Emil Enzo Felix Hugo Ivan Jonas Kai Leon Luca Marco "
    first += "Mateo Milan Nico Noah Oscar Pablo Rafael Sami Theo Tomas Viktor Yusuf Zane Omar Rui"
    last = "Albrecht Barros Castell Dvorak Eklund Ferreira Gallo Haugen Ibarra Jansen Kovac Moreau "
    last += "Novak Ortega Petrov Quinn Rossi Silva Torres Vidal Weber Yilmaz Costa Duarte Keller"
    rng = random.Random(seed)
    initials = "ABCDEFGHJKLMNPRSTVW"
    names: set[str] = set()
    while len(names) < len(players):
        a, b = rng.choice(initials), rng.choice(initials)
        names.add(f"{rng.choice(first.split())} {a}.{b}. {rng.choice(last.split())}")
    out = players.copy()
    out["name"] = rng.sample(sorted(names), len(names))
    return out
