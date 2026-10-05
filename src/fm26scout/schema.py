"""The column registry: every column of the canonical player table and who may see it.

visibility:
  visible  shown to a manager in the game; the only columns a model may use as inputs
  hidden   stored in the save but never shown in the game; prediction targets only
  meta     identifiers and bookkeeping; never a model input

Anything fmsave returns that is not registered here (for example the hidden 1-100
`raw_attributes_*`, condition, clause details) is dropped at extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Visibility = Literal["visible", "hidden", "meta"]
Kind = Literal["num", "cat", "ord", "text"]  # ord: in-game label with a natural order


@dataclass(frozen=True)
class Column:
    name: str
    kind: Kind
    visibility: Visibility
    note: str = ""


ATTRIBUTES = (
    "crossing dribbling finishing heading long_shots marking off_the_ball passing penalty_taking "
    "tackling vision handling aerial_reach command_of_area communication kicking throwing "
    "anticipation decisions one_on_ones positioning reflexes first_touch technique flair corners "
    "teamwork work_rate long_throws eccentricity rushing_out punching acceleration "
    "free_kick_taking strength stamina pace jumping_reach leadership balance bravery aggression "
    "agility natural_fitness determination composure concentration"
).split()
POSITIONS = "gk sw dl dc dr dm ml mc mr aml amc amr stc wbl wbr".split()
TRAIT_CODES = range(61)  # fmsave trait codes; many have no public name yet
SEASON_PER_90 = (
    "goals assists expected_goals expected_assists shots key_passes dribbles tackles_completed "
    "interceptions pressures_completed headers_won clearances fouls_made yellow_cards red_cards "
    "mistakes_leading_to_goal saves goals_allowed"
).split()
HIDDEN_ATTRIBUTES = "consistency important_matches injury_proneness dirtiness versatility".split()
PERSONALITY = (
    "professionalism ambition adaptability loyalty pressure temperament sportsmanship controversy"
).split()

COLUMNS: tuple[Column, ...] = (
    Column("player_id", "num", "meta"),
    Column("name", "text", "meta"),
    Column("club", "text", "meta", "club name; club_id is the model input"),
    Column("split", "text", "meta", "fixed train/test assignment"),
    Column("age", "num", "visible"),
    Column("height_cm", "num", "visible"),
    Column("left_foot", "ord", "visible", "in-game label; bands approximate"),
    Column("right_foot", "ord", "visible", "in-game label; bands approximate"),
    Column("nationalities", "num", "visible", "1 + second nationalities"),
    Column("reputation", "ord", "visible", "in-game label; bands approximate"),
    Column("club_stars", "ord", "visible", "0.5-5 stars; bands approximate"),
    Column("club_league_position", "num", "visible", "last league finish"),
    Column("market_value", "num", "visible", "missing unless fmsave state is 'ok'; also a target"),
    Column("wage", "num", "visible"),
    Column("contract_years_left", "num", "visible", "fmsave lacks the end date for ~40%"),
    Column("years_at_club", "num", "visible", "fmsave lacks the join date for ~40%"),
    Column("on_loan", "num", "visible"),
    Column("nation_id", "cat", "visible"),
    Column("club_id", "cat", "visible"),
    Column("club_nation_id", "cat", "visible", "country of the club's league"),
    Column("squad_status", "cat", "visible"),
    Column("contract_kind", "cat", "visible"),
    Column("natural_positions", "cat", "visible", "e.g. 'AMR|STC'"),
    Column("home_grown_club", "num", "visible", "trained at the current club"),
    Column("home_grown_nation", "num", "visible", "trained in the club's country"),
    Column("release_clause", "num", "visible", "minimum fee release clause, if any"),
    Column("season_apps", "num", "visible", "this season, all competitions"),
    Column("season_minutes", "num", "visible"),
    Column("season_rating", "num", "visible", "average match rating"),
    Column("season_potm", "num", "visible", "player of the match awards"),
    Column("season_pass_pct", "num", "visible"),
    *(Column(f"season_{s}_p90", "num", "visible", "per 90 minutes") for s in SEASON_PER_90),
    Column("injuries", "num", "visible", "injury history in the save (~2 seasons)"),
    Column("injuries_major", "num", "visible"),
    Column("injuries_in_match", "num", "visible"),
    *(Column(a, "num", "visible", "1-20") for a in ATTRIBUTES),
    *(Column(f"pos_{p}", "ord", "visible", "in-game familiarity label") for p in POSITIONS),
    *(Column(f"trait_{c}", "num", "visible", "player trait 0/1") for c in TRAIT_CODES),
    Column("potential_ability", "num", "hidden", "PA 1-200"),
    Column("current_ability", "num", "hidden", "CA 1-200"),
    *(Column(a, "num", "hidden", "1-20") for a in HIDDEN_ATTRIBUTES),
    *(Column(p, "num", "hidden", "1-20") for p in PERSONALITY),
)
BY_NAME = {c.name: c for c in COLUMNS}
VISIBLE = [c.name for c in COLUMNS if c.visibility == "visible"]
HIDDEN = [c.name for c in COLUMNS if c.visibility == "hidden"]
CATEGORICAL = {c.name for c in COLUMNS if c.kind == "cat"}
ORDINAL = {c.name for c in COLUMNS if c.kind == "ord"}

# What the game shows instead of the stored 1-20 or 0-10000 number: (upper bounds, labels).
# Position bands match fmsave's docs (natural 18-20, accomplished 15-17); the rest are
# approximations of the in-game labels, since the save does not store the thresholds.
FOOT = ([3, 7, 10, 13, 17, 20],
        ["Very Weak", "Weak", "Reasonable", "Fairly Strong", "Strong", "Very Strong"])
FAMILIARITY = ([1, 4, 9, 14, 17, 20],
               ["Ineffectual", "Awkward", "Unconvincing", "Competent", "Accomplished", "Natural"])
REPUTATION = ([999, 2499, 3999, 5999, 7999, 10000],
              ["Obscure", "Local", "Regional", "National", "Continental", "Worldwide"])
STARS = ([1000 * i for i in range(1, 11)], [f"{i / 2:g}" for i in range(1, 11)])


@dataclass(frozen=True)
class Target:
    name: str
    low: float
    high: float
    log: bool = False  # fit on log1p, report errors on both scales


TARGETS: dict[str, Target] = {
    t.name: t
    for t in (
        Target("potential_ability", 1, 200),
        Target("current_ability", 1, 200),
        *(Target(name, 1, 20) for name in HIDDEN_ATTRIBUTES + PERSONALITY),
        # Visible in the game, but missing for about half the save: supervised reconstruction.
        Target("market_value", 0, float("inf"), log=True),
    )
}


def feature_columns(target: str) -> list[str]:
    """Model inputs for `target`: every visible column except the target itself."""
    features = [name for name in VISIBLE if name != target]
    assert not set(features) & set(HIDDEN), "hidden column among features"
    return features
