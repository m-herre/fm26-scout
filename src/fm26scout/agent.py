"""The LLM layer: football language -> a search spec, or a question back. Nothing else.

The model (DeepSeek, OpenAI-compatible) sees a glossary of what the engine can filter, rank and
predict, the conversation, the current spec and the current shortlist. It returns JSON: a short
reply, at most two questions with a recommended option, the readings of vague words, and a spec.
Code validates the spec by running it; the engine picks and ranks the players and the page shows
every number. The model never estimates a player.
"""

from __future__ import annotations

import json
import os
import re

from .schema import ATTRIBUTES, POSITIONS, SEASON_PER_90, TARGETS
from .scout import Scout

PROMPT = """You are the planner of an FM26 scouting app. You turn the manager's words into a search
spec for a deterministic engine, or ask what you must know. You never state or guess a player's
numbers: the app shows those. Reply with one JSON object:
{{"reply": "1-2 short sentences in plain football language",
  "questions": [{{"question": "...", "options": ["...", "..."], "recommended": 0}}],
  "reading": [["vague word", "what you took it to mean"]],
  "spec": {{...}} or null,
  "open_player": "player id or name" or null}}

Rules
- "reply" is never empty: always at least one sentence to the manager.
- Ask only if something essential is missing and no sensible default exists (at most 2
  questions, 2-4 options each, recommended first). With questions, spec is null.
- Otherwise build the spec at once and explain vague words in "reading" (e.g. "wonderkid") in
  plain football words, never column names.
- Follow-ups ("cheaper", "only left-footed", "younger") edit the current spec; keep the rest.
- There is no "my club" or squad in this app. For a replacement with no name given, search by
  the position now and say in the reply that naming the player lets you match his profile.
  With a name, use "like" with that name and his position.
- If asked about something the model cannot predict (list below), say so plainly; never rank on it.
- "open_player" opens a player's page, e.g. for "tell me about the second one" (use the id from
  the shortlist).
- Money: write full numbers ("20m" = 20000000, "500k" = 500000). The app shows prices in the
  save's own units, which are not yet matched to in-game euros; don't mention this unless asked.

Spec (all keys optional)
{{"filters": {{column: [min, max] (null = open) or [allowed values], "position": [codes]}},
  "chance": {{"target": name, "at_least": number, "min": 0-1}},
  "rank": {{column: weight}}, "risk": 0-1, "like": "name or id", "limit": 10,
  "explain": "potential_ability"}}
- position codes: {positions}. CM = MC, CB = DC, ST = STC, winger = AML/AMR,
  full-back = DL/DR/WBL/WBR, "position" means Accomplished or Natural there.
- columns: age, value (price; where the save has none it is TabPFN's estimate),
  value_estimated ([true] = no price in the save), free_agent ([true] = no club), height_cm,
  contract_years_left, wage, injuries, reputation (labels {reputation}), club_stars ("0.5".."5"),
  left_foot / right_foot (labels {feet}),
  visible attributes 1-20 ({attributes}), season_apps, season_minutes, season_rating,
  season_<stat>_p90 ({season}).
- predictions (estimates the engine may use): {predictions}. Use "pred_<name>" in filters/rank,
  and "chance" for "how likely X or more".
- derived: value_gap (> 0: worth more than his price), pa_vs_peers (estimated potential vs same
  age and position), similarity (with "like"), chance.
- the model cannot predict: {cannot}.
- risk penalises wide (uncertain) estimates: "safe" 0.5, "boom or bust" 0.

Defaults when the manager doesn't say
- wonderkid / next big thing: age up to 21, chance potential_ability 160+ at least 0.4,
  rank by chance.  prospect: age up to 21, rank by pred_potential_ability.
- world class: 170+. ready now / proven: rank by pred_current_ability.
- bargain / undervalued: value_gap 0.5 plus the quality that matters 0.5 (pred_current_ability
  for prime / ready now, pred_potential_ability for young players), never value_gap alone.
- replacement for X: like X, his position, and put similarity in the ranking (e.g. 0.4) next to
  what the manager wants most.
- rank weights sum to 1; limit 10.
"""


def glossary(scout: Scout) -> str:
    f = scout.frame
    return PROMPT.format(
        positions=" ".join(p.upper() for p in POSITIONS),
        reputation=", ".join(f["reputation"].cat.categories),
        feet=", ".join(f["left_foot"].cat.categories),
        attributes=", ".join(ATTRIBUTES),
        season=", ".join(SEASON_PER_90),
        predictions=", ".join(f"{t} ({scout.trust.loc[t, 'verdict']}, scale "
                              f"{TARGETS[t].low:g}-{TARGETS[t].high:g})" for t in scout.targets
                              if t != "market_value") + ", market_value (fair value)",
        cannot=", ".join(t for t in TARGETS if t not in scout.targets),
    )


NO_KEY = "No DeepSeek API key: add one in Settings (or DEEPSEEK_API_KEY in .env) to chat"


def _key() -> str | None:
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")


def _client():
    from openai import OpenAI

    key = _key()
    if not key:
        raise RuntimeError(NO_KEY)
    return OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))


def _ask(messages: list[dict], tries: int = 3) -> dict:
    """The model's JSON answer. DeepSeek's JSON mode now and then returns empty or trailing
    text, so read the first object and ask again if there is none."""
    for _ in range(tries):
        response = _client().chat.completions.create(
            model=os.environ.get("LLM_MODEL", "deepseek-flash"), messages=messages,
            response_format={"type": "json_object"}, temperature=0.2)
        text = response.choices[0].message.content or ""
        if "{" in text:
            try:
                return json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
            except json.JSONDecodeError:
                pass
    raise RuntimeError("The language model gave no usable answer; please try again")


def shortlist_facts(report: dict | None) -> str:
    if not report:
        return "none yet"
    return "\n".join(
        f"{i}. id {p['player_id']} {p['name']}, {p['age']}, {p['club']}, {p['positions']}"
        for i, p in enumerate(report["players"][:10], 1))


def turn(scout: Scout, history: list[dict], spec: dict | None, report: dict | None,
         by: str | None = None, ask=None) -> dict:
    """One chat turn. `history`: [{"role": "user"|"assistant", "content": str}, ...]; `by`: the
    page's Ranking switch (expected, ceiling or safe); `ask`: the planner (default the language
    model; a recorded example replays its answer)."""
    context = (f"Current spec: {json.dumps(spec) if spec else 'none'}\n"
               f"Current shortlist:\n{shortlist_facts(report)}")
    messages = [{"role": "system", "content": glossary(scout)},
                {"role": "system", "content": context}, *history[-12:]]
    for _ in range(2):  # one repair round if the engine rejects the spec
        answer = (ask or _ask)(messages)
        if "spec" not in answer and {"filters", "rank", "like", "chance"} & set(answer):
            answer = {"spec": answer}  # the model sometimes answers with the bare spec
        answer["reply"] = answer.get("reply") or answer.get("message") or ""
        new = answer.get("spec")
        idle = not (answer["reply"] or answer.get("questions") or answer.get("open_player"))
        if idle and (not new or new == spec):  # said nothing and changed nothing: ask again
            messages += [{"role": "assistant", "content": json.dumps(answer)},
                         {"role": "user", "content": "Your answer had no reply. Answer the "
                          "manager's last message with the JSON format."}]
            continue
        if answer.get("questions") or not new or new == spec:  # nothing new to search
            return {**answer, "spec": spec, "report": None}
        said = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        new, candidates = _resolve(scout, new, said)
        if candidates:
            return {"reply": f"I found several players called \"{new['like']}\". Which one?",
                    "questions": [{"question": "Which player?", "options": candidates,
                                   "recommended": 0}], "spec": None, "report": None}
        new["explain"] = main_target(scout, new)
        if by:
            new["by"] = by
        try:
            result = scout.search({**new, "explain": None})
        except (ValueError, KeyError) as error:
            messages += [{"role": "assistant", "content": json.dumps(answer)},
                         {"role": "user", "content": f"The engine rejected that spec: {error}. "
                                                     "Fix it and answer with JSON again."}]
            continue
        return {**answer, "spec": new, "report": result, "tip": _tip(result)}
    return {"reply": "Sorry, I couldn't turn that into a search. Could you put it differently?",
            "questions": [], "spec": spec, "report": None}


def recorded(examples: list[dict], history: list[dict], spec: dict | None,
             report: dict | None) -> dict | None:
    """The planner's recorded answer when a new chat opens with one of the page's examples
    (case and spacing aside), else None. Lets the demo run without a DeepSeek key."""
    if spec or (report or {}).get("players") or len(history) != 1:
        return None
    said = " ".join(str(history[0].get("content", "")).lower().split())
    return next((e["answer"] for e in examples
                 if " ".join(e["prompt"].lower().split()) == said), None)


def main_target(scout: Scout, spec: dict) -> str:
    """The prediction a search is really about: shown on every row and explained."""
    rank = spec.get("rank") or {}
    for column in sorted(rank, key=lambda c: -abs(rank[c])):
        if column.startswith("pred_") and column[5:] in scout.targets:
            return column[5:]
    if spec.get("chance"):
        return spec["chance"]["target"]
    filters = spec.get("filters") or {}
    for column in filters:
        if column.startswith("pred_") and column[5:] in scout.targets:
            return column[5:]
    if "value_gap" in rank:
        return "market_value"
    # ranked on visible values only: how good he is now, or for young players how good he can get
    age = filters.get("age")
    young = isinstance(age, list) and len(age) == 2 and age[1] is not None and age[1] <= 21
    return "potential_ability" if young or "current_ability" not in scout.targets \
        else "current_ability"


def _resolve(scout: Scout, spec: dict, said: str) -> tuple[dict, list[str]]:
    """Pin "like" to one player id. A name that fits several players is settled by a club the
    manager mentioned; otherwise the candidates come back as options to choose from."""
    like = spec.get("like")
    if like is None:
        return spec, []
    f = scout.frame
    hits = f[(f["player_id"].astype(str) == str(like)) | (f["name"] == like)]
    if hits.empty:
        hits = f[f["name"].str.contains(str(like), case=False, na=False, regex=False)]
    aliases = {"CM": "MC", "CB": "DC", "ST": "STC", "LB": "DL", "RB": "DR", "CDM": "DM"}
    codes = {aliases.get(w, w) for w in re.findall(r"\b[A-Z]{2,3}\b", said.upper())}
    words = [w for w in re.findall(r"\w+", said.lower())
             if len(w) > 3 and w not in str(like).lower()]
    for column, match in (("club", lambda club: any(w in club.lower() for w in words)),
                          ("natural_positions", lambda pos: bool(set(pos.split("|")) & codes))):
        narrowed = hits[hits[column].fillna("").map(match)]
        hits = narrowed if len(narrowed) else hits  # each hint narrows, none empties the list
    if len(hits) == 1:
        return {**spec, "like": int(hits["player_id"].iloc[0])}, []
    if hits.empty:
        return spec, []  # the engine reports "no player matches"
    best = "pred_current_ability" if "pred_current_ability" in hits else "pred_potential_ability"
    top = hits.sort_values(best, ascending=False).head(4)
    return spec, [f"{r['name']} · {r['age']:.0f} · {r['club']} (id {r['player_id']})"
                  for _, r in top.iterrows()]


def _tip(report: dict) -> str | None:
    """If nobody is left, say which step removed the last players."""
    if report["players"]:
        return None
    for (_, before), (step, after) in zip(report["funnel"], report["funnel"][1:], strict=False):
        if after == 0:
            return f"Nobody is left after \"{step}\" ({before:,} before it). Loosen that one?"
    return None
