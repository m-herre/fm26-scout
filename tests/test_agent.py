"""The agent's guard rails, with the language model replaced by canned answers."""

import pytest

from fm26scout import agent
from fm26scout.scout import Scout


@pytest.fixture
def scout(folder):
    return Scout(folder)


def run(monkeypatch, scout, answers, said="find me someone"):
    replies = iter(answers)
    monkeypatch.setattr(agent, "_ask", lambda messages: next(replies))
    return agent.turn(scout, [{"role": "user", "content": said}], None, None)


def test_bare_spec_and_main_target(monkeypatch, scout):
    out = run(monkeypatch, scout, [{"filters": {"age": [None, 21]},
                                    "rank": {"pred_current_ability": 1}}])
    assert out["report"] and out["spec"]["explain"] == "current_ability"
    assert out["reply"] == ""


@pytest.mark.parametrize("filters, shown", [({"position": ["DC"]}, "current_ability"),
                                             ({"age": [None, 20]}, "potential_ability")])
def test_visible_only_ranking_shows_current_or_potential(scout, filters, shown):
    spec = {"filters": filters, "rank": {"pace": 0.5, "strength": 0.5}}
    assert agent.main_target(scout, spec) == shown


def test_rejected_spec_is_repaired(monkeypatch, scout):
    bad = {"reply": "ok", "spec": {"rank": {"pred_injury_proneness": 1}}}
    good = {"reply": "ok", "spec": {"rank": {"pred_potential_ability": 1}}}
    out = run(monkeypatch, scout, [bad, good])
    assert out["spec"]["rank"] == {"pred_potential_ability": 1} and out["report"]


def test_ambiguous_name_becomes_options(monkeypatch, scout):
    name = scout.frame["name"].str.split().str[-1].value_counts().index[0]  # a common surname
    out = run(monkeypatch, scout, [{"reply": "ok", "spec": {"like": name}}])
    assert out["questions"] and out["report"] is None
    assert all("(id " in option for option in out["questions"][0]["options"])


def test_empty_result_names_the_step(monkeypatch, scout):
    out = run(monkeypatch, scout, [{"reply": "ok", "spec": {"filters": {"age": [None, 5]}}}])
    assert out["report"]["players"] == [] and "age up to 5" in out["tip"]


def test_json_is_read_even_with_trailing_text(monkeypatch):
    class Fake:
        def __init__(self, text):
            self.choices = [type("C", (), {"message": type("M", (), {"content": text})})]
    texts = iter(["", '  {"reply": "hi"} and more'])
    client = type("Client", (), {"chat": type("Chat", (), {"completions": type(
        "Comp", (), {"create": staticmethod(lambda **kw: Fake(next(texts)))})})})
    monkeypatch.setattr(agent, "_client", lambda: client)
    assert agent._ask([]) == {"reply": "hi"}
