"""Settings page: keys in .env, missing-key warnings, the web app's guards. No network."""

import json
import os
import re
import shutil
import stat
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from fm26scout import settings, web


@pytest.fixture
def env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("# mine\nOTHER=1\n")
    monkeypatch.setattr(settings, "ENV", path)
    for name in ("DEEPSEEK_API_KEY", "LLM_API_KEY", "TABPFN_TOKEN", "FM26_TABPFN",
                 "FM26_TABPFN_DEVICE", "FM26_TABPFN_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "check_key", lambda key, value: None)
    return path


def test_key_is_saved_masked_and_removed(env):
    settings.save_key("deepseek", "  sk-0123456789abcdef0123  ")
    assert "DEEPSEEK_API_KEY=sk-0123456789abcdef0123" in env.read_text()
    assert "# mine" in env.read_text() and "OTHER=1" in env.read_text()
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    status = settings.key_status()["deepseek"]
    assert status["set"] and status["source"] == ".env"
    assert "0123456789abcdef" not in status["masked"] and status["masked"].endswith("0123")
    settings.remove_key("deepseek")
    assert "DEEPSEEK" not in env.read_text() and "DEEPSEEK_API_KEY" not in os.environ


def test_shell_key_cannot_be_removed_here(env, monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "a-token-from-the-shell-123")
    assert settings.key_status()["tabpfn"]["source"] == "environment"
    with pytest.raises(ValueError, match="shell"):
        settings.remove_key("tabpfn")


def test_bad_looking_key_is_refused_before_any_call(monkeypatch):
    with pytest.raises(ValueError, match="doesn't look like"):
        settings.check_key("deepseek", "sk-has spaces in it 12345")


def test_missing_keys_are_problems(env):
    found = settings.problems()
    assert any("DeepSeek" in p for p in found) and any("TabPFN API token" in p for p in found)
    settings.save_key("deepseek", "sk-0123456789abcdef0123")
    settings.write_env("FM26_TABPFN", "local")
    assert settings.problems() == []


def test_env_writes_from_many_threads_all_land(env):
    threads = [threading.Thread(target=settings.write_env, args=(f"FM26_TEST_{i}", str(i)))
               for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stored = settings.read_env(env)
    assert all(stored[f"FM26_TEST_{i}"] == str(i) for i in range(20)) and stored["OTHER"] == "1"
    assert [p.name for p in env.parent.iterdir()] == [".env"]  # no temporary file left
    for i in range(20):
        os.environ.pop(f"FM26_TEST_{i}")


def test_cloud_without_token_fails_with_words(env):
    from fm26scout.models import _regressor

    with pytest.raises(RuntimeError, match="Settings"):
        _regressor([], 0, local=False)


@pytest.fixture
def app(env, tmp_path, monkeypatch, folder):
    data = tmp_path / "data"
    shutil.copytree(folder, data / "career-2076-07-01")
    monkeypatch.setattr(web, "DATA", data)
    monkeypatch.setattr(web, "REMEMBER", data / "web.json")
    monkeypatch.setattr(web, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(web, "CHATS", data / "chats")
    return web.App(None)


def test_demo_save_opens_when_nothing_is_remembered(app):
    assert app.active == "demo" and app.need().folder == web.DEMO
    app.open("career-2076-07-01")
    assert web.App(None).active == "career-2076-07-01"  # the user's pick wins
    assert web.App(web.DEMO).active == "demo"  # and --data wins over that


def test_app_without_save_says_so(app, tmp_path, monkeypatch):
    monkeypatch.setattr(web, "DEMO", tmp_path / "no-demo")
    web.REMEMBER.unlink()
    app = web.App(None)
    assert app.info()["save"] is None and app.settings()["problems"]
    with pytest.raises(RuntimeError, match="No save loaded"):
        app.need()
    with pytest.raises(ValueError, match=r"\.fm"):
        app.upload("players.csv", 10, None)
    with pytest.raises(RuntimeError, match="TabPFN"):  # no token, cloud: refused before upload
        app.upload("career.fm", 10, None)


def test_recent_chats_keep_the_last_five_per_save(app):
    app.open("career-2076-07-01")
    ids = [app.save_chat({"title": f"chat {i}", "turns": [{"text": str(i)}]})["id"]
           for i in range(7)]
    assert [c["title"] for c in app.chats()] == [f"chat {i}" for i in (6, 5, 4, 3, 2)]
    app.save_chat({"id": ids[3], "title": "chat 3", "turns": [{}, {}]})  # a follow-up
    assert app.chats()[0] == app.chats()[0] | {"id": ids[3], "turns": 2}
    assert app.chats(ids[3])["turns"] == [{}, {}]
    with pytest.raises(KeyError):
        app.chats(ids[0])  # dropped as the oldest
    app.open("demo")
    assert app.chats() == []  # chats belong to their save
    app.open("career-2076-07-01")
    assert len(app.delete_chat(ids[3])) == 4
    app.delete("career-2076-07-01")
    assert not (web.CHATS / "career-2076-07-01.json").exists()


def test_user_picks_the_save_and_can_delete_it(app):
    listed = {s["id"]: s for s in app.settings()["saves"]}
    assert set(listed) == {"demo", "career-2076-07-01"} and not listed["demo"]["deletable"]
    app.open("career-2076-07-01")
    assert app.info()["players"] == 2000 and web.REMEMBER.exists()
    assert web.App(None).active == "career-2076-07-01"  # remembered across restarts
    app.open("demo")
    assert app.active == "demo"  # exactly one save in use
    with pytest.raises(ValueError, match="can't be deleted"):
        app.delete("demo")
    app.open("career-2076-07-01")
    app.delete("career-2076-07-01")
    assert app.info()["save"] is None and not (web.DATA / "career-2076-07-01").exists()
    assert [s["id"] for s in app.settings()["saves"]] == ["demo"]


def test_new_save_fits_first_stage_then_rest_on_first_use(tmp_path, monkeypatch, folder):
    from fm26scout import prepare
    from fm26scout.scout import Scout

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    save = tmp_path / "save"
    shutil.copytree(folder, save)
    players, full = pd.read_parquet(save / "players.parquet"), pd.read_parquet(
        save / "predictions.parquet")
    (save / "db").mkdir()

    def fit(target):
        cols = [f"{p}_{target}" for p in ("lo", "pred", "hi")]
        full[cols].to_parquet(save / "db" / f"{target}.parquet")

    for target in prepare.FIRST:
        fit(target)
    prepare._assemble(save, players)
    trust = pd.read_csv(save / "trust.csv").set_index("target")
    assert len(trust) == 16 and trust.loc["dirtiness", "checked_on"] == "demo save"
    assert trust.loc["potential_ability", "checked_on"] == "this save"
    assert len(prepare.worth_fitting()) < 16  # unpredictable targets are never fitted

    scout = Scout(save)
    assert "current_ability" in scout.pending and "potential_ability" not in scout.pending
    assert "current_ability" not in scout.card(0)["predictions"]
    fitted = []
    monkeypatch.setattr(prepare, "predict", lambda folder, targets, **kw: (
        fitted.extend(targets), fit(targets[0]), prepare._assemble(folder, players)))
    report = scout.search({"rank": {"pred_current_ability": 1}, "limit": 3})
    assert fitted == ["current_ability"] and "current_ability" not in scout.pending
    assert "current_ability" in report["players"][0]["predictions"]
    scout.search({"rank": {"pred_current_ability": 1}})
    assert fitted == ["current_ability"]  # once


def serving(monkeypatch) -> tuple[web.ThreadingHTTPServer, int]:
    """web.serve() on a free port in a background thread, once it accepts requests."""
    started, ready = {}, threading.Event()
    original = web.ThreadingHTTPServer.__init__

    def capture(self, address, handler):
        original(self, ("127.0.0.1", 0), handler)
        started["httpd"] = self
        ready.set()

    monkeypatch.setattr(web.ThreadingHTTPServer, "__init__", capture)
    threading.Thread(target=web.serve, args=(None,), kwargs={"port": 0}, daemon=True).start()
    assert ready.wait(10), "the server did not start"
    return started["httpd"], started["httpd"].server_address[1]


def test_requests_from_other_sites_are_refused(env, tmp_path, monkeypatch):
    monkeypatch.setattr(web, "DATA", tmp_path / "data")
    monkeypatch.setattr(web, "REMEMBER", tmp_path / "data" / "web.json")
    httpd, port = serving(monkeypatch)

    def call(headers, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/"
                                     + ("search" if body else "settings"),
                                     data=body, headers=headers)
        try:
            with urllib.request.urlopen(req) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    try:
        # serve() built its allowed hosts for port 0; the real check is by name
        assert call({"Host": "evil.example:80"})[0] == 403
        assert call({"Host": "localhost:0", "Origin": "https://evil.example"})[0] == 403
        status, body = call({"Host": "localhost:0"})
        assert status == 200 and body["problems"]
        status, body = call({"Host": "localhost:0"}, b"{not json")
        assert status == 400 and "JSON" in body["error"]
    finally:
        httpd.shutdown()


def test_unreadable_remembered_save_is_ignored(app):
    web.REMEMBER.write_text("{half a file")
    assert web.App(None).active == "demo"


def test_examples_replay_a_recorded_plan_without_a_deepseek_key(app, tmp_path, monkeypatch):
    """No key: a new chat that opens with an example runs the engine on the recorded plan and
    says so; anything else, or a follow-up, is refused in words."""
    answer = {"reply": "Young strikers.", "reading": [], "questions": [],
              "spec": {"filters": {"age": [None, 21]}, "rank": {"pred_potential_ability": 1}}}
    monkeypatch.setattr(web, "EXAMPLES", tmp_path / "examples.json")
    web.EXAMPLES.write_text(json.dumps([{"prompt": "Young  Strikers", "answer": answer}]))
    app.open("career-2076-07-01")

    def chat(text, **body):
        return app.chat({"history": [{"role": "user", "content": text}], **body})

    out = chat(" young strikers ", by="ceiling")
    assert out["recorded"] and out["reply"] == "Young strikers." and out["report"]["players"]
    assert out["spec"]["explain"] == "potential_ability" and out["spec"]["by"] == "ceiling"
    again = chat("young strikers")  # replayed as often as asked, the record is not changed
    assert again["recorded"] and again["reply"] == "Young strikers." and "explain" not in answer
    with pytest.raises(RuntimeError, match="No DeepSeek API key.*examples"):
        chat("old keepers")
    with pytest.raises(RuntimeError, match="No DeepSeek API key"):
        chat("young strikers", spec=out["spec"])  # a follow-up needs the language model


def test_page_examples_are_the_recorded_ones():
    page = web.PAGE.read_text()
    shown = re.findall(r"'([^']*)'", re.search(r"const EXAMPLES = \[(.*?)\];", page, re.S)[1])
    recorded = json.loads(web.EXAMPLES.read_text())
    assert [e["prompt"] for e in recorded] == shown
    assert all(e["answer"]["spec"] and e["answer"]["reply"] for e in recorded)


class FakeRegressor:
    """Stands in for TabPFN: 'default' predicts 10 with range 8-12 (four bars 6-8-10-12-14
    holding 10/40/40/10%), 'thinking' predicts 11."""

    def __init__(self, tier):
        self.tier = tier

    def fit(self, X, y):
        return self

    def predict(self, X, output_type="mean", quantiles=None):
        if self.tier == "thinking" and output_type != "mean":
            raise RuntimeError("thinking returns the mean only")
        if output_type == "quantiles":
            q = np.interp(quantiles, [0, 0.1, 0.5, 0.9, 1], [6.0, 8, 10, 12, 14])
            return [np.full(len(X), v) for v in q]
        return np.full(len(X), 11.0 if self.tier == "thinking" else 10.0)


def test_rate_limited_thinking_call_does_not_repeat_the_default_one(monkeypatch):
    from fm26scout import models, quota

    calls = []

    class Limited(FakeRegressor):
        def predict(self, X, output_type="mean", quantiles=None):
            calls.append(self.tier)
            if self.tier == "thinking" and calls.count("thinking") == 1:
                raise RuntimeError("Fail to call predict: [HTTP 429] Rate limit exceeded.")
            return super().predict(X, output_type, quantiles)

    monkeypatch.setattr(models, "_regressor", lambda cats, seed, local, tier="default":
                        Limited(tier))
    monkeypatch.setattr(quota, "_hook", lambda: None)
    monkeypatch.setattr(quota, "_sleep", lambda seconds: None)
    X = pd.DataFrame({"a": [1.0, 2.0]})
    model = models.TabPFN(X, [1, 2], local=False, tier="thinking")
    q, _ = model.distribution(X, [0.1, 0.5, 0.9], np.array([10.0, 11, 12]))
    assert calls == ["default", "thinking", "thinking"] and np.allclose(q[1], 11)


def test_thinking_ranges_are_centred_on_its_estimate(monkeypatch):
    from fm26scout import models

    monkeypatch.setattr(models, "_regressor", lambda cats, seed, local, tier="default":
                        FakeRegressor(tier))
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    grid = np.array([8.0, 9, 10, 11, 12, 13])
    q, cdf = models.TabPFN(X, [1, 2, 3], local=False, tier="thinking").distribution(
        X, [0.1, 0.5, 0.9], grid)
    assert np.allclose(q[:, 0], [9, 11, 13]) and q.shape == (3, 3)
    assert np.allclose(cdf[0], [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]) and cdf.shape == (3, 6)
    q, cdf = models.TabPFN(X, [1, 2, 3], local=False, tier="fast").distribution(
        X, [0.1, 0.5, 0.9], grid)
    assert np.allclose(q[:, 0], [8, 10, 12]) and np.allclose(cdf[0, ::2], [0.1, 0.5, 0.9])


def test_thinking_is_cloud_only(env):
    from fm26scout import models

    with pytest.raises(RuntimeError, match="cloud"):
        models._regressor([], 0, local=True, tier="thinking")
    settings.write_env("FM26_TABPFN", "local")
    with pytest.raises(ValueError, match="cloud"):
        settings.set_model("thinking")


def test_switching_model_rebuilds_the_active_save_only(app, monkeypatch):
    calls = []
    monkeypatch.setattr(web.prepare, "remodel", lambda folder, tier, step: calls.append(
        (folder.name, tier)))
    monkeypatch.setenv("TABPFN_TOKEN", "a-token-from-the-shell-123")
    app.open("demo")
    app.model({"model": "fast"})  # the demo save keeps its estimates
    assert calls == [] and settings.model() == "fast" and app.job is None
    app.open("career-2076-07-01")
    app.model({"model": "thinking"})
    for _ in range(100):
        if app.job["done"] or app.job["error"]:
            break
        threading.Event().wait(0.02)
    assert calls == [("career-2076-07-01", "thinking")] and app.job["done"]
    assert app.scout.frozen is None


def test_each_model_keeps_its_own_estimates(tmp_path, monkeypatch, folder):
    from fm26scout import predict, prepare

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    save = tmp_path / "save"
    shutil.copytree(folder, save)
    full = pd.read_parquet(save / "predictions.parquet")
    used = []

    def fake(players, target, n_train, on_fold=None, tier="default"):
        used.append(tier)
        cdf = np.tile(np.linspace(0, 1, predict.POINTS, dtype=np.float32), (len(players), 1))
        return full[[f"{p}_{target}" for p in ("lo", "pred", "hi")]], cdf

    monkeypatch.setattr(predict, "predict_target", fake)
    prepare.remodel(save, "fast", lambda *a: None)
    assert (save / "db" / "fast" / "potential_ability.parquet").exists()
    assert not (save / "db" / "potential_ability.parquet").exists()
    assert prepare.model_of(save) == "fast" and set(used) == {"fast"}
    grid, cdf = prepare.distribution(save / "db" / "fast", "potential_ability")
    assert cdf.shape == (len(full), predict.POINTS) and cdf.dtype == np.float16
    assert grid[-1] == 200  # in the target's units: prices are kept as money, not log
    money = prepare.distribution(save / "db" / "fast", "market_value")[0]
    assert np.allclose(money, np.expm1(predict.grid("market_value")))
    assert "From the full distribution" in (tmp_path / "reports" / "save" / "trust.md").read_text()
    prepare.remodel(save, "fast", lambda *a: None)
    assert len(used) == 2  # already fitted with Fast: nothing fitted again


def test_another_save_in_the_same_folder_starts_afresh(tmp_path, monkeypatch, folder):
    """Same file name and in-game date, other players: nothing fitted before is reused, and
    estimates are joined by player id, never by row position."""
    from fm26scout import predict, prepare

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    save = tmp_path / "save"
    shutil.copytree(folder, save)
    full = pd.read_parquet(save / "predictions.parquet").set_index("player_id")
    fitted = []

    def fake(players, target, n_train, on_fold=None, tier="default"):
        fitted.append(target)
        cols = [f"{p}_{target}" for p in ("lo", "pred", "hi")]
        cdf = np.zeros((len(players), predict.POINTS), dtype=np.float32)
        return full.loc[players["player_id"], cols].reset_index(drop=True), cdf

    monkeypatch.setattr(predict, "predict_target", fake)
    prepare.predict(save, targets=["potential_ability"], log=lambda _: None)
    assert fitted == ["potential_ability"]
    players = pd.read_parquet(save / "players.parquet")
    other = players.iloc[::-1].head(1500)  # another save: fewer players, another order
    other.to_parquet(save / "players.parquet")
    (save / "why").mkdir()
    (save / "why" / "default_potential_ability.json").write_text("{}")
    prepare.predict(save, targets=["potential_ability"], log=lambda _: None)
    assert fitted == ["potential_ability"] * 2 and not (save / "why").exists()
    db = pd.read_parquet(save / "predictions.parquet")
    assert db["player_id"].tolist() == other["player_id"].tolist()
    assert np.allclose(db["pred_potential_ability"],
                       full.loc[other["player_id"], "pred_potential_ability"])
    shuffled = pd.read_parquet(save / "db" / "potential_ability.parquet").sample(frac=1)
    shuffled.to_parquet(save / "db" / "potential_ability.parquet")
    prepare._assemble(save, other)
    assert pd.read_parquet(save / "predictions.parquet").equals(db)


def test_failed_upload_leaves_nothing_behind(tmp_path, monkeypatch, folder):
    from fm26scout import prepare

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    leftover = tmp_path / "data" / ".extracting"
    leftover.mkdir(parents=True)
    (leftover / "players.parquet").write_text("from a stopped upload")

    def extract(save, out, log=print):
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(folder / "players.parquet", out / "players.parquet")
        (out / "meta.json").write_text(json.dumps({"save_date": "2076-07-01"}))

    def refused(*args, **kwargs):
        raise RuntimeError("Your Prior Labs token budget is used up")

    monkeypatch.setattr(prepare, "extract", extract)
    monkeypatch.setattr(prepare, "predict", refused)
    upload = tmp_path / "career.fm"
    upload.write_text("save")
    with pytest.raises(RuntimeError, match="budget"):
        prepare.from_upload(upload, lambda text, fraction: None)
    assert [p.name for p in (tmp_path / "data").iterdir()] == []


def test_upload_while_the_active_save_is_fitting_is_ready_not_failed(app, monkeypatch):
    app.open("demo")
    app.scout.fitting = {"target": "consistency", "done": 1, "total": 5}
    monkeypatch.setattr(web.prepare, "from_upload", lambda save, step, tier, on_folder:
                        web.DATA / "career-2076-07-01")
    app.job = {"kind": "upload", "model": "default", "done": False, "error": None}
    app._prepare(web.UPLOADS / "career.fm")
    assert app.job["done"] and app.job["error"] is None and app.job["note"]
    assert app.job["save"] == "career-2076-07-01" and app.active == "demo"


def test_no_model_switch_while_a_target_is_fitted(app, monkeypatch):
    monkeypatch.setenv("TABPFN_TOKEN", "a-token-from-the-shell-123")
    app.open("career-2076-07-01")
    with app.scout._lock:  # ensure() holds it for the whole fit
        with pytest.raises(RuntimeError, match="estimating"):
            app.model({"model": "fast"})
    assert app.scout.frozen is None and app.job is None
    app.scout.freeze("rebuilding")
    app.scout.pending = ["consistency"]
    with pytest.raises(RuntimeError, match="rebuilding"):
        app.scout.ensure(["consistency"])
