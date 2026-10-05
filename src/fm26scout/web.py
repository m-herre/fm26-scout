"""`fm26 web`: the chat app. Standard-library server, one page, JSON endpoints.

One save is active at a time: the one the user picked in Settings (remembered across restarts),
else the demo save.
Other prepared saves stay on disk until the user switches to them or deletes them.

GET  /api/info           save, player count, what the model can and cannot tell, what is missing
GET  /api/player?id=...  a player's page
GET  /api/status         the TabPFN fit running for this save, if any (first use of a target)
GET  /api/cost           Prior Labs' token quote per model: the active save's first stage, a target
POST /api/chat           {history, spec, shortlist, by} -> reply, questions, readings, spec, results
                         (no DeepSeek key: the page's examples replay a recorded plan, "recorded")
POST /api/search         {spec} the current search again (the Ranking switch), no language model
POST /api/why            {ids, target, all} -> why lines (one TabPFN call for the players not
                         kept yet in the save's why/ folder; the demo save ships some)
POST /api/estimate       {target} fit a target not estimated yet, for all players
GET  /api/settings       the active save, every save, masked keys, TabPFN runtime, the upload
POST /api/keys           {key, value} save a key (checked with its provider) | {key, remove}
POST /api/runtime        {local, device}
POST /api/model          {model} the TabPFN model; rebuilds the active save's estimates with it
POST /api/saves/open     {id} make it the active save
POST /api/saves/delete   {id} remove a prepared save and its reports (not the demo)
POST /api/upload?name=   the raw .fm file; prepared in the background, progress in /api/settings
GET  /api/chats[?id=]    the active save's recent chats (newest first), or one chat in full
POST /api/chats/save     {id?, title, history, spec, shortlist, turns} -> {id, chats}; keeps 5
POST /api/chats/delete   {id}
The page runs the conversation; the server keeps the save in use and its last CHATS_KEPT chats
(data/chats/<save>.json, deleted with the save; older chats are dropped).
Only this computer may talk to it: other hosts and other sites' pages are refused.
"""

from __future__ import annotations

import copy
import json
import shutil
import threading
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pyarrow.parquet as pq

from . import agent, prepare, quota, settings
from .models import TIERS, hosted
from .predict import FOLDS
from .schema import TARGETS
from .scout import GOOD, MONEY, NOT_REASONS, Scout

PAGE = Path(__file__).with_name("static") / "index.html"
DEMO = prepare.SAMPLE.parent
DATA = prepare.ROOT / "data"
EXAMPLES = DEMO / "examples.json"  # the planner's recorded answers to the page's examples
REMEMBER = DATA / "web.json"  # the active save, kept across restarts
UPLOADS = DATA / "uploads"
REPORTS = prepare.ROOT / "reports"
MAX_UPLOAD = 4 << 30
CHATS = DATA / "chats"
CHATS_KEPT = 5
NO_SAVE = "No save loaded yet: upload your FM26 save in Settings, or use the demo save."


class App:
    """The active save (exactly one, swapped whole when the user picks another), the list of
    prepared saves on disk, and the upload job."""

    def __init__(self, folder: Path | None):
        self.scout: Scout | None = None
        self.active: str | None = None
        self.extra: dict[str, Path] = {}  # --data outside data/
        self.job: dict | None = None
        self.lock = threading.Lock()
        if folder is not None and prepare.ready(folder):
            if folder != DEMO and folder.parent != DATA:
                self.extra[self._id(folder)] = folder
            self.open(self._id(folder))
        elif (remembered := _remembered()) in self.folders():
            self.open(remembered)
        elif "demo" in self.folders():  # first start: the demo save works at once
            self.open("demo")

    @staticmethod
    def _id(folder: Path) -> str:
        return "demo" if folder == DEMO else folder.name

    def folders(self) -> dict[str, Path]:
        found = {"demo": DEMO} if prepare.ready(DEMO) else {}
        if DATA.exists():
            found |= {d.name: d for d in sorted(DATA.iterdir())
                      if d.is_dir() and not d.name.startswith(".") and prepare.ready(d)}
        return found | self.extra

    def need(self) -> Scout:
        if self.scout is None:
            raise RuntimeError(NO_SAVE)
        return self.scout

    def describe(self, sid: str) -> dict:
        """A save as the Settings page lists it, read from its files (no need to load it)."""
        folder = self.folders()[sid]
        meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() \
            else {}
        trust = pd.read_csv(folder / "trust.csv")
        have = set(pq.read_schema(folder / "predictions.parquet").names)
        pending = [t for t, v in zip(trust["target"], trust["verdict"], strict=True)
                   if v in GOOD and f"pred_{t}" not in have]
        made = meta.get("prepared") or datetime.fromtimestamp(
            (folder / "predictions.parquet").stat().st_mtime).isoformat(timespec="seconds")
        size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
        first = [t for t in prepare.FIRST if t in prepare.worth_fitting()]
        ready = [m for m in TIERS
                 if all((prepare.db_dir(folder, m) / f"{t}.parquet").exists() for t in first)]
        return {"id": sid, "model": prepare.model_of(folder), "models_ready": ready,
                "name": "Demo save" if sid == "demo" else
                Path(meta.get("source", folder.name)).stem,
                "demo": sid == "demo", "path": str(folder.relative_to(prepare.ROOT)
                                                     if folder.is_relative_to(prepare.ROOT)
                                                     else folder),
                "players": meta.get("players"), "save_date": meta.get("save_date"),
                "build": meta.get("build"), "prepared": made, "pending": pending,
                "names": "made up" if meta.get("names") == "made up" else "real",
                "similarity": (folder / "embedding.npy").exists(), "size_mb": round(size / 1e6),
                "deletable": sid != "demo" and sid not in self.extra}

    def info(self) -> dict:
        out = {"save": None, "problems": settings.problems(),
               "tabpfn": "hosted" if hosted() else "local"}
        if self.scout is None:
            return out
        scout, meta = self.scout, self.describe(self.active)
        trust = scout.trust.reset_index()
        if "checked_on" not in trust:
            trust["checked_on"] = "this save"
        trust["pending"] = trust["target"].isin(scout.pending)
        return out | {"save": self.active, "players": len(scout.frame), "model": scout.model,
                      "save_date": meta["save_date"], "build": meta["build"],
                      "demo": meta["demo"], "name": meta["name"],
                      "targets": trust[["target", "verdict", "skill", "coverage_80", "mae",
                                        "checked_on", "pending"]].round(3).to_dict("records")}

    def settings(self) -> dict:
        saves = [self.describe(s) for s in self.folders()]
        return {"save": next((s for s in saves if s["id"] == self.active), None), "saves": saves,
                "keys": settings.key_status(), "runtime": settings.runtime(), "job": self.job,
                "problems": settings.problems(), "demo": "demo" in self.folders(),
                "fits": len([t for t in prepare.FIRST if t in prepare.worth_fitting()]) * FOLDS}

    def cost(self) -> dict:
        """Prior Labs' quote in tokens per model (None when local, offline or no token):
        `save` rebuilds the active save's first-stage estimates (an upload of a save this size
        costs the same), `target` estimates one more value on first use."""
        scout = self.scout
        folder = scout.folder if scout else DEMO
        players = pd.read_parquet(folder / "players.parquet", columns=list(TARGETS))
        first = [t for t in prepare.FIRST if t in prepare.worth_fitting()]
        later = (scout.pending if scout and scout.pending else first)[:1]
        return {m: {"save": quota.estimate(players, first, m),
                    "target": quota.estimate(players, later, m)} for m in TIERS}

    def status(self) -> dict:
        return {"fitting": self.scout.fitting if self.scout else None}

    def why(self, body: dict) -> dict:
        scout = self.need()
        rows = [scout.row(i) for i in body["ids"]]
        effects = scout.explain(rows, body.get("target", "potential_ability"))
        out = {}
        for row in rows:
            e = effects[row]
            key = str(scout.frame.loc[row, "player_id"])
            reasons = e if body.get("all") else [x for x in e if x[0] not in NOT_REASONS][:4]
            out[key] = {"why": reasons, "money": round(sum(v for n, v in e if n in MONEY), 2)}
        return out

    def chat(self, body: dict) -> dict:
        """One chat turn. Without a DeepSeek key, a new chat that opens with one of the page's
        examples replays the planner's recorded answer (marked "recorded"); the engine and the
        why lines run as usual."""
        scout, history, spec = self.need(), body["history"], body.get("spec")
        report = {"players": body.get("shortlist") or []}
        if agent._key():
            return agent.turn(scout, history, spec, report, body.get("by"))
        examples = json.loads(EXAMPLES.read_text()) if EXAMPLES.exists() else []
        answer = agent.recorded(examples, history, spec, report)
        if answer is None:
            raise RuntimeError(f"{agent.NO_KEY}, or start a new chat with one of the examples")
        return agent.turn(scout, history, None, report, body.get("by"),
                          ask=lambda _: copy.deepcopy(answer)) | {"recorded": True}

    def search(self, body: dict) -> dict:
        spec = body["spec"]
        report = self.need().search({**spec, "explain": None})
        return {"spec": spec, "report": report, "tip": agent._tip(report)}

    def estimate(self, body: dict) -> dict:
        scout = self.need()
        if body["target"] not in scout.targets:
            raise ValueError(f"{body['target']} can't be estimated from what the game shows")
        scout.ensure([body["target"]])
        return {"pending": scout.pending}

    def keys(self, body: dict) -> dict:
        if body.get("remove"):
            settings.remove_key(body["key"])
            return {"keys": settings.key_status(), "problems": settings.problems()}
        warning = settings.save_key(body["key"], str(body.get("value", "")))
        return {"keys": settings.key_status(), "problems": settings.problems(), "warning": warning}

    def runtime(self, body: dict) -> dict:
        settings.set_runtime(bool(body["local"]), body.get("device"))
        return self.settings()

    def model(self, body: dict) -> dict:
        """Use another TabPFN model: for new saves, and for the active save's estimates (rebuilt
        in the background; the demo save keeps its default estimates)."""
        tier = body["model"]
        if tier not in TIERS:
            raise ValueError(f"Unknown TabPFN model {tier!r}")
        scout, sid = self.scout, self.active
        rebuild = scout is not None and sid != "demo" and scout.model != tier
        if rebuild:
            if self.job and not (self.job["done"] or self.job["error"]):
                raise RuntimeError("A save is being prepared right now: wait until it is ready")
            blocking = [p for p in settings.problems() if "TabPFN" in p]
            if blocking:
                raise RuntimeError(blocking[0])
            scout.freeze("This save's estimates are being rebuilt with another model: try "
                         "again when that is done")
        try:
            settings.set_model(tier)
            if rebuild:
                self._start({"kind": "model", "name": self.describe(sid)["name"], "model": tier,
                             "save": sid, "step": "Starting", "fraction": 0.0},
                            lambda: self._remodel(scout, sid, tier))
        except Exception:
            if rebuild:
                scout.frozen = None
            raise
        return self.settings()

    def _start(self, job: dict, work) -> None:
        with self.lock:
            if self.job and not (self.job["done"] or self.job["error"]):
                raise RuntimeError("A save is being prepared right now: wait until it is ready")
            self.job = job | {"done": False, "error": None}
        threading.Thread(target=work, daemon=True).start()

    def _remodel(self, scout: Scout, sid: str, tier: str) -> None:
        try:
            prepare.remodel(scout.folder, tier,
                            lambda text, f: self.job.update(step=text, fraction=round(f, 3)))
            if self.active == sid:
                self.open(sid)
            self.job |= {"done": True}
        except Exception as error:
            traceback.print_exc()
            self.job |= {"error": str(error) or type(error).__name__}
        finally:
            scout.frozen = None

    def open(self, sid: str) -> dict:
        """Make `sid` the save the agent works with (the previous one is dropped from memory)."""
        folders = self.folders()
        if sid not in folders:
            raise ValueError("That save no longer exists")
        if sid != self.active and self.scout and self.scout.fitting:
            raise RuntimeError("TabPFN is estimating for the active save right now: switch "
                               "when it is done")
        scout = Scout(folders[sid])
        with self.lock:
            self.scout, self.active = scout, sid
        REMEMBER.parent.mkdir(exist_ok=True)
        REMEMBER.write_text(json.dumps({"save": sid}))
        return self.settings()

    def delete(self, sid: str) -> dict:
        folders = self.folders()
        if sid not in folders:
            raise ValueError("That save no longer exists")
        if sid == "demo" or sid in self.extra:
            raise ValueError("The demo save ships with the app and can't be deleted here")
        if self.job and self.job.get("save") == sid and not (self.job["done"] or
                                                             self.job["error"]):
            raise RuntimeError("This save is being prepared right now: wait until it is ready")
        with self.lock:
            if sid == self.active:
                if self.scout and self.scout.fitting:
                    raise RuntimeError("TabPFN is estimating for this save right now: try again "
                                       "when it is done")
                self.scout, self.active = None, None
                REMEMBER.unlink(missing_ok=True)
            shutil.rmtree(folders[sid])
            shutil.rmtree(REPORTS / sid, ignore_errors=True)
            (CHATS / f"{sid}.json").unlink(missing_ok=True)
        return self.settings()

    def _chat_file(self) -> Path:
        self.need()
        return CHATS / f"{self.active}.json"

    def _read_chats(self) -> list[dict]:
        file = self._chat_file()
        return json.loads(file.read_text()) if file.exists() else []

    def chats(self, cid: str | None = None):
        chats = self._read_chats()
        if cid is not None:
            found = next((c for c in chats if c["id"] == cid), None)
            if found is None:
                raise KeyError("That chat is no longer stored")
            return found
        return [{k: c[k] for k in ("id", "title", "updated")} | {"turns": len(c["turns"])}
                for c in chats]

    def save_chat(self, body: dict) -> dict:
        """Store a chat (new or updated) as the newest; only the last CHATS_KEPT are kept."""
        with self.lock:
            chats = self._read_chats()
            cid = body.get("id") or datetime.now().strftime("%Y%m%d%H%M%S%f")
            chat = {"id": cid, "title": str(body.get("title") or "Chat")[:80],
                    "updated": datetime.now().isoformat(timespec="seconds"),
                    **{k: body.get(k) for k in ("history", "spec", "shortlist", "turns")}}
            self._write_chats([chat, *(c for c in chats if c["id"] != cid)][:CHATS_KEPT])
        return {"id": cid, "chats": self.chats()}

    def delete_chat(self, cid: str) -> list[dict]:
        with self.lock:
            if self._chat_file().exists():
                self._write_chats([c for c in self._read_chats() if c["id"] != cid])
        return self.chats()

    def _write_chats(self, chats: list[dict]) -> None:
        """Written whole to a temporary file and swapped in: never half a file."""
        file = self._chat_file()
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(chats, default=str))
        tmp.replace(file)

    def upload(self, name: str, size: int, stream) -> dict:
        name = Path(name).name
        if not name.lower().endswith(".fm"):
            raise ValueError("Choose a Football Manager 26 save file (.fm)")
        if not 0 < size <= MAX_UPLOAD:
            raise ValueError("That file is empty or larger than 4 GB")
        blocking = [p for p in settings.problems() if "TabPFN" in p]
        if blocking:
            raise RuntimeError(blocking[0])
        with self.lock:
            if self.job and not (self.job["done"] or self.job["error"]):
                raise RuntimeError("A save is already being prepared: wait until it is ready")
            self.job = {"kind": "upload", "name": name, "step": "Receiving the save",
                        "fraction": 0.0, "done": False, "error": None, "save": None,
                        "model": settings.model()}
        UPLOADS.mkdir(parents=True, exist_ok=True)
        target = UPLOADS / f"{prepare.slug(Path(name).stem)}.fm"
        part = target.with_suffix(".part")
        try:
            with part.open("wb") as f:
                left = size
                while left:
                    chunk = stream.read(min(left, 1 << 20))
                    if not chunk:
                        raise ValueError("The upload stopped before the whole file arrived")
                    f.write(chunk)
                    left -= len(chunk)
            part.replace(target)
        except Exception as error:
            part.unlink(missing_ok=True)
            self.job |= {"error": str(error)}
            raise
        threading.Thread(target=self._prepare, args=(target,), daemon=True).start()
        return {"job": self.job}

    def _prepare(self, save: Path) -> None:
        def step(text: str, fraction: float) -> None:
            self.job |= {"step": text, "fraction": round(fraction, 3)}

        try:
            folder = prepare.from_upload(save, step, tier=self.job["model"],
                                         on_folder=lambda f: self.job.update(save=f.name))
            try:
                self.open(folder.name)  # the user uploaded it to work with it
            except RuntimeError as busy:  # the active save is mid-fit: switch later
                self.job |= {"note": str(busy)}
            self.job |= {"done": True, "save": folder.name}
        except Exception as error:
            traceback.print_exc()
            self.job |= {"error": str(error) or type(error).__name__}
        finally:
            save.unlink(missing_ok=True)


def _remembered() -> str | None:
    """The save picked last time; None when there is none or the file is unreadable."""
    try:
        return json.loads(REMEMBER.read_text()).get("save")
    except (OSError, ValueError, AttributeError):
        return None


def serve(folder: Path | None, host: str = "127.0.0.1", port: int = 8626) -> None:
    app = App(folder)
    hosts = {f"{h}:{port}" for h in (host, "127.0.0.1", "localhost")}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload, kind: str = "application/json") -> None:
            data = payload if isinstance(payload, bytes) else \
                json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _run(self, call) -> None:
            try:
                self._send(200, call())
            except (KeyError, ValueError, RuntimeError) as error:
                self._send(400, {"error": str(error)})
            except Exception as error:  # show the message on the page, keep serving
                traceback.print_exc()
                self._send(500, {"error": f"{type(error).__name__}: {error}"})

        def _local(self) -> bool:
            """This computer only: blocks other hosts' names (DNS rebinding) and requests sent
            by other websites open in the same browser."""
            origin = self.headers.get("Origin")
            if self.headers.get("Host") in hosts and (origin is None or
                                                      urlparse(origin).netloc in hosts):
                return True
            self._send(403, {"error": "Forbidden"})
            return False

        def do_GET(self) -> None:
            if not self._local():
                return
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/info":
                self._run(app.info)
            elif url.path == "/api/settings":
                self._run(app.settings)
            elif url.path == "/api/status":
                self._run(app.status)
            elif url.path == "/api/cost":
                self._run(app.cost)
            elif url.path == "/api/chats":
                self._run(lambda: app.chats(query.get("id", [None])[0]))
            elif url.path == "/api/player":
                self._run(lambda: app.need().details(app.need().row(query["id"][0])))
            else:
                self._send(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if not self._local():
                return
            url = urlparse(self.path)
            size = int(self.headers.get("Content-Length") or 0)
            if url.path == "/api/upload":
                name = parse_qs(url.query).get("name", [""])[0]
                self._run(lambda: app.upload(name, size, self.rfile))
                return
            try:
                body = json.loads(self.rfile.read(size) or b"{}")
            except ValueError:
                self._send(400, {"error": "The request is not valid JSON"})
                return
            routes = {
                "/api/chat": lambda: app.chat(body),
                "/api/search": lambda: app.search(body),
                "/api/why": lambda: app.why(body),
                "/api/estimate": lambda: app.estimate(body),
                "/api/keys": lambda: app.keys(body),
                "/api/runtime": lambda: app.runtime(body),
                "/api/model": lambda: app.model(body),
                "/api/saves/open": lambda: app.open(body["id"]),
                "/api/saves/delete": lambda: app.delete(body["id"]),
                "/api/chats/save": lambda: app.save_chat(body),
                "/api/chats/delete": lambda: app.delete_chat(body["id"]),
            }
            if url.path in routes:
                self._run(routes[url.path])
            else:
                self._send(404, {"error": "Not found"})

        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

    loaded = (f"{len(app.scout.frame):,} players from {app.scout.folder}" if app.scout
              else "no save yet: upload one in Settings")
    print(f"FM26 Scout: http://{host}:{port}  ({loaded})")
    if folder is not None and not prepare.ready(folder):
        print(f"  warning: {folder} is not a prepared save (it needs "
              f"{', '.join(prepare.READY)}): run fm26 predict on it first")
    for problem in settings.problems():
        print(f"  warning: {problem}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
