"""What the Settings page reads and writes: API keys and the TabPFN runtime, kept in the
project's .env (git-ignored, owner-only permissions).

Keys never leave this process: the page only ever gets a masked form, and nothing here logs
them. A value from the shell environment wins over .env at start-up, like everywhere else.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the project folder: every file the app writes
SAMPLE = ROOT / "sample" / "players.parquet"  # the demo save, shipped with the app
ENV = ROOT / ".env"
WEIGHTS = ROOT / "data" / "tabpfn-weights"  # local TabPFN weights, downloaded on first use

# id -> (variable, other variables that also count, what it is for)
KEYS = {
    "deepseek": ("DEEPSEEK_API_KEY", ("LLM_API_KEY",), "Runs the chat agent"),
    "tabpfn": ("TABPFN_TOKEN", (), "Prior Labs cloud estimates"),
}
_PLAIN = re.compile(r"[\x21-\x7e]{16,512}")  # printable, no spaces, sensible length
_writing = threading.Lock()  # one change to .env at a time


def read_env(path: Path | None = None) -> dict[str, str]:
    """NAME=value lines; comments and blank lines skipped."""
    path, out = path or ENV, {}
    for line in path.read_text().splitlines() if path.exists() else []:
        name, sep, value = line.partition("=")
        if sep and name.strip() and not name.lstrip().startswith("#"):
            out[name.strip()] = value.strip().strip("'\"")
    return out


def write_env(name: str, value: str | None, path: Path | None = None) -> None:
    """Set (or with None remove) one variable in .env, keeping every other line. Written to a
    temporary file first and swapped in, readable by the owner only."""
    path = path or ENV
    with _writing:
        lines = path.read_text().splitlines() if path.exists() else []
        kept = [line for line in lines if line.partition("=")[0].strip() != name]
        if value is not None:
            kept.append(f"{name}={value}")
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")  # owner-only
        try:
            with os.fdopen(fd, "w") as f:
                f.write("\n".join(kept) + ("\n" if kept else ""))
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def mask(value: str) -> str:
    return value[:3] + "•" * 8 + value[-4:] if len(value) > 12 else "•" * 8


def key_status() -> dict:
    """Per key: set or not, masked value, and where it comes from (.env or the shell)."""
    stored, out = read_env(), {}
    for key, (name, others, use) in KEYS.items():
        found = next((n for n in (name, *others) if os.environ.get(n)), None)
        value = os.environ.get(found, "") if found else ""
        source = None if not found else ".env" if stored.get(found) == value else "environment"
        out[key] = {"variable": name, "use": use, "set": bool(found), "masked": mask(value)
                    if found else "", "source": source}
    return out


def check_key(key: str, value: str) -> str | None:
    """Raises ValueError when the provider rejects the key; returns a warning when it could
    not be checked (offline, provider down), so the key is still saved."""
    if not _PLAIN.fullmatch(value):
        raise ValueError("That doesn't look like an API key: paste it without spaces or quotes")
    try:
        if key == "deepseek":
            import openai

            client = openai.OpenAI(api_key=value, timeout=10, max_retries=0,
                                   base_url=os.environ.get("LLM_BASE_URL",
                                                           "https://api.deepseek.com"))
            try:
                client.models.list()
            except openai.AuthenticationError:
                raise ValueError("DeepSeek rejected this key") from None
        else:
            from tabpfn_client.client import ServiceClient

            valid = ServiceClient.is_auth_token_outdated(value)
            if valid is False:
                raise ValueError("Prior Labs rejected this token")
            if valid is None:
                return "Saved. Prior Labs says the account behind this token is not verified yet."
    except ValueError:
        raise
    except Exception:  # network trouble: keep the key, say it is unchecked
        return "Saved, but it could not be checked right now (no connection to the provider)."
    return None


def save_key(key: str, value: str) -> str | None:
    if key not in KEYS:
        raise ValueError(f"Unknown key {key!r}")
    value = value.strip()
    warning = check_key(key, value)
    write_env(KEYS[key][0], value)
    return warning


def remove_key(key: str) -> None:
    if key not in KEYS:
        raise ValueError(f"Unknown key {key!r}")
    name, others, _ = KEYS[key]
    stored = read_env()
    if not any(n in stored for n in (name, *others)):
        raise ValueError("This key is set in your shell environment, not in .env: unset it there")
    for n in (name, *others):
        if n in stored:
            write_env(n, None)


def local_weights() -> bool:
    """TabPFN-3.5's weights are on this computer (fetched on the first local fit, once the
    Prior Labs licence is accepted)."""
    folder = Path(os.environ.get("TABPFN_MODEL_CACHE_DIR", WEIGHTS))
    return any(folder.glob("tabpfn-v3.5-[0-9]*"))


_devices: list[str] | None = None


def devices() -> list[str]:
    """Devices this computer can run TabPFN on (asks torch once)."""
    global _devices
    if _devices is None:
        import torch

        _devices = ["cpu"] + [d for d, ok in (("cuda", torch.cuda.is_available()),
                                              ("mps", torch.backends.mps.is_available())) if ok]
    return _devices


def device() -> str:
    """The chosen device, or the fastest one found."""
    chosen = os.environ.get("FM26_TABPFN_DEVICE", "").lower()
    found = devices()
    return chosen if chosen in found else found[-1]


def model() -> str:
    """The TabPFN model new estimates are made with (models.TIERS)."""
    from .models import TIERS

    chosen = os.environ.get("FM26_TABPFN_MODEL", "default")
    return chosen if chosen in TIERS else "default"


def set_model(tier: str) -> None:
    from .models import TIERS, hosted

    if tier not in TIERS:
        raise ValueError(f"Unknown TabPFN model {tier!r}")
    if tier == "thinking" and not hosted():
        raise ValueError("TabPFN-3.5 Thinking runs only in the Prior Labs cloud: turn off "
                         "Run TabPFN locally first")
    write_env("FM26_TABPFN_MODEL", tier)


def runtime() -> dict:
    """The TabPFN runtime, and when hosted with a token the Prior Labs account's usage (None
    when it can't be read right now)."""
    from . import quota
    from .models import THINKING_S, hosted

    return {"local": not hosted(), "device": device(),
            "devices": devices(), "model": model(),
            "thinking_s": THINKING_S, "usage": quota.summary(quota.usage()),
            "thinking_hour_limit": quota.THINKING_HOUR}


def set_runtime(local: bool, chosen: str | None = None) -> None:
    if local and model() == "thinking":
        raise ValueError("TabPFN-3.5 Thinking runs only in the Prior Labs cloud: pick "
                         "TabPFN-3.5 Fast or TabPFN-3.5 first")
    if chosen is not None:
        if chosen not in devices():
            raise ValueError(f"No {chosen.upper()} device on this computer")
        write_env("FM26_TABPFN_DEVICE", chosen)
    write_env("FM26_TABPFN", "local" if local else "hosted")


def problems() -> list[str]:
    """What stops the full app from working, in words for the page and the terminal."""
    from .models import hosted

    keys, out = key_status(), []
    if not keys["deepseek"]["set"]:
        out.append("No DeepSeek API key: the chat only replays the example questions. Add "
                   "one in Settings to ask your own.")
    if hosted() and not keys["tabpfn"]["set"]:
        out.append("No TabPFN API token: no new estimates or why lines. Add one in Settings, "
                   "or turn on Run TabPFN locally.")
    return out
