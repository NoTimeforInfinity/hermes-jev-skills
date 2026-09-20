"""One small, strict client for TypeSafe Jev (POST /v1/systemone).

Jev answers typed questions about a state: ``choice`` (one of a closed set),
``score`` (a position on an ordered rubric) and ``noul`` (probability of yes).
It never writes text. Every helper here validates the reply against the question
that was asked, so a malformed or surprising answer becomes a ``JevError`` and
the caller takes its fail-open path rather than acting on junk.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from . import keystore

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
# Jev, reached through OpenRouter's Decisions API instead of TypeSafe directly: one key
# instead of two for anyone already on OpenRouter. Same request, same answers, same model -
# only the URL and the model id differ. Contributed as PR #1 by Lorenzo DZ (@Barba2k2),
# whose version prompted a chat model for JSON instead; that returns an LLM's guess with a
# made-up confidence, which is the one thing a decision model exists not to do.
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_MODEL = "~typesafe/jev-latest"
# Venice serves the same decision model as a first-class modality (`type: "decision"`), reached
# at POST /api/v1/decisions — not /chat/completions, which answers 404 for it. It is listed by
# /models?type=decision (and only by type=all otherwise), named "Jev (System One)", and priced
# 0 usd / 0 diem. Request and answers are the same shapes as the other two providers.
VENICE_ENDPOINT = "https://api.venice.ai/api/v1/decisions"
VENICE_MODEL = "jev-latest"
MAX_RESPONSE_BYTES = 1_000_000
MAX_STATE_CHARS = 60_000
USER_AGENT = "hermes-jev-skills/0.1"

State = Union[str, Mapping[str, Any], Sequence[Any]]
Transport = Callable[[bytes, Dict[str, str], float], bytes]


class JevError(RuntimeError):
    """Anything that means "do not trust or use this Jev result"."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


# ── question builders ────────────────────────────────────────────────────────

def choice(instructions: str, criteria: Mapping[str, str]) -> Dict[str, Any]:
    if len(criteria) < 2:
        raise ValueError("a choice needs at least two options")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions: str, levels: Sequence[str]) -> Dict[str, Any]:
    if len(levels) < 2:
        raise ValueError("a score needs at least two levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def noul(instructions: str) -> Dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


# ── transport ────────────────────────────────────────────────────────────────

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401, ANN001
        # A redirect would carry the bearer token to another origin.
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def _http_transport(body: bytes, headers: Dict[str, str], timeout: float, url: str = ENDPOINT) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        code = {401: "auth_failed", 403: "auth_failed", 402: "credits_exhausted",
                429: "rate_limited", 529: "overloaded"}.get(error.code, f"http_{error.code}")
        raise JevError(code) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise JevError("network") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise JevError("response_too_large")
    return raw


def _openrouter_transport(body: bytes, headers: Dict[str, str], timeout: float) -> bytes:
    return _http_transport(body, headers, timeout, OPENROUTER_ENDPOINT)


def _venice_transport(body: bytes, headers: Dict[str, str], timeout: float) -> bytes:
    return _http_transport(body, headers, timeout, VENICE_ENDPOINT)


def _default_model(via: str) -> str:
    """Read the module globals at call time: binding them at import would defeat a later
    monkeypatch, which is the trap CONTRIBUTING.md warns about."""
    if via == "openrouter":
        return OPENROUTER_MODEL
    if via == "venice":
        return VENICE_MODEL
    return DEFAULT_MODEL


def _transport_for(via: str) -> Callable[[bytes, Dict[str, str], float], bytes]:
    if via == "openrouter":
        return _openrouter_transport
    if via == "venice":
        return _venice_transport
    return _http_transport


_RETRYABLE = {"rate_limited", "overloaded", "network", "http_500", "http_502", "http_503", "http_504"}


# ── validation ───────────────────────────────────────────────────────────────

def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError("malformed", f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number) or not -1e-6 <= number <= 1 + 1e-6:
        raise JevError("malformed", f"{name} is outside 0..1")
    return min(1.0, max(0.0, number))


def _check_answer(name: str, question: Mapping[str, Any], answer: Any) -> Dict[str, Any]:
    if not isinstance(answer, dict) or answer.get("type") != question["type"]:
        raise JevError("malformed", f"answer {name} has the wrong type")
    kind = question["type"]
    if kind == "noul":
        return {"type": "noul", "noul": _unit(answer.get("noul"), f"{name}.noul")}
    if kind == "choice":
        options = set(question["criteria"])
        picked = answer.get("choice")
        if picked not in options:
            raise JevError("malformed", f"answer {name} chose an option that was not offered")
        raw = answer.get("probabilities")
        if not isinstance(raw, dict) or not set(raw) <= options:
            raise JevError("malformed", f"answer {name} has bad probabilities")
        probabilities = {key: _unit(value, f"{name}.p[{key}]") for key, value in raw.items()}
        return {"type": "choice", "choice": picked, "probabilities": probabilities,
                "confidence": _unit(answer.get("confidence"), f"{name}.confidence")}
    levels = len(question["criteria"])
    value = answer.get("score")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise JevError("malformed", f"answer {name} has no numeric score")
    if not -0.5 <= float(value) <= levels - 0.5:
        raise JevError("malformed", f"answer {name} scored off the rubric")
    # The per-level spread says far more than the averaged score: an unsure answer averages to
    # the middle of the rubric, which looks like a real "medium-hard" unless you read the spread.
    raw = answer.get("probabilities")
    spread = {}
    if isinstance(raw, dict):
        for key, probability in raw.items():
            if str(key).isdigit() and int(key) < levels:
                spread[int(key)] = _unit(probability, f"{name}.p[{key}]")
    return {"type": "score", "score": float(value), "probabilities": spread,
            "confidence": _unit(answer.get("confidence", 1.0), f"{name}.confidence")}


# ── public call ──────────────────────────────────────────────────────────────

def ask(
    state: State,
    questions: Mapping[str, Mapping[str, Any]],
    *,
    timeout: float = 4.0,
    retries: int = 1,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    transport: Optional[Transport] = None,
) -> Dict[str, Any]:
    """Ask Jev every question against one state, in a single request.

    Returns ``{"answers": {...validated...}, "usage": {...}, "latency_ms": int}``.
    Raises ``JevError`` for anything the caller should not act on. ``timeout`` is a
    total wall-clock budget across retries, not a per-attempt one.
    """
    if not questions:
        raise ValueError("no questions")
    via = provider or ("typesafe" if api_key else keystore.provider())
    if via not in keystore.PROVIDERS:
        via = "typesafe"
    key = api_key or keystore.resolve(via)
    if not key:
        raise JevError("no_key", "run `jev setup-key`")
    encoded_state = state if isinstance(state, str) else json.dumps(state, separators=(",", ":"), default=str)
    if len(encoded_state) > MAX_STATE_CHARS:
        raise JevError("state_too_large")
    default_model = _default_model(via)
    body = json.dumps(
        {"state": state, "model": model or os.environ.get("TYPESAFE_MODEL") or default_model,
         "questions": {name: dict(q) for name, q in questions.items()}},
        separators=(",", ":"), default=str,
    ).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "Accept": "application/json", "User-Agent": USER_AGENT}
    if via == "openrouter":
        # OpenRouter asks callers to identify themselves; neither header carries anything
        # about the person or the decision.
        headers["HTTP-Referer"] = "https://github.com/kerpopule/hermes-jev-skills"
        headers["X-Title"] = "Hermes Jev Skills"
    send = transport or _transport_for(via)

    started = time.monotonic()
    attempt = 0
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0.05:
            raise JevError("timeout")
        try:
            raw = send(body, headers, remaining)
            break
        except JevError as error:
            attempt += 1
            if error.code not in _RETRYABLE or attempt > retries:
                raise
            time.sleep(min(0.25 * attempt, max(0.0, timeout - (time.monotonic() - started) - 0.1)))

    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise JevError("malformed", "reply is not JSON") from None
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        raise JevError("malformed", "reply has no answers")
    checked = {name: _check_answer(name, question, answers.get(name)) for name, question in questions.items()}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {"answers": checked, "usage": usage, "latency_ms": int((time.monotonic() - started) * 1000)}


def verify_key(api_key: str, timeout: float = 10.0, provider: str = "typesafe") -> bool:
    """One tiny synthetic call. True means the key is accepted by that provider."""
    try:
        ask("The build finished and all tests passed.",
            {"ok": noul("The text reports a successful outcome")},
            api_key=api_key, provider=provider, timeout=timeout)
        return True
    except JevError:
        return False


def batches(items: Sequence[Any], size: int) -> List[Sequence[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]
