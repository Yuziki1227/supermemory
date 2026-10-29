"""Claude Code hook entry points.

Hooks are short-lived child processes. They must be fast, silent on failure,
and never write to stdout except the JSON the harness expects — a broken hook
must not break the agent's session.

Events used:
  SessionStart  inject the briefing as additional context
  PreCompact    remind the agent to persist state before the window collapses
  SessionEnd    housekeeping (purge expired soft-deletes, trim the event log)
"""

from __future__ import annotations

import json
import os
import sys
import traceback


def _read_payload() -> dict:
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


def _apply_cwd(payload: dict) -> None:
    """Pin the namespace to the real project directory.

    Hook payloads carry ``cwd``; Claude Code also exports ``CLAUDE_PROJECT_DIR``.
    Prefer the payload, then the env var, and only then inherit cwd.
    """
    cwd = payload.get("cwd")
    if not (isinstance(cwd, str) and cwd.strip()):
        cwd = os.environ.get("CLAUDE_PROJECT_DIR")
    if isinstance(cwd, str) and cwd.strip():
        os.environ["SUPERMEMORY_PROJECT_CWD"] = cwd.strip()


def _emit(event: str, additional_context: str) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": additional_context,
        }
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()


def session_start(payload: dict) -> int:
    from .config import load_config
    from .core import MemoryStore

    _apply_cwd(payload)
    cfg = load_config()
    budget = int(cfg["briefing"].get("max_tokens", 900))
    source = str(payload.get("source") or "startup")
    with MemoryStore() as store:
        stats = store.stats()
        if stats.get("current", 0) == 0:
            return 0  # nothing remembered yet: stay out of the way
        text = store.briefing(max_tokens=budget)
    header = (
        f"[SuperMemory · 会话来源 {source}]\n"
        "以下是本项目的长期记忆简报。若发现过期或错误的信息，用 memory_update 修订。\n\n"
    )
    _emit("SessionStart", header + text)
    return 0


def pre_compact(payload: dict) -> int:
    """PreCompact cannot inject context (its hookSpecificOutput is rejected).

    Plain stdout is consumed by some builds as extra compaction instructions, so
    we emit a short, safe nudge and never emit JSON. The guaranteed path is the
    SessionStart hook with source=compact, which re-injects the briefing into the
    fresh window.
    """
    _apply_cwd(payload)
    sys.stdout.write(
        "压缩摘要时请保留：当前任务与进度、本轮决策及原因、未解决问题与下一步、"
        "新发现的项目约束或坑。这些内容也应写入 SuperMemory 长期记忆。\n"
    )
    sys.stdout.flush()
    return 0


def session_end(payload: dict) -> int:
    from .core import MemoryStore

    _apply_cwd(payload)
    with MemoryStore() as store:
        store.maintenance()
    return 0


EVENTS = {
    "session-start": session_start,
    "pre-compact": pre_compact,
    "session-end": session_end,
}


def run(event: str) -> int:
    if event not in EVENTS:
        print(f"unknown hook event: {event}", file=sys.stderr)
        return 2
    try:
        return EVENTS[event](_read_payload())
    except Exception:  # noqa: BLE001 - a hook must never break the session
        traceback.print_exc(file=sys.stderr)
        return 0
