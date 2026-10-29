"""A dependency-free MCP server over stdio (newline-delimited JSON-RPC 2.0).

Hand-rolled on purpose: the memory server must start on a machine that has
nothing but CPython and numpy, and it must never take the host agent down. The
implemented surface is the subset every MCP client actually exercises —
``initialize``, ``tools/list``, ``tools/call``, plus tolerant stubs for
``resources``/``prompts``/``logging``.

stdout is the protocol channel: nothing else may ever be written there. All
diagnostics go to stderr.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable

from . import __version__
from .core import MemoryStore, parse_id
from .render import render_recall, render_records, render_remember, render_stats

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05", "2024-10-07")
LATEST_PROTOCOL = SUPPORTED_PROTOCOLS[0]
SERVER_NAME = "supermemory"

INSTRUCTIONS = """\
SuperMemory is a persistent, local long-term memory shared by Claude Code and Codex.

When to use it:
- At the start of a session, call `memory_briefing` once. It returns project
  state, pinned rules, open tasks, recent decisions and (optionally) memories
  relevant to the current task. Prefer it over guessing.
- When you learn something durable — a decision and its rationale, a constraint,
  a user preference, a gotcha, a file/API contract, a task state — call
  `memory_remember` immediately. Do not wait until the end; context may be
  compacted without warning.
- Before asking the user for context that may already be known, call `memory_recall`.
- To correct or extend an existing memory, call `memory_update(id=...)`. It keeps
  the old version in history and supersedes it. Never create a near-duplicate.
- `memory_forget` soft-deletes; `memory_get` returns full text of known ids.

Conventions:
- Memories are namespaced by project automatically. Use namespace="global" for
  knowledge that must follow the user across all projects.
- kinds: fact | decision | preference | task | procedure | gotcha | episode | artifact | reference
- Set importance 0.8-1.0 for rules the user must always follow; pinned=true for
  invariants that belong in every session briefing.
- Keep each memory atomic: one claim, one decision, one task. Long documents are
  chunked automatically, but atomic memories retrieve better.
- Reply to the user in their language; the memory database handles both.
"""


# --------------------------------------------------------------------------- #
# Tool schemas
# --------------------------------------------------------------------------- #

KIND_ENUM = [
    "fact", "decision", "preference", "task",
    "procedure", "gotcha", "episode", "artifact", "reference",
]

TOOLS: list[dict] = [
    {
        "name": "memory_remember",
        "description": (
            "Store durable information in long-term memory. Call this as soon as you learn "
            "something that a future session would need: a decision and its rationale, a "
            "constraint, a user preference, a gotcha, an API/file contract, or task state. "
            "Keep memories atomic (one claim each). Returns memory ids; use them with "
            "memory_update / memory_get later."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory text. Be specific and self-contained."},
                "title": {"type": "string", "description": "Short label (<=120 chars). Omit to derive from content."},
                "kind": {"type": "string", "enum": KIND_ENUM, "default": "fact"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Retrieval tags."},
                "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5,
                               "description": "0-1. Rules the user must always follow: 0.8-1.0."},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8},
                "source": {"type": "string", "description": "Where this came from, e.g. 'conversation', 'code-review'."},
                "source_ref": {"type": "string", "description": "File path, commit, URL or session id."},
                "pinned": {"type": "boolean", "default": False,
                           "description": "Pinned memories are injected into every session briefing."},
                "namespace": {"type": "string", "description": "Defaults to the current project. Use 'global' for cross-project knowledge."},
                "chunk": {"type": "boolean", "default": True, "description": "Auto-chunk long content."},
                "ttl_days": {"type": "number", "description": "Optional expiry in days for time-bound facts."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_recall",
        "description": (
            "Search long-term memory. Hybrid retrieval (BM25 + vector + recency + importance) "
            "with an attention-weighted, diversity-aware ranking and a token budget. Use it "
            "whenever prior context could change your answer. Returns compact entries with ids, "
            "titles, summaries and scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you want to know. Natural language is fine. Empty = recent memories."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 8},
                "kinds": {"type": "array", "items": {"type": "string", "enum": KIND_ENUM}},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Require all of these tags."},
                "max_tokens": {"type": "integer", "minimum": 100, "maximum": 20000, "description": "Output budget."},
                "namespace": {"type": "string", "description": "Default: current project + global. '*' searches everything."},
                "include_superseded": {"type": "boolean", "default": False,
                                       "description": "Include memory versions that have been superseded."},
                "explain": {"type": "boolean", "default": False,
                            "description": "Append the per-feature score breakdown (lexical, vector, importance, recency, reinforcement, pinned) to each result."},
                "format": {"type": "string", "enum": ["text", "json"], "default": "text"},
            },
            "required": [],
        },
    },
    {
        "name": "memory_get",
        "description": "Fetch the full text of specific memories by id (e.g. ['M12','M7']).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": ["string", "integer"]}, "minItems": 1},
                "format": {"type": "string", "enum": ["text", "json"], "default": "text"},
            },
            "required": ["ids"],
        },
    },
    {
        "name": "memory_update",
        "description": (
            "Revise an existing memory. Writes a new version and marks the old one superseded "
            "(history is retained). Use this instead of remembering a near-duplicate when you "
            "are correcting, extending or replacing an existing fact. Returns the new id."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": ["string", "integer"], "description": "Memory id, e.g. 'M12'."},
                "content": {"type": "string"},
                "title": {"type": "string"},
                "kind": {"type": "string", "enum": KIND_ENUM},
                "tags": {"type": "array", "items": {"type": "string"}},
                "importance": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "pinned": {"type": "boolean"},
                "source": {"type": "string"},
                "source_ref": {"type": "string"},
                "namespace": {"type": "string"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "memory_forget",
        "description": (
            "Remove memories. Soft-delete by default (recoverable, excluded from retrieval, "
            "purged after the retention window); hard=true deletes permanently."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ids": {"type": "array", "items": {"type": ["string", "integer"]}, "minItems": 1},
                "hard": {"type": "boolean", "default": False},
            },
            "required": ["ids"],
        },
    },
    {
        "name": "memory_link",
        "description": "Create a typed relation between two memories (e.g. 'refines', 'contradicts', 'depends-on').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "src": {"type": ["string", "integer"]},
                "dst": {"type": ["string", "integer"]},
                "relation": {"type": "string", "default": "related"},
                "weight": {"type": "number", "default": 1.0},
            },
            "required": ["src", "dst"],
        },
    },
    {
        "name": "memory_briefing",
        "description": (
            "Session hand-off document: pinned rules, open tasks, key decisions, preferences, "
            "recent updates, and — if you pass a query — the memories most relevant to the "
            "current task. Call once at session start. Token-budgeted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Current task or goal, optional."},
                "max_tokens": {"type": "integer", "minimum": 200, "maximum": 4000, "default": 900},
                "namespace": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "memory_stats",
        "description": "Health and size of the memory database.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["text", "json"], "default": "text"},
            },
            "required": [],
        },
    },
    {
        "name": "memory_export",
        "description": "Export the whole memory database to a JSON file (portable backup).",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or cwd-relative file path."}},
            "required": ["path"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Argument coercion (models routinely send "8" where 8 is expected)
# --------------------------------------------------------------------------- #


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _require(args: dict, name: str) -> Any:
    if name not in args or args[name] in (None, ""):
        raise ValueError(f"missing required argument: {name}")
    return args[name]


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #


def _tool_remember(store: MemoryStore, args: dict) -> str:
    ttl_value = args.get("ttl_days")
    ttl_days = _as_float(ttl_value, 0.0) or None if ttl_value not in (None, "") else None
    result = store.remember(
        str(_require(args, "content")),
        title=str(args.get("title") or ""),
        kind=str(args.get("kind") or "fact"),
        tags=_as_list(args.get("tags")),
        importance=_as_float(args.get("importance"), 0.5),
        confidence=_as_float(args.get("confidence"), 0.8),
        source=str(args.get("source") or ""),
        source_ref=str(args.get("source_ref") or ""),
        pinned=_as_bool(args.get("pinned")),
        namespace=args.get("namespace") or None,
        chunk=_as_bool(args.get("chunk"), True),
        ttl_days=ttl_days,
    )
    return render_remember(result)


def _tool_recall(store: MemoryStore, args: dict) -> str:
    results = store.recall(
        str(args.get("query") or ""),
        limit=_as_int(args.get("limit"), 8),
        kinds=_as_list(args.get("kinds")) or None,
        tags=_as_list(args.get("tags")) or None,
        namespace=args.get("namespace") or None,
        max_tokens=_as_int(args.get("max_tokens")),
        include_superseded=_as_bool(args.get("include_superseded")),
    )
    if str(args.get("format") or "text") == "json":
        return json.dumps(
            {"query": args.get("query") or "", "count": len(results),
             "results": [item.to_dict(include_content=False) for item in results]},
            ensure_ascii=False, indent=2,
        )
    budget = _as_int(args.get("max_tokens")) or store.config["recall"].get("max_tokens")
    return render_recall(
        results,
        query=str(args.get("query") or ""),
        budget=budget,
        explain=_as_bool(args.get("explain")),
    )


def _tool_get(store: MemoryStore, args: dict) -> str:
    ids = _as_list(_require(args, "ids"))
    memories = store.get(ids)
    if not isinstance(memories, list):
        memories = [memories] if memories else []
    if str(args.get("format") or "text") == "json":
        return json.dumps([m.to_dict() for m in memories], ensure_ascii=False, indent=2)
    return render_records(memories)


def _tool_update(store: MemoryStore, args: dict) -> str:
    memory_id = parse_id(_require(args, "id"))
    changes = {k: v for k, v in args.items() if k != "id" and v is not None}
    for key in ("importance", "confidence"):
        if key in changes:
            changes[key] = _as_float(changes[key], 0.5)
    if "pinned" in changes:
        changes["pinned"] = _as_bool(changes["pinned"])
    if "tags" in changes:
        changes["tags"] = _as_list(changes["tags"])
    new = store.update(memory_id, **changes)
    return (
        f"✓ 已更新 M{memory_id} → {new.label}\n"
        f"  [{new.kind}] {new.title}\n"
        f"  旧版本已保留并标记为 superseded。"
    )


def _tool_forget(store: MemoryStore, args: dict) -> str:
    ids = _as_list(_require(args, "ids"))
    count = store.forget(ids, hard=_as_bool(args.get("hard")))
    mode = "永久删除" if _as_bool(args.get("hard")) else "软删除（可恢复）"
    return f"✓ {mode} {count} 条记忆"


def _tool_link(store: MemoryStore, args: dict) -> str:
    store.link(
        _require(args, "src"),
        _require(args, "dst"),
        relation=str(args.get("relation") or "related"),
        weight=_as_float(args.get("weight"), 1.0),
    )
    return "✓ 已建立关联"


def _tool_briefing(store: MemoryStore, args: dict) -> str:
    return store.briefing(
        query=str(args.get("query") or ""),
        max_tokens=_as_int(args.get("max_tokens")),
        namespace=args.get("namespace") or None,
    )


def _tool_stats(store: MemoryStore, args: dict) -> str:
    stats = store.stats()
    if str(args.get("format") or "text") == "json":
        return json.dumps(stats, ensure_ascii=False, indent=2)
    return render_stats(stats)


def _tool_export(store: MemoryStore, args: dict) -> str:
    path = str(_require(args, "path"))
    payload = store.export_json(path)
    return f"✓ 已导出 {len(payload['memories'])} 条记忆到 {path}"


HANDLERS: dict[str, Callable[[MemoryStore, dict], str]] = {
    "memory_remember": _tool_remember,
    "memory_recall": _tool_recall,
    "memory_get": _tool_get,
    "memory_update": _tool_update,
    "memory_forget": _tool_forget,
    "memory_link": _tool_link,
    "memory_briefing": _tool_briefing,
    "memory_stats": _tool_stats,
    "memory_export": _tool_export,
}


# --------------------------------------------------------------------------- #
# JSON-RPC plumbing
# --------------------------------------------------------------------------- #


class Server:
    def __init__(self, store_factory: Callable[[], MemoryStore] = MemoryStore) -> None:
        self._store_factory = store_factory
        self._store: MemoryStore | None = None
        self._initialized = False

    @property
    def store(self) -> MemoryStore:
        if self._store is None:
            self._store = self._store_factory()
        return self._store

    def handle(self, message: dict) -> dict | None:
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}
        is_notification = "id" not in message

        if method is None:
            # a response to something we never sent; ignore
            return None

        try:
            if method == "initialize":
                result = self._initialize(params)
                self._initialized = True
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                result = self._call_tool(params)
            elif method == "resources/list":
                result = {"resources": []}
            elif method == "resources/templates/list":
                result = {"resourceTemplates": []}
            elif method == "prompts/list":
                result = {"prompts": []}
            elif method == "logging/setLevel":
                result = {}
            elif method in (
                "notifications/initialized",
                "notifications/cancelled",
                "notifications/progress",
            ):
                return None
            else:
                return None if is_notification else _error(msg_id, -32601, f"method not found: {method}")
        except Exception as exc:  # noqa: BLE001 - JSON-RPC must always answer
            traceback.print_exc(file=sys.stderr)
            if is_notification:
                return None
            return _error(msg_id, -32603, f"{type(exc).__name__}: {exc}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _initialize(self, params: dict) -> dict:
        requested = str(params.get("protocolVersion") or "")
        version = requested if requested in SUPPORTED_PROTOCOLS else LATEST_PROTOCOL
        return {
            "protocolVersion": version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {},
                "prompts": {},
            },
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
            "instructions": INSTRUCTIONS,
        }

    def _call_tool(self, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name not in HANDLERS:
            return {
                "content": [{"type": "text", "text": f"unknown tool: {name}"}],
                "isError": True,
            }
        if not isinstance(arguments, dict):
            arguments = {}
        try:
            text = HANDLERS[name](self.store, arguments)
        except (ValueError, KeyError, TypeError) as exc:
            return {"content": [{"type": "text", "text": f"错误: {exc}"}], "isError": True}
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            return {
                "content": [{"type": "text", "text": f"内部错误: {type(exc).__name__}: {exc}"}],
                "isError": True,
            }
        return {"content": [{"type": "text", "text": text}], "isError": False}


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _write(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve() -> int:
    """Run the stdio server until stdin closes."""
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    server = Server()
    print(f"[supermemory] mcp server v{__version__} ready", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            _write(_error(None, -32700, "parse error"))
            continue

        if isinstance(message, list):  # legacy JSON-RPC batch
            responses = [
                response for response in (server.handle(item) for item in message if isinstance(item, dict))
                if response is not None
            ]
            if responses:
                _write(responses)
            continue

        if not isinstance(message, dict):
            _write(_error(None, -32600, "invalid request"))
            continue

        response = server.handle(message)
        if response is not None:
            _write(response)

    if server._store is not None:
        server._store.close()
    return 0


def main() -> int:
    try:
        return serve()
    except KeyboardInterrupt:
        return 0
    except Exception:  # noqa: BLE001 - last-resort logging
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
