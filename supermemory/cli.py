"""Command line interface: ``python -m supermemory <command>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from . import config as config_mod
from .core import MemoryStore, parse_id
from .render import render_recall, render_records, render_remember, render_stats


# --------------------------------------------------------------------------- #


def _store(args) -> MemoryStore:
    return MemoryStore(namespace=getattr(args, "namespace", None))


def _read_content(args) -> str:
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8")
    parts = list(getattr(args, "content", None) or [])
    if parts:
        return " ".join(parts) if len(parts) > 1 else parts[0]
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("请提供内容，或用 --file / 管道输入")


def _json_out(data) -> None:
    sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_init(args) -> int:
    with _store(args) as store:
        stats = store.stats()
    print(f"✓ 记忆库已就绪: {config_mod.db_path()}")
    print(f"  命名空间: {stats['namespace']} · 现有记忆 {stats['current']} 条")
    return 0


def cmd_remember(args) -> int:
    content = _read_content(args)
    with _store(args) as store:
        result = store.remember(
            content,
            title=args.title or "",
            kind=args.kind,
            tags=args.tag,
            importance=args.importance,
            confidence=args.confidence,
            source=args.source or "cli",
            source_ref=args.source_ref or "",
            pinned=args.pinned,
            chunk=not args.no_chunk,
            ttl_days=args.ttl_days,
            namespace=args.namespace,
        )
    print(render_remember(result))
    print("\n提示: 用 'supermemory recall \"关键词\"' 验证检索效果。")
    return 0


def cmd_recall(args) -> int:
    query = " ".join(args.query) if args.query else ""
    with _store(args) as store:
        results = store.recall(
            query,
            limit=args.limit,
            kinds=args.kind or None,
            tags=args.tag or None,
            max_tokens=args.max_tokens,
            include_superseded=args.all,
            mmr=not args.no_mmr,
        )
        budget = args.max_tokens or store.config["recall"].get("max_tokens")
    if args.json:
        _json_out([item.to_dict(include_content=False) for item in results])
    else:
        print(render_recall(results, query=query, budget=budget))
        if results:
            print(f"\n(完整内容: supermemory get {results[0].memory.label})")
    return 0


def cmd_get(args) -> int:
    with _store(args) as store:
        memories = store.get([parse_id(i) for i in args.ids])
    memories = memories if isinstance(memories, list) else [memories]
    if args.json:
        _json_out([m.to_dict() for m in memories])
    elif not memories:
        print("未找到对应记忆。")
    else:
        print(render_records(memories))
    return 0


def cmd_update(args) -> int:
    changes = {}
    for field in ("content", "title", "kind", "importance", "confidence", "source", "source_ref"):
        value = getattr(args, field, None)
        if value is not None:
            changes[field] = value
    if args.tag is not None:
        changes["tags"] = args.tag
    if args.pinned is not None:
        changes["pinned"] = args.pinned
    if not changes:
        raise SystemExit("没有需要修改的字段")
    with _store(args) as store:
        memory = store.update(parse_id(args.id), **changes)
    print(f"✓ 已更新 {args.id} → {memory.label}（旧版本保留为历史）")
    print(f"  [{memory.kind}] {memory.title}")
    return 0


def cmd_forget(args) -> int:
    with _store(args) as store:
        count = store.forget([parse_id(i) for i in args.ids], hard=args.hard)
    print(f"✓ 已{'永久删除' if args.hard else '软删除（可恢复）'} {count} 条记忆")
    return 0


def cmd_link(args) -> int:
    with _store(args) as store:
        store.link(args.src, args.dst, args.relation, args.weight)
    print(f"✓ {args.src} --{args.relation}--> {args.dst}")
    return 0


def cmd_list(args) -> int:
    with _store(args) as store:
        memories = store.list_memories(
            kind=args.kind, namespace=args.namespace,
            limit=args.limit, include_superseded=args.all,
        )
    if args.json:
        _json_out([m.to_dict(include_content=False) for m in memories])
        return 0
    if not memories:
        print("（空）")
        return 0
    print(f"{'ID':>7}  {'类型':<10} {'重要':<5} 标题")
    print("-" * 72)
    for memory in memories:
        pin = "★" if memory.pinned else " "
        date = time.strftime("%m-%d", time.localtime(memory.updated_at))
        print(
            f"{memory.label:>7}{pin} {memory.kind:<10} {memory.importance:<5.2f}"
            f" {memory.title[:44]}  ({date})"
        )
    return 0


def cmd_briefing(args) -> int:
    with _store(args) as store:
        text = store.briefing(query=args.query or "", max_tokens=args.max_tokens,
                              namespace=args.namespace)
    print(text)
    return 0


def cmd_stats(args) -> int:
    with _store(args) as store:
        stats = store.stats()
    if args.json:
        _json_out(stats)
    else:
        print(render_stats(stats))
    return 0


def cmd_export(args) -> int:
    with _store(args) as store:
        payload = store.export_json(args.path)
    print(f"✓ 已导出 {len(payload['memories'])} 条记忆 → {args.path}")
    return 0


def cmd_import(args) -> int:
    payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
    if payload.get("format") != "supermemory-export":
        raise SystemExit("不是 SuperMemory 导出文件")
    with _store(args) as store:
        inserted = store.import_json(payload)
    print(f"✓ 已导入 {inserted} 条记忆（跳过 {len(payload.get('memories', [])) - inserted} 条已存在的）")
    return 0


def cmd_reembed(args) -> int:
    with _store(args) as store:
        if args.provider:
            cfg = config_mod.load_config()
            cfg["embedder"]["provider"] = args.provider
            config_mod.save_config(cfg)
        count = store.reembed()
        name = store.embedder.name
    print(f"✓ 已用 {name} 重新计算 {count} 条记忆的向量")
    return 0


def cmd_maintenance(args) -> int:
    with _store(args) as store:
        result = store.maintenance(
            hard_delete_after_days=args.hard_delete_after_days,
            keep_events=args.keep_events,
        )
    print(f"✓ 维护完成: 永久删除 {result['purged']} 条，事件日志保留 {result['events_kept']} 条")
    return 0


def cmd_config(args) -> int:
    if args.set:
        key, _, value = args.set.partition("=")
        if not key or not _:
            raise SystemExit("用法: --set key=value（点号分隔，如 ranking.temperature=0.4）")
        cfg = config_mod.load_config()
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = value
        node[parts[-1]] = parsed
        config_mod.save_config(cfg)
        print(f"✓ {key} = {parsed!r}")
        return 0
    cfg = config_mod.load_config(reload=True)
    print(f"配置文件: {config_mod.config_path()}")
    _json_out(cfg)
    return 0


def cmd_doctor(args) -> int:
    import platform
    import sqlite3

    ok = True
    print("SuperMemory doctor")
    print(f"  版本      : {__version__}")
    print(f"  Python    : {platform.python_version()} ({sys.executable})")
    print(f"  SQLite    : {sqlite3.sqlite_version}")
    try:
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        print("  FTS5      : ✓")
    except sqlite3.Error as exc:
        ok = False
        print(f"  FTS5      : ✗ ({exc})")
    try:
        import numpy
        print(f"  numpy     : {numpy.__version__}")
    except ImportError:
        ok = False
        print("  numpy     : ✗ 未安装（pip install numpy）")
    print(f"  HOME      : {config_mod.home_dir()}")
    print(f"  数据库    : {config_mod.db_path()}")
    try:
        with MemoryStore() as store:
            stats = store.stats()
        print(f"  记忆      : {stats['current']} 条有效 · {stats['superseded']} 条历史")
        print(f"  命名空间  : {stats['namespace']}")
        print(f"  嵌入模型  : {stats['embedder']}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  数据库    : ✗ {exc}")
    print("  结论      : " + ("✓ 可用" if ok else "✗ 有问题，见上"))
    return 0 if ok else 1


def cmd_mcp(args) -> int:
    from .mcp_server import main as mcp_main
    return mcp_main()


def cmd_hook(args) -> int:
    from . import hooks
    return hooks.run(args.event)


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supermemory",
        description="SuperMemory — 给 Claude / Codex 用的本地注意力记忆库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"SuperMemory {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--namespace", help="覆盖默认命名空间（默认取当前目录名；'*' 表示全部）")

    def add(name: str, **kwargs) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[common], **kwargs)

    p = add("init", help="初始化记忆库")
    p.set_defaults(func=cmd_init)

    p = add("remember", help="写入记忆")
    p.add_argument("content", nargs="*", help="记忆内容（省略则读 stdin）")
    p.add_argument("-f", "--file", help="从文件读取内容")
    p.add_argument("-t", "--title", help="标题")
    p.add_argument("-k", "--kind", default="fact",
                   choices=["fact", "decision", "preference", "task", "procedure",
                            "gotcha", "episode", "artifact", "reference"])
    p.add_argument("--tag", action="append", help="标签（可重复）")
    p.add_argument("-i", "--importance", type=float, default=0.5, help="重要度 0-1")
    p.add_argument("--confidence", type=float, default=0.8, help="置信度 0-1")
    p.add_argument("--source", help="来源描述")
    p.add_argument("--source-ref", help="来源引用（文件/提交/URL）")
    p.add_argument("--pinned", action="store_true", help="固定（每次简报必带）")
    p.add_argument("--ttl-days", type=float, help="过期天数")
    p.add_argument("--no-chunk", action="store_true", help="不做自动分块")
    p.set_defaults(func=cmd_remember)

    p = add("recall", help="检索记忆")
    p.add_argument("query", nargs="*", help="查询（省略则返回最近的）")
    p.add_argument("-k", "--limit", type=int, default=None, help="返回条数")
    p.add_argument("--kind", action="append", help="限定类型（可重复）")
    p.add_argument("--tag", action="append", help="限定标签（可重复，需全部命中）")
    p.add_argument("--max-tokens", type=int, help="输出令牌预算")
    p.add_argument("--all", action="store_true", help="包含已被修订的历史版本")
    p.add_argument("--no-mmr", action="store_true", help="关闭多样性去重")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    p.set_defaults(func=cmd_recall)

    p = add("get", help="读取完整记忆")
    p.add_argument("ids", nargs="+", help="记忆 ID，如 M12")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_get)

    p = add("update", help="修订记忆（保留历史版本）")
    p.add_argument("id")
    p.add_argument("--content")
    p.add_argument("--title")
    p.add_argument("--kind")
    p.add_argument("--tag", action="append")
    p.add_argument("--importance", type=float)
    p.add_argument("--confidence", type=float)
    p.add_argument("--source")
    p.add_argument("--source-ref")
    p.add_argument("--pinned", dest="pinned", action="store_true", default=None)
    p.add_argument("--unpinned", dest="pinned", action="store_false")
    p.set_defaults(func=cmd_update)

    p = add("forget", help="删除记忆")
    p.add_argument("ids", nargs="+")
    p.add_argument("--hard", action="store_true", help="永久删除")
    p.set_defaults(func=cmd_forget)

    p = add("link", help="建立关联")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--relation", default="related")
    p.add_argument("--weight", type=float, default=1.0)
    p.set_defaults(func=cmd_link)

    p = add("list", help="列出记忆")
    p.add_argument("--kind")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--all", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = add("briefing", help="生成会话简报")
    p.add_argument("--query", help="当前任务，用于挑选相关记忆")
    p.add_argument("--max-tokens", type=int)
    p.set_defaults(func=cmd_briefing)

    p = add("stats", help="统计信息")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_stats)

    p = add("export", help="导出为 JSON")
    p.add_argument("path")
    p.set_defaults(func=cmd_export)

    p = add("import", help="从 JSON 导入")
    p.add_argument("path")
    p.set_defaults(func=cmd_import)

    p = add("reembed", help="重算向量（换嵌入模型后用）")
    p.add_argument("--provider", choices=["local", "openai", "none"])
    p.set_defaults(func=cmd_reembed)

    p = add("maintenance", help="清理过期数据")
    p.add_argument("--hard-delete-after-days", type=float)
    p.add_argument("--keep-events", type=int)
    p.set_defaults(func=cmd_maintenance)

    p = add("config", help="查看/修改配置")
    p.add_argument("--set", help="key=value，如 ranking.temperature=0.4")
    p.set_defaults(func=cmd_config)

    p = add("doctor", help="环境自检")
    p.set_defaults(func=cmd_doctor)

    p = add("mcp", help="以 MCP stdio 服务器运行")
    p.set_defaults(func=cmd_mcp)

    p = add("hook", help="Claude Code hook 入口")
    p.add_argument("event", choices=["session-start", "pre-compact", "session-end"])
    p.set_defaults(func=cmd_hook)

    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
