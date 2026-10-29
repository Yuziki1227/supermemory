#!/usr/bin/env python
"""One-shot installer for SuperMemory.

What it does (all merges are idempotent, every modified file is backed up):

1. Copies the package to ``~/.supermemory/app`` (or points at this repo with
   ``--dev``).
2. Registers the MCP server in Claude Code (``~/.claude.json``, user scope)
   and Codex (``~/.codex/config.toml``).
3. Adds Claude Code hooks: SessionStart injects the briefing, PreCompact asks
   the agent to persist state, SessionEnd runs housekeeping.
4. Appends a small instruction block to ``~/.claude/CLAUDE.md`` (marked and
   removable).
5. Runs ``doctor`` to prove it works.

    python install.py             # install
    python install.py --dry-run   # show what would change
    python install.py --uninstall # undo config changes (keeps the database)

Nothing here talks to the network.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
HOME = Path.home()
SM_HOME = Path(os.environ.get("SUPERMEMORY_HOME") or (HOME / ".supermemory")).expanduser()
APP = SM_HOME / "app"
BACKUP_DIR = SM_HOME / "backups"

CLAUDE_JSON = HOME / ".claude.json"
CLAUDE_SETTINGS = HOME / ".claude" / "settings.json"
CLAUDE_MD = HOME / ".claude" / "CLAUDE.md"
CODEX_CONFIG = HOME / ".codex" / "config.toml"

BLOCK_START = "<!-- supermemory:start -->"
BLOCK_END = "<!-- supermemory:end -->"
INSTRUCTION_BLOCK = f"""{BLOCK_START}
## SuperMemory（长期记忆）

本机已安装 SuperMemory MCP 服务器，记忆库跨会话、跨 Claude/Codex 共享。

- 会话开始：调用 `memory_briefing` 一次，加载项目状态、固定规则、未完成任务与关键决策。
- 学到需要长期保留的信息（决策及原因、约束、用户偏好、踩坑、接口/文件契约、任务进度）时，立即 `memory_remember`，不要等到结束——上下文随时可能被压缩。
- 询问用户之前，若可能有历史信息，先 `memory_recall`。
- 修正或扩展已有记忆用 `memory_update(id=...)`（保留历史版本）；不要新建近似重复项。
- 跨项目通用的知识用 `namespace="global"`；用户必须始终遵守的规则用 `pinned=true`。
{BLOCK_END}
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class Plan:
    """Collect actions so --dry-run can print them without touching disk."""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.actions: list[str] = []

    def log(self, message: str) -> None:
        print(("  [dry] " if self.dry_run else "  ✓ ") + message)

    def backup(self, path: Path) -> None:
        if self.dry_run or not path.exists():
            return
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = BACKUP_DIR / f"{path.name}.{stamp}.bak"
        shutil.copy2(path, target)
        print(f"    备份 → {target}")


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(plan: Plan, path: Path, data: dict) -> None:
    plan.backup(path)
    if plan.dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _mentions(value, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value.lower()
    if isinstance(value, dict):
        return any(_mentions(v, needle) for v in value.values())
    if isinstance(value, list):
        return any(_mentions(v, needle) for v in value)
    return False


def _q(path: Path | str) -> str:
    """Quote a path for a shell command line.

    Windows paths keep their backslashes: cmd.exe mangles forward-slash paths
    when it splits the command line. A quoted .cmd launcher works from cmd,
    PowerShell and Git Bash alike, which is why hooks invoke the launcher rather
    than python directly.
    """
    return f'"{path}"'


def _hook_command(launcher: Path, event: str) -> str:
    return f"{_q(launcher)} hook {event}"


def launcher_path() -> Path:
    return SM_HOME / "bin" / ("supermemory.cmd" if os.name == "nt" else "supermemory")


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #


def install_app(plan: Plan, dev: bool) -> Path:
    print("\n[1/6] 安装程序文件")
    if dev:
        plan.log(f"开发模式：直接使用仓库 {REPO}")
        return REPO

    target_pkg = APP / "supermemory"
    if APP.exists():
        plan.log(f"更新已有安装 {APP}")
        if not plan.dry_run:
            shutil.rmtree(target_pkg, ignore_errors=True)
    else:
        plan.log(f"复制到 {APP}")
    if not plan.dry_run:
        APP.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            REPO / "supermemory",
            target_pkg,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
        shutil.copy2(REPO / "run.py", APP / "run.py")
    return APP


def register_claude_mcp(plan: Plan, python: str, launcher: Path, remove: bool = False) -> None:
    print("\n[2/6] 注册 Claude Code MCP 服务器")
    data = _read_json(CLAUDE_JSON)
    if not isinstance(data.get("mcpServers"), dict):
        data["mcpServers"] = {}
    servers = data["mcpServers"]
    if remove:
        if servers.pop("supermemory", None) is not None:
            _write_json(plan, CLAUDE_JSON, data)
            plan.log(f"已从 {CLAUDE_JSON} 移除")
        else:
            plan.log("未注册，跳过")
        return
    servers["supermemory"] = {
        "type": "stdio",
        "command": str(python),
        "args": [str(launcher), "mcp"],
        "env": {"PYTHONUTF8": "1"},
    }
    _write_json(plan, CLAUDE_JSON, data)
    plan.log(f"已写入 {CLAUDE_JSON} → mcpServers.supermemory")


def register_claude_hooks(plan: Plan, launcher: Path, remove: bool = False) -> None:
    print("\n[3/6] 安装 Claude Code 钩子")
    settings = _read_json(CLAUDE_SETTINGS)
    if not isinstance(settings.get("hooks"), dict):
        settings["hooks"] = {}
    hooks = settings["hooks"]

    if remove:
        for event in ("SessionStart", "PreCompact", "SessionEnd"):
            groups = hooks.get(event)
            if isinstance(groups, list):
                hooks[event] = [g for g in groups if not _mentions(g, "supermemory")]
                if not hooks[event]:
                    hooks.pop(event, None)
        _write_json(plan, CLAUDE_SETTINGS, settings)
        plan.log(f"已从 {CLAUDE_SETTINGS} 移除 SuperMemory 钩子")
        return

    # No matchers: match every occurrence of each event. SessionStart also
    # fires after compaction (source="compact"), which is the only supported way
    # to get text in front of the model in the post-compaction window.
    wanted = {
        "SessionStart": {
            "hooks": [{"type": "command", "command": _hook_command(launcher, "session-start"),
                       "timeout": 30}],
        },
        "PreCompact": {
            "hooks": [{"type": "command", "command": _hook_command(launcher, "pre-compact"),
                       "timeout": 15}],
        },
        "SessionEnd": {
            "hooks": [{"type": "command", "command": _hook_command(launcher, "session-end"),
                       "timeout": 30}],
        },
    }
    for event, group in wanted.items():
        groups = [g for g in hooks.get(event, []) if not _mentions(g, "supermemory")]
        groups.append(group)
        hooks[event] = groups
        plan.log(f"{event} → {group['hooks'][0]['command']}")
    _write_json(plan, CLAUDE_SETTINGS, settings)


def register_codex(plan: Plan, python: str, launcher: Path, remove: bool = False) -> None:
    print("\n[4/6] 注册 Codex MCP 服务器")
    path = CODEX_CONFIG
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    marker = "[mcp_servers.supermemory]"

    if remove:
        if marker not in text:
            plan.log("未注册，跳过")
            return
        lines = text.splitlines()
        out: list[str] = []
        skipping = False
        for line in lines:
            stripped = line.strip()
            if stripped == marker:
                skipping = True
                continue
            if skipping and stripped.startswith("[") and stripped.endswith("]"):
                skipping = False
            if not skipping:
                out.append(line)
        plan.backup(path)
        if not plan.dry_run:
            path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
        plan.log(f"已从 {path} 移除")
        return

    if marker in text:
        plan.log(f"{path} 中已存在，跳过")
        return
    py = str(python).replace("\\", "/")
    launcher_text = str(launcher).replace("\\", "/")
    block = (
        f"\n{marker}\n"
        f"command = '{py}'\n"
        f'args = ["{launcher_text}", "mcp"]\n'
        f'env = {{ PYTHONUTF8 = "1" }}\n'
        f"startup_timeout_sec = 30\n"
    )
    plan.backup(path)
    if not plan.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(block)
    plan.log(f"已追加到 {path} → mcp_servers.supermemory")


def write_instructions(plan: Plan, remove: bool = False) -> None:
    print("\n[5/6] 写入代理使用说明")
    path = CLAUDE_MD
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    has_block = BLOCK_START in text and BLOCK_END in text

    if remove:
        if not has_block:
            plan.log("没有说明块，跳过")
            return
        start = text.index(BLOCK_START)
        end = text.index(BLOCK_END) + len(BLOCK_END)
        new_text = (text[:start] + text[end:]).strip() + "\n"
        plan.backup(path)
        if not plan.dry_run:
            path.write_text(new_text, encoding="utf-8")
        plan.log(f"已从 {path} 移除说明块")
        return

    if has_block:
        start = text.index(BLOCK_START)
        end = text.index(BLOCK_END) + len(BLOCK_END)
        new_text = text[:start] + INSTRUCTION_BLOCK.strip() + text[end:]
        if new_text == text:
            plan.log(f"{path} 说明块已是最新")
            return
    else:
        new_text = (text.rstrip() + "\n\n" + INSTRUCTION_BLOCK) if text.strip() else INSTRUCTION_BLOCK
    plan.backup(path)
    if not plan.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_text, encoding="utf-8")
    plan.log(f"已写入 {path}")


def verify(plan: Plan, python: str, launcher: Path) -> bool:
    print("\n[6/6] 自检")
    if plan.dry_run:
        plan.log("跳过（dry-run）")
        return True
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [python, str(launcher), "doctor"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return proc.returncode == 0


def write_launcher(plan: Plan, python: str, launcher: Path) -> None:
    """A convenience ``supermemory`` command for humans (not used by MCP)."""
    bin_dir = SM_HOME / "bin"
    if plan.dry_run:
        plan.log(f"启动器 {bin_dir / 'supermemory.cmd'}")
        return
    bin_dir.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        content = (
            "@echo off\r\n"
            f'"{python}" "{launcher}" %*\r\n'
        )
        (bin_dir / "supermemory.cmd").write_text(content, encoding="utf-8")
    else:
        content = f'#!/bin/sh\nexec "{python}" "{launcher}" "$@"\n'
        target = bin_dir / "supermemory"
        target.write_text(content, encoding="utf-8")
        target.chmod(0o755)
    plan.log(f"命令行入口 {bin_dir}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SuperMemory 安装器")
    parser.add_argument("--dev", action="store_true", help="不复制文件，直接使用当前仓库")
    parser.add_argument("--dry-run", action="store_true", help="只显示将要做的修改")
    parser.add_argument("--uninstall", action="store_true", help="移除 Claude/Codex 配置（保留数据库）")
    parser.add_argument("--purge", action="store_true", help="卸载时同时删除 ~/.supermemory")
    parser.add_argument("--no-claude", action="store_true", help="不修改 Claude Code 配置")
    parser.add_argument("--no-codex", action="store_true", help="不修改 Codex 配置")
    parser.add_argument("--no-hooks", action="store_true", help="不安装 Claude Code 钩子")
    parser.add_argument("--no-instructions", action="store_true", help="不写 CLAUDE.md 说明块")
    args = parser.parse_args(argv)

    plan = Plan(args.dry_run)
    python = sys.executable
    print(f"SuperMemory 安装器\n  Python: {python}\n  仓库  : {REPO}")

    if args.uninstall:
        print("\n卸载 SuperMemory 配置")
        if not args.no_claude:
            register_claude_mcp(plan, python, REPO / "run.py", remove=True)
            register_claude_hooks(plan, launcher_path(), remove=True)
        if not args.no_codex:
            register_codex(plan, python, REPO / "run.py", remove=True)
        if not args.no_instructions:
            write_instructions(plan, remove=True)
        if args.purge:
            print(f"\n删除 {SM_HOME}")
            if not args.dry_run:
                shutil.rmtree(SM_HOME, ignore_errors=True)
            plan.log("已删除数据目录")
        else:
            print(f"\n记忆库保留在 {SM_HOME}（用 --purge 一并删除）")
        return 0

    app = install_app(plan, args.dev)
    launcher = app / "run.py"
    write_launcher(plan, python, launcher)

    if not args.no_claude:
        register_claude_mcp(plan, python, launcher)
        if not args.no_hooks:
            register_claude_hooks(plan, launcher_path())
    if not args.no_codex:
        register_codex(plan, python, launcher)
    if not args.no_instructions:
        write_instructions(plan)

    ok = verify(plan, python, launcher)

    print("\n完成情况")
    print(f"  记忆库   : {SM_HOME / 'memory.db'}")
    print(f"  配置     : {SM_HOME / 'config.json'}（可选，覆盖默认权重）")
    print(f"  备份     : {BACKUP_DIR}")
    print("  下一步   : 重启 Claude Code / Codex 使 MCP 与钩子生效")
    print("  常用命令 : supermemory doctor | recall \"关键词\" | briefing | stats")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
