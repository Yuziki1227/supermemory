"""Token-economical rendering for tool output, CLI and hooks."""

from __future__ import annotations

import time
from typing import Iterable

from .text import estimate_tokens

def _date(ts: float | None) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _explain_line(item) -> str:
    """One-line decomposition of the attention score for a recall result."""
    features = getattr(item, "features", None) or {}
    pin = 1 if features.get("pinned") else 0
    return (
        "    ↳ "
        f"lex {features.get('lexical', 0.0):.2f} · "
        f"vec {features.get('vector', 0.0):.2f} · "
        f"imp {features.get('importance', 0.0):.2f} · "
        f"rec {features.get('recency', 0.0):.2f} · "
        f"rein {features.get('reinforcement', 0.0):.2f} · "
        f"pin {pin} · score {getattr(item, 'score', 0.0):.3f} · "
        f"attn {getattr(item, 'attention', 0.0):.4f}"
    )


def render_recall(
    results: list,
    query: str = "",
    budget: int | None = None,
    explain: bool = False,
) -> str:
    if not results:
        suffix = f'（查询: "{query}"）' if query else ""
        return f"没有命中记忆{suffix}。可以用 memory_remember 记录新信息。"

    lines: list[str] = []
    header = f"记忆 {len(results)} 条"
    if query:
        header += f' · 查询 "{query}"'
    if budget:
        header += f" · 预算 {budget}t"
    lines.append(header)
    lines.append("")

    used = 0
    for item in results:
        memory = item.memory
        confidence = []
        if memory.kind:
            confidence.append(memory.kind)
        if getattr(item, "score", None) is not None:
            confidence.append(f"{item.score:.2f}")
        if memory.pinned:
            confidence.append("pinned")
        tags = (" #" + " #".join(memory.tags)) if memory.tags else ""
        source = f" ←{memory.source}" if memory.source else ""
        lines.append(f"{memory.label} [{' '.join(confidence)}] {memory.title}{tags}{source}")

        body = memory.summary or memory.content[:200]
        body = " ".join(body.split())
        lines.append(f"    {body}")
        used += estimate_tokens(body) + 24
        if explain:
            line = _explain_line(item)
            lines.append(line)
            used += estimate_tokens(line)
        if budget and used >= budget:
            lines.append("    …（已达令牌预算，剩余结果省略）")
            break

    return "\n".join(lines)


def render_records(memories: list) -> str:
    if not memories:
        return "未找到对应记忆。"
    blocks: list[str] = []
    for memory in memories:
        meta = [
            memory.kind,
            _date(memory.updated_at),
            f"importance {memory.importance:.2f}",
        ]
        if memory.tags:
            meta.append("tags: " + ", ".join(memory.tags))
        if memory.namespace:
            meta.append(f"ns: {memory.namespace}")
        block = [
            f"{memory.label} · {memory.title}",
            " · ".join(meta),
            "-" * 24,
            memory.content,
        ]
        if memory.superseded_by:
            block.insert(1, f"⚠ 已被 M{memory.superseded_by} 修订（历史版本）")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def render_remember(result) -> str:
    lines: list[str] = []
    if result.memories:
        labels = ", ".join(m.label for m in result.memories)
        lines.append(f"✓ 已记忆 {len(result.memories)} 条: {labels}")
        for memory in result.memories[:5]:
            lines.append(f"  {memory.label} [{memory.kind}] {memory.title}")
        if result.chunked:
            lines.append("  （内容较长，已自动分块存储）")
    if result.duplicates:
        labels = ", ".join(m.label for m in result.duplicates)
        lines.append(f"= 已存在相同内容，未重复写入: {labels}")
    if result.near_duplicates:
        lines.append("⚠ 检测到相似记忆，若为修订请用 memory_update(id=…)：")
        for hint in result.near_duplicates[:3]:
            lines.append(
                f"  {hint['label']} (相似度 {hint['similarity']:.2f}) {hint['title']}"
            )
    return "\n".join(lines) or "没有写入任何内容。"


def render_briefing(
    *,
    namespace: str,
    namespaces: list[str] | None,
    sections: list[tuple[str, list]],
    relevant: list,
    stats: dict,
    budget: int,
) -> str:
    """Compose the session hand-off document within a token budget."""
    scope = namespace if namespaces is None else " + ".join(namespaces)
    lines: list[str] = [
        f"# SuperMemory 简报 · {scope}",
        f"有效记忆 {stats.get('current', 0)} 条 · 已修订 {stats.get('superseded', 0)} 条"
        f" · 库大小 {_human_bytes(stats.get('db_bytes', 0))}",
    ]
    used = estimate_tokens("\n".join(lines))
    shown: set[int] = set()

    def add_section(title: str, memories: Iterable, limit_lines: int = 6) -> None:
        nonlocal used
        entries = [m for m in memories if m.id not in shown]
        if not entries:
            return
        heading = f"\n## {title}"
        heading_cost = estimate_tokens(heading)
        if used + heading_cost > budget:
            return
        lines.append(heading)
        used += heading_cost
        for memory in entries[:limit_lines]:
            shown.add(memory.id)
            tags = (" #" + " #".join(memory.tags)) if memory.tags else ""
            prefix = "★ " if memory.pinned else ""
            date = _date(memory.updated_at)
            title = " ".join((memory.title or "").split())
            summary = " ".join((memory.summary or "").split())
            if title and summary and title not in summary and summary not in title:
                text = f"{title} — {summary}"
            else:
                text = summary or title
            text = text[:220]
            line = f"- {prefix}{memory.label} [{memory.kind}] {text}{tags} ({date})"
            cost = estimate_tokens(line) + 1
            if used + cost > budget:
                lines.append("- …（超出预算，其余省略）")
                return
            lines.append(line)
            used += cost

    for title, memories in sections:
        add_section(title, memories)

    if relevant:
        entries = [(item.memory, item.score) for item in relevant if item.memory.id not in shown]
        if entries:
            heading = "\n## 与本次任务相关"
            if used + estimate_tokens(heading) < budget:
                lines.append(heading)
                used += estimate_tokens(heading)
                for memory, score in entries:
                    text = " ".join((memory.summary or memory.title).split())[:200]
                    line = f"- {memory.label} [{memory.kind} {score:.2f}] {text}"
                    cost = estimate_tokens(line) + 1
                    if used + cost > budget:
                        break
                    lines.append(line)
                    used += cost

    lines.append(
        "\n使用约定：新决策/约束/偏好/踩坑 → memory_remember；"
        "查历史 → memory_recall；修订旧记忆 → memory_update（不要新建重复项）。"
    )
    return "\n".join(lines)


def render_stats(stats: dict) -> str:
    kinds = ", ".join(f"{k}×{v}" for k, v in stats.get("kinds", {}).items()) or "—"
    namespaces = ", ".join(f"{k}×{v}" for k, v in stats.get("namespaces", {}).items()) or "—"
    return "\n".join(
        [
            f"当前命名空间 : {stats.get('namespace')}",
            f"有效记忆     : {stats.get('current')}（历史修订 {stats.get('superseded')}，软删除 {stats.get('soft_deleted')}）",
            f"类型分布     : {kinds}",
            f"命名空间     : {namespaces}",
            f"嵌入模型     : {stats.get('embedder')}",
            f"数据库       : {_human_bytes(stats.get('db_bytes', 0))}",
            f"schema       : v{stats.get('schema_version')}",
        ]
    )


def _human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"
