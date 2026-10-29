"""Offline retrieval benchmark.

A cases file is a self-contained corpus plus labelled queries, so a run is
reproducible on any machine:

    {"type": "memory", "id": "gate", "content": "...", "kind": "decision"}
    {"type": "query", "query": "怎么避免召回无关记忆", "expect": ["gate"]}

``expect`` may be empty: the query is then a negative case that must return
nothing. Results are scored with Recall@k, MRR and nDCG@k; the token column
approximates what the same results would cost in an agent's context window.

The benchmark never touches the real memory database: it seeds a temporary
SQLite file and uses ``DEFAULT_CONFIG`` unless ``--config`` is given.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config as config_mod
from .core import MemoryStore
from .text import estimate_tokens


@dataclass
class Case:
    label: str
    content: str
    title: str = ""
    kind: str = "fact"
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.8
    pinned: bool = False
    chunk: bool = False


@dataclass
class Query:
    query: str
    expect: list[str]
    note: str = ""


@dataclass
class BenchReport:
    cases_path: str
    memories: int
    queries: int
    positive: int
    negative: int
    limit: int
    max_tokens: int
    embedder: str
    duration_ms: int
    metrics: dict
    per_query: list[dict]
    db_path: str = ""

    def to_dict(self) -> dict:
        metrics = {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in self.metrics.items()
        }
        return {
            "cases": self.cases_path,
            "memories": self.memories,
            "queries": self.queries,
            "positive": self.positive,
            "negative": self.negative,
            "limit": self.limit,
            "max_tokens": self.max_tokens,
            "embedder": self.embedder,
            "duration_ms": self.duration_ms,
            "metrics": metrics,
            "per_query": self.per_query,
            "db_path": self.db_path,
        }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _tags(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError("tags must be a string or a list of strings")
    return [str(item).strip() for item in value if str(item).strip()]


def _flag(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _parse_memory(row: dict, where: str) -> Case:
    label = str(row.get("id") or "").strip()
    if not label:
        raise ValueError(f"{where}: memory entry needs a non-empty 'id'")
    content = str(row.get("content") or "").strip()
    if not content:
        raise ValueError(f"{where}: memory '{label}' needs non-empty 'content'")
    try:
        importance = float(row.get("importance", 0.5))
        confidence = float(row.get("confidence", 0.8))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: memory '{label}' has a non-numeric score") from exc
    return Case(
        label=label,
        content=content,
        title=str(row.get("title") or ""),
        kind=str(row.get("kind") or "fact"),
        tags=_tags(row.get("tags")),
        importance=importance,
        confidence=confidence,
        pinned=_flag(row.get("pinned", False)),
        chunk=_flag(row.get("chunk", False)),
    )


def _parse_query(row: dict, where: str, known: set[str]) -> Query:
    query = str(row.get("query") or "").strip()
    expect_raw = row.get("expect") or []
    if isinstance(expect_raw, str):
        expect_raw = [expect_raw]
    if not isinstance(expect_raw, (list, tuple)):
        raise ValueError(f"{where}: 'expect' must be a label or a list of labels")
    expect = list(dict.fromkeys(
        str(item).strip() for item in expect_raw if str(item).strip()
    ))
    unknown = sorted(set(expect) - known)
    if unknown:
        raise ValueError(
            f"{where}: query expects unknown memories: {', '.join(unknown)}"
        )
    return Query(query=query, expect=expect, note=str(row.get("note") or ""))


def load_cases(path: str | Path) -> tuple[list[Case], list[Query]]:
    """Load a JSONL cases file (or a single JSON object with both lists)."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read cases {target}: {exc}") from exc

    raw_memories: list[dict] = []
    raw_queries: list[dict] = []
    if target.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except ValueError as exc:
            raise ValueError(f"{target}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{target}: JSON cases must be an object")
        memories_value = payload.get("memories") or []
        queries_value = payload.get("queries") or []
        if not isinstance(memories_value, list) or not isinstance(queries_value, list):
            raise ValueError(f"{target}: 'memories' and 'queries' must be lists")
        raw_memories = list(memories_value)
        raw_queries = list(queries_value)
    else:
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"{target}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{target}:{lineno}: each line must be a JSON object")
            kind = row.get("type")
            if kind == "memory":
                raw_memories.append(row)
            elif kind == "query":
                raw_queries.append(row)
            else:
                raise ValueError(
                    f"{target}:{lineno}: 'type' must be 'memory' or 'query'"
                )

    memories: list[Case] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_memories, 1):
        case = _parse_memory(row, f"{target}:memory#{index}")
        if case.label in seen:
            raise ValueError(f"{target}: duplicate memory id '{case.label}'")
        seen.add(case.label)
        memories.append(case)

    queries = [
        _parse_query(row, f"{target}:query#{index}", seen)
        for index, row in enumerate(raw_queries, 1)
    ]
    if not memories:
        raise ValueError(f"{target}: no memories defined")
    if not queries:
        raise ValueError(f"{target}: no queries defined")
    return memories, queries


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_query(ranked: list[str], expected: list[str], k: int) -> dict:
    """Recall@k, reciprocal rank and binary nDCG@k for one query."""
    top = ranked[:k]
    relevant = set(expected)
    hits = {label for label in top if label in relevant}
    recall = len(hits) / len(relevant) if relevant else 0.0

    rr = 0.0
    for position, label in enumerate(top, 1):
        if label in relevant:
            rr = 1.0 / position
            break

    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, label in enumerate(top, 1)
        if label in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg else 0.0
    return {"recall": recall, "rr": rr, "ndcg": ndcg}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_bench(
    cases_path: str | Path,
    *,
    limit: int = 8,
    max_tokens: int = 20000,
    config_path: str | Path | None = None,
    namespace: str = "bench",
    keep: bool = False,
) -> BenchReport:
    """Seed a temporary store from the cases file and score every query."""
    memories, queries = load_cases(cases_path)
    config = (
        config_mod.load_config_file(config_path)
        if config_path
        else copy.deepcopy(config_mod.DEFAULT_CONFIG)
    )
    started = time.perf_counter()
    temp_dir = Path(tempfile.mkdtemp(prefix="supermemory-bench-"))
    db_path = temp_dir / "bench.db"
    store = MemoryStore(namespace=namespace, config=config, path=db_path)
    id_to_label: dict[int, str] = {}
    rows: list[dict] = []
    embedder_name = store.embedder.name

    try:
        for case in memories:
            result = store.remember(
                case.content,
                title=case.title,
                kind=case.kind,
                tags=case.tags,
                importance=case.importance,
                confidence=case.confidence,
                source="bench",
                source_ref=case.label,
                pinned=case.pinned,
                namespace=namespace,
                chunk=case.chunk,
                dedupe=False,
            )
            created = result.memories or result.duplicates
            if not created:
                raise ValueError(f"memory '{case.label}' was not stored")
            for memory in created:
                id_to_label[memory.id] = case.label

        for query in queries:
            results = store.recall(
                query.query,
                limit=limit,
                namespace=namespace,
                max_tokens=max_tokens,
                touch=False,
            )
            ranked: list[str] = []
            for item in results:
                label = id_to_label.get(item.memory.id, item.memory.label)
                if label not in ranked:
                    ranked.append(label)
            tokens = sum(
                estimate_tokens(item.memory.summary or item.memory.title) + 24
                for item in results
            )
            scores = score_query(ranked, query.expect, limit)
            negative = not query.expect
            hit = scores["recall"] > 0.0
            passed = (not ranked) if negative else hit
            rows.append(
                {
                    "query": query.query,
                    "expect": query.expect,
                    "ranked": ranked[:limit],
                    "recall": round(scores["recall"], 4),
                    "rr": round(scores["rr"], 4),
                    "ndcg": round(scores["ndcg"], 4),
                    "tokens": tokens,
                    "negative": negative,
                    "hit": hit,
                    "complete": (not negative) and scores["recall"] >= 1.0,
                    "pass": passed,
                    "note": query.note,
                }
            )
    finally:
        store.close()
        if not keep:
            shutil.rmtree(temp_dir, ignore_errors=True)

    positive = [row for row in rows if not row["negative"]]
    negative = [row for row in rows if row["negative"]]
    metrics = {
        "recall_at_k": _mean([row["recall"] for row in positive]),
        "mrr": _mean([row["rr"] for row in positive]),
        "ndcg_at_k": _mean([row["ndcg"] for row in positive]),
        "negative_pass_rate": _mean([1.0 if row["pass"] else 0.0 for row in negative]),
        "negative_passed": sum(1 for row in negative if row["pass"]),
        "avg_tokens": _mean([float(row["tokens"]) for row in rows]) or 0.0,
    }
    return BenchReport(
        cases_path=str(cases_path),
        memories=len(memories),
        queries=len(queries),
        positive=len(positive),
        negative=len(negative),
        limit=limit,
        max_tokens=max_tokens,
        embedder=embedder_name,
        duration_ms=int((time.perf_counter() - started) * 1000),
        metrics=metrics,
        per_query=rows,
        db_path=str(db_path) if keep else "",
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def format_report(report: BenchReport) -> str:
    metrics = report.metrics

    def number(value, digits: int = 3) -> str:
        return "—" if value is None else f"{value:.{digits}f}"

    negative = "—"
    if report.negative:
        negative = f"{metrics['negative_passed']}/{report.negative}"

    lines = [
        "SuperMemory bench",
        f"  评测集   : {report.cases_path}",
        f"  语料     : {report.memories} 条 · 查询 {report.queries}"
        f"（{report.positive} 正例 / {report.negative} 负例）",
        f"  运行配置 : {report.embedder} · top {report.limit}"
        f" · 上限 {report.max_tokens} tokens",
        "",
        f"  Recall@{report.limit} : {number(metrics['recall_at_k'])}",
        f"  MRR        : {number(metrics['mrr'])}",
        f"  nDCG@{report.limit}   : {number(metrics['ndcg_at_k'])}",
        f"  负例通过   : {negative}",
        f"  平均令牌   : {number(metrics['avg_tokens'], 0)} t/查询",
        f"  耗时       : {report.duration_ms} ms",
    ]
    if report.db_path:
        lines.append(f"  临时库     : {report.db_path}")

    problems = [
        row
        for row in report.per_query
        if (row["negative"] and row["ranked"]) or (not row["negative"] and not row["complete"])
    ]
    lines.append("")
    if not problems:
        lines.append("全部查询符合预期。")
    else:
        lines.append(f"未完全命中 / 误召回 {len(problems)} 条：")
        for row in problems[:10]:
            expected = ", ".join(row["expect"]) or "（空）"
            actual = ", ".join(row["ranked"][:5]) or "（空）"
            lines.append(f'  - "{row["query"]}"')
            lines.append(
                f"    期望 {expected} → 实际 {actual} · recall {row['recall']:.2f}"
            )
        if len(problems) > 10:
            lines.append(f"  …其余 {len(problems) - 10} 条省略")
    return "\n".join(lines)
