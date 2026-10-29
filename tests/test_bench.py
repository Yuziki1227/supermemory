"""Benchmark harness tests. Run with: python -m unittest discover -s tests -v"""

from __future__ import annotations

import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supermemory.bench import format_report, load_cases, run_bench, score_query  # noqa: E402
from supermemory.cli import main as cli_main  # noqa: E402


MEMORIES = [
    {"type": "memory", "id": "wal", "title": "并发写入",
     "content": "SQLite 使用 WAL 模式支持并发读写。", "kind": "fact", "tags": ["并发"]},
    {"type": "memory", "id": "embed", "title": "嵌入",
     "content": "默认嵌入是离线哈希向量。", "kind": "fact", "tags": ["嵌入"]},
]
QUERIES = [
    {"type": "query", "query": "并发读写怎么实现", "expect": ["wal"]},
    {"type": "query", "query": "今天晚饭吃什么", "expect": []},
]


def write_cases(directory: Path, rows: list[dict]) -> Path:
    path = directory / "cases.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


class TestScoreQuery(unittest.TestCase):
    def test_single_expected_at_second_position(self):
        scores = score_query(["a", "b", "c"], ["b"], 3)
        self.assertEqual(scores["recall"], 1.0)
        self.assertAlmostEqual(scores["rr"], 0.5)
        self.assertAlmostEqual(scores["ndcg"], 1.0 / math.log2(3))

    def test_multi_expected_and_cutoff(self):
        scores = score_query(["a", "x", "b"], ["a", "b"], 2)
        self.assertAlmostEqual(scores["recall"], 0.5)  # b is outside top-2
        self.assertEqual(scores["rr"], 1.0)

    def test_empty_expected_is_zero(self):
        scores = score_query(["a"], [], 5)
        self.assertEqual(scores, {"recall": 0.0, "rr": 0.0, "ndcg": 0.0})


class TestLoadCases(unittest.TestCase):
    def test_loads_memories_and_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_cases(Path(tmp), MEMORIES + QUERIES)
            memories, queries = load_cases(path)
        self.assertEqual([m.label for m in memories], ["wal", "embed"])
        self.assertEqual(queries[0].expect, ["wal"])
        self.assertEqual(queries[1].expect, [])

    def test_unknown_expected_label_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_cases(
                Path(tmp),
                MEMORIES + [{"type": "query", "query": "x", "expect": ["nope"]}],
            )
            with self.assertRaises(ValueError) as ctx:
                load_cases(path)
        self.assertIn("nope", str(ctx.exception))

    def test_duplicate_memory_id_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_cases(Path(tmp), MEMORIES + [MEMORIES[0]] + QUERIES)
            with self.assertRaises(ValueError) as ctx:
                load_cases(path)
        self.assertIn("duplicate", str(ctx.exception))


class TestRunBench(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_cases(Path(tmp), MEMORIES + QUERIES)
            report = run_bench(path)
        self.assertEqual(report.memories, 2)
        self.assertEqual(report.queries, 2)
        self.assertEqual(report.positive, 1)
        self.assertEqual(report.negative, 1)
        self.assertEqual(report.metrics["recall_at_k"], 1.0)
        self.assertEqual(report.metrics["mrr"], 1.0)
        self.assertEqual(report.metrics["negative_passed"], 1)
        self.assertGreater(report.metrics["avg_tokens"], 0)
        self.assertIn("全部查询符合预期", format_report(report))
        json.dumps(report.to_dict())  # must be serialisable

    def test_threshold_failure_sets_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_cases(Path(tmp), MEMORIES + QUERIES)
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = cli_main(
                    ["bench", str(path), "--fail-under-recall", "1.1"]
                )
        self.assertEqual(code, 1)
        self.assertIn("未达到阈值", err.getvalue())

    def test_missing_file_is_a_user_error(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main(["bench", "does-not-exist.jsonl"])
        self.assertEqual(code, 1)
        self.assertIn("cannot read cases", err.getvalue())


if __name__ == "__main__":
    unittest.main()
