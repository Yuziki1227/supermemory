"""Core behaviour tests. Run with: python -m unittest discover -s tests -v"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supermemory import config as config_mod  # noqa: E402
from supermemory.core import MemoryStore  # noqa: E402
from supermemory.render import render_recall  # noqa: E402
from supermemory.text import chunk_text, estimate_tokens, tokenize  # noqa: E402


class IsolatedStore(unittest.TestCase):
    """Each test gets its own SUPERMEMORY_HOME and namespace."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old_home = os.environ.get("SUPERMEMORY_HOME")
        self._old_ns = os.environ.get("SUPERMEMORY_NAMESPACE")
        os.environ["SUPERMEMORY_HOME"] = self._tmp.name
        os.environ["SUPERMEMORY_NAMESPACE"] = "testns"
        self.store = MemoryStore(namespace="testns")

    def tearDown(self) -> None:
        self.store.close()
        for key, value in (("SUPERMEMORY_HOME", self._old_home),
                           ("SUPERMEMORY_NAMESPACE", self._old_ns)):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()


class TestText(unittest.TestCase):
    def test_tokenize_cjk_emits_unigrams_and_bigrams(self):
        tokens = tokenize("检索层")
        self.assertIn("检", tokens)
        self.assertIn("检索", tokens)
        self.assertIn("索层", tokens)

    def test_tokenize_identifiers(self):
        tokens = tokenize("MemoryStore JSON-RPC")
        for expected in ("memorystore", "memory", "store", "json", "rpc"):
            self.assertIn(expected, tokens)

    def test_estimate_tokens_cjk_vs_ascii(self):
        self.assertGreater(estimate_tokens("记忆" * 50), estimate_tokens("a" * 50))

    def test_chunking_respects_budget_and_keeps_content(self):
        paragraph = "这是一段用于测试分块的中文内容，包含多个句子。它会重复很多次以超过分块上限。" * 40
        chunks = chunk_text(paragraph, max_tokens=120, overlap_tokens=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(estimate_tokens(chunk), 200)
        self.assertIn("分块", "".join(chunks))


class TestStore(IsolatedStore):
    def test_remember_and_recall_chinese(self):
        result = self.store.remember(
            "检索层采用混合打分：BM25 + 向量 + 时近性 + 重要性加权。",
            title="检索层选型", kind="decision", tags=["检索"],
        )
        self.assertEqual(len(result.memories), 1)
        hits = self.store.recall("检索怎么打分", limit=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0].memory.title, "检索层选型")

    def test_recall_explain_shows_feature_breakdown(self):
        self.store.remember("检索层采用混合打分。", title="检索层")
        hits = self.store.recall("检索怎么打分", limit=3)
        text = render_recall(hits, query="检索怎么打分", explain=True)
        self.assertIn("↳", text)
        self.assertIn("lex", text)
        self.assertIn("score", text)
        self.assertIn("attn", text)

    def test_unrelated_query_returns_nothing(self):
        self.store.remember("SQLite 开启 WAL 模式以支持并发读写。", title="WAL")
        self.assertEqual(self.store.recall("完全无关的园艺话题", limit=3), [])

    def test_exact_duplicate_is_not_stored_twice(self):
        first = self.store.remember("同一句话只应存一次。", title="重复测试")
        second = self.store.remember("同一句话只应存一次。", title="重复测试")
        self.assertEqual(len(second.memories), 0)
        self.assertEqual(second.duplicates[0].id, first.memories[0].id)
        self.assertEqual(self.store.stats()["current"], 1)

    def test_update_supersedes_and_keeps_history(self):
        original = self.store.remember("缓存在内存里。", title="缓存策略")
        new = self.store.update(original.memories[0].id, content="缓存在 SQLite 里，重启不丢。")
        old = self.store.get(original.memories[0].id)
        self.assertEqual(old.superseded_by, new.id)
        hits = self.store.recall("缓存", limit=5)
        self.assertEqual([h.memory.id for h in hits], [new.id])
        history = self.store.recall("缓存", limit=5, include_superseded=True)
        self.assertEqual(len({h.memory.id for h in history}), 2)

    def test_forget_soft_then_hard(self):
        memory = self.store.remember("临时信息。", title="临时").memories[0]
        self.store.forget([memory.id])
        self.assertEqual(self.store.recall("临时信息", limit=3), [])
        self.store.forget([memory.id], hard=True)
        self.assertEqual(self.store.get(memory.id), None)

    def test_namespace_isolation(self):
        self.store.remember("只在 A 项目存在的记忆。", title="A", namespace="alpha")
        self.store.remember("只在 B 项目存在的记忆。", title="B", namespace="beta")
        alpha = MemoryStore(namespace="alpha")
        try:
            titles = [h.memory.title for h in alpha.recall("项目 记忆", limit=10)]
        finally:
            alpha.close()
        self.assertIn("A", titles)
        self.assertNotIn("B", titles)

    def test_token_budget_caps_results(self):
        for i in range(8):
            self.store.remember(
                f"预算测试第 {i} 条：这是一段比较长的记忆内容，用来确认令牌预算会截断结果。"
                * 3,
                title=f"预算 {i}", tags=["预算"],
            )
        hits = self.store.recall("预算测试", limit=8, max_tokens=150)
        self.assertLessEqual(len(hits), 3)

    def test_briefing_contains_pinned_and_tasks(self):
        self.store.remember("用户要求始终使用中文。", kind="preference", pinned=True,
                            importance=0.95, title="语言偏好")
        self.store.remember("实现 MCP 服务器。", kind="task", title="MCP 任务")
        text = self.store.briefing()
        self.assertIn("语言偏好", text)
        self.assertIn("MCP 任务", text)

    def test_export_import_roundtrip(self):
        self.store.remember("导出导入测试。", title="导出")
        payload = self.store.export_json()
        self.assertEqual(len(payload["memories"]), 1)
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SUPERMEMORY_HOME"] = tmp
            other = MemoryStore(namespace="other")
            try:
                inserted = other.import_json(payload)
                self.assertEqual(inserted, 1)
                self.assertEqual(other.stats()["current"], 1)
            finally:
                other.close()

    def test_config_override_changes_weights(self):
        cfg = config_mod.load_config()
        cfg["ranking"]["weights"]["lexical"] = 0.9
        self.store.config = cfg
        self.store.remember("加权配置测试。", title="权重")
        hits = self.store.recall("加权配置", limit=3)
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
