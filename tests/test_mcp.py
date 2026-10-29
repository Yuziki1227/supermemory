"""End-to-end MCP stdio tests: spawn the real server exactly as a client would."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class McpSession:
    def __init__(self, home: str) -> None:
        env = dict(os.environ)
        env["SUPERMEMORY_HOME"] = home
        env["SUPERMEMORY_NAMESPACE"] = "mcptest"
        env["PYTHONUTF8"] = "1"
        self.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "run.py"), "mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(ROOT), env=env, text=True, encoding="utf-8", bufsize=1,
        )
        self.counter = 0

    def send(self, method: str, params: dict | None = None, notify: bool = False):
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notify:
            self.counter += 1
            message["id"] = self.counter
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        line = self.proc.stdout.readline()
        return json.loads(line)

    def call(self, name: str, arguments: dict) -> tuple[str, bool]:
        response = self.send("tools/call", {"name": name, "arguments": arguments})
        result = response.get("result", {})
        text = "".join(part.get("text", "") for part in result.get("content", []))
        return text, bool(result.get("isError"))

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class TestMcpServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.session = McpSession(cls._tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.close()
        cls._tmp.cleanup()

    def test_initialize(self):
        response = self.session.send(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {},
             "clientInfo": {"name": "unittest", "version": "0"}},
        )
        result = response["result"]
        self.assertEqual(result["serverInfo"]["name"], "supermemory")
        self.assertIn("protocolVersion", result)
        self.assertIn("memory_briefing", result["instructions"])
        self.session.send("notifications/initialized", notify=True)

    def test_tools_list(self):
        response = self.session.send("tools/list")
        names = {tool["name"] for tool in response["result"]["tools"]}
        self.assertTrue({
            "memory_remember", "memory_recall", "memory_get", "memory_update",
            "memory_forget", "memory_link", "memory_briefing", "memory_stats",
        } <= names)

    def test_remember_recall_get_cycle(self):
        text, error = self.session.call(
            "memory_remember",
            {"content": "MCP 服务器通过 stdio 传 JSON-RPC，每行一条消息。",
             "title": "MCP 传输", "kind": "fact", "tags": ["mcp"]},
        )
        self.assertFalse(error)
        label = text.split("M")[1].split()[0].strip(",:")
        self.assertTrue(label.isdigit())

        text, error = self.session.call(
            "memory_recall", {"query": "JSON-RPC 怎么传", "limit": 3}
        )
        self.assertFalse(error)
        self.assertIn("MCP 传输", text)
        self.assertIn(f"M{label}", text)

        text, error = self.session.call("memory_get", {"ids": [f"M{label}"]})
        self.assertFalse(error)
        self.assertIn("每行一条消息", text)

    def test_update_and_supersede(self):
        text, error = self.session.call(
            "memory_remember", {"content": "响应必须立即 flush。", "title": "flush 规则"}
        )
        self.assertFalse(error)
        memory_id = text.split("M")[1].split()[0].strip(",:")
        text, error = self.session.call(
            "memory_update", {"id": f"M{memory_id}", "content": "响应必须立即 flush，否则客户端会挂起。"}
        )
        self.assertFalse(error)
        self.assertIn("已更新", text)
        text, _ = self.session.call("memory_get", {"ids": [f"M{memory_id}"]})
        self.assertIn("已被", text)

    def test_json_format(self):
        self.session.call(
            "memory_remember", {"content": "JSON 输出用于程序化处理。", "title": "JSON 格式"}
        )
        text, error = self.session.call(
            "memory_recall", {"query": "JSON 输出", "limit": 2, "format": "json"}
        )
        self.assertFalse(error)
        payload = json.loads(text)
        self.assertIn("results", payload)
        self.assertTrue(payload["results"])

    def test_unknown_tool_is_tool_error_not_crash(self):
        text, error = self.session.call("memory_nope", {})
        self.assertTrue(error)
        # server still answers afterwards
        response = self.session.send("ping")
        self.assertEqual(response["result"], {})

    def test_missing_argument_is_graceful(self):
        text, error = self.session.call("memory_remember", {})
        self.assertTrue(error)
        self.assertIn("content", text)


if __name__ == "__main__":
    unittest.main()
