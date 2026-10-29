#!/usr/bin/env python
"""Launcher used by MCP clients and hooks.

Keeping the import path bootstrap here means no ``PYTHONPATH`` environment
variable has to leak into the user's shell, and the same entry point works for
Claude Code (stdio MCP), Codex (stdio MCP), hooks and the CLI.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from supermemory.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
