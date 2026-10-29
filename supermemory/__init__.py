"""SuperMemory — a local, attention-weighted long-term memory for coding agents.

Storage : SQLite (+ FTS5) in ``$SUPERMEMORY_HOME`` (default ``~/.supermemory``)
Retrieval: hybrid lexical + vector + recency + importance + reinforcement,
           combined with a softmax "attention" layer and MMR de-duplication.
Transport: hand-rolled MCP stdio server, so Claude Code and Codex share one brain.
"""

__version__ = "1.1.0"
__all__ = ["__version__"]
