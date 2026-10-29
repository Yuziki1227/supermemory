# Contributing to SuperMemory

Thanks for taking the time to contribute. This project is deliberately small:
one SQLite file, one MCP server, no cloud. Contributions that keep it that way
are the most welcome.

## Ground rules

- **Local-first stays true.** No runtime dependency beyond `numpy`, and no
  network call in the default path.
- **Do not silently change semantics.** Updates keep history, near-duplicates are
  never merged, hooks never block a session, and MCP stdout carries protocol
  messages only. See [`AGENTS.md`](AGENTS.md) for the full invariant list.
- **Docs are bilingual.** `README.md` (English) and `README.zh-CN.md` (Chinese)
  are mirrors; a new flag, config key or tool belongs in both.
- **Be kind.** Assume good intent, keep reviews about the code.

## Development setup

```bash
git clone https://github.com/Yuziki1227/supermemory.git
cd supermemory

python -m venv .venv
# Windows (Git Bash / PowerShell): .venv/Scripts/activate
# macOS / Linux:                    source .venv/bin/activate

pip install -e .          # installs numpy and the `supermemory` entry point
```

No other tooling is required. If you prefer not to install the package,
`python run.py <command>` and `python -m supermemory <command>` work from the
checkout as long as `numpy` is available.

To wire the checkout into Claude Code or Codex without copying files:

```bash
python install.py --dev
```

## Running the tests

```bash
python -m unittest discover -s tests -v
```

The suite covers the tokenizer, chunking, store semantics, version history,
deduplication, namespace isolation, token budgets, briefing, export/import and a
real MCP subprocess handshake. Tests point `SUPERMEMORY_HOME` at a temporary
directory, so they never touch your real memory database.

Retrieval changes also need the benchmark:

```bash
python -m supermemory bench bench/cases.jsonl --fail-under-recall 0.85 --fail-under-mrr 0.75
```

Add a case to `bench/cases.jsonl` for every retrieval bug you fix; the file
format and metric definitions are in [`bench/README.md`](bench/README.md).

Before opening a pull request, also run:

```bash
python -m supermemory doctor
python install.py --dry-run
```

## Trying a change by hand

Use a throwaway data directory so experiments stay out of your real memory:

```bash
export SUPERMEMORY_HOME=/tmp/sm-dev          # Windows: set SUPERMEMORY_HOME=%TEMP%\sm-dev
python -m supermemory remember "test fact" -t demo -k fact
python -m supermemory recall "test"
python -m supermemory stats
```

For MCP changes, run the server directly and send newline-delimited JSON-RPC:

```bash
python run.py mcp
```

## What to work on

- Bug reports are most useful with: OS, Python version, `python -m supermemory
  doctor` output, the exact command or MCP call, and what you expected.
- Ranking-quality reports are most useful with a small reproducible pair: the
  query, the memory that should have won, and (if possible) `--json` recall
  output.
- For larger features, open an issue first so the design can be discussed before
  you invest time.

## Pull requests

1. Keep one PR focused on one concern.
2. Add or update tests for behaviour changes. Bug fixes need a regression test.
3. Run the full suite and make sure it passes.
4. Update both READMEs and `AGENTS.md` when the public surface changes.
5. Do not include real memory exports, personal data, API keys or tokens in
   fixtures, logs or screenshots.
6. Do not bump the version yourself; maintainers do that at release time.

Commit messages: imperative mood, short subject, optional scope
(`recall: skip superseded rows before scoring`). Explain *why* in the body when
the change is not obvious.

## Security and privacy

SuperMemory stores whatever an agent writes into `~/.supermemory/memory.db`. That
can include project details you would not want public. Before sharing logs or
reproductions, redact memory content.

To report a vulnerability, use GitHub's **private vulnerability reporting**
(Security tab → Report a vulnerability) rather than a public issue.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
