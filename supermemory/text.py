"""Text utilities: tokenisation, token estimation, summarisation, chunking.

The tokeniser is language-aware on purpose. SQLite's ``unicode61`` tokenizer
treats an entire Chinese sentence as a single token, which makes CJK search
useless. We pre-tokenise: every CJK character becomes a unigram, and adjacent
characters also emit a bigram. ASCII text is split on camelCase / snake_case so
``MemoryStore`` matches ``memory`` and ``store``.

The same function is applied to indexed text and to queries, so the two always
speak the same vocabulary.
"""

from __future__ import annotations

import re

# (start, end) inclusive code point ranges treated as CJK / kana / hangul.
_CJK_RANGES = (
    (0x2E80, 0x2EFF),    # CJK radicals
    (0x3040, 0x30FF),    # hiragana + katakana
    (0x3400, 0x4DBF),    # CJK ext A
    (0x4E00, 0x9FFF),    # CJK unified
    (0xF900, 0xFAFF),    # CJK compatibility
    (0xAC00, 0xD7AF),    # hangul syllables
    (0x20000, 0x2FA1F),  # CJK ext B+
)

_CJK_CLASS = "".join(
    f"{chr(start)}-{chr(end)}" if start != end else chr(start)
    for start, end in _CJK_RANGES
)

_TOKEN_RE = re.compile(rf"[A-Za-z0-9_]+|[{_CJK_CLASS}]+")
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

# Words that carry no retrieval signal on their own.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "of",
    "to", "in", "on", "at", "by", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "there", "here", "we", "you", "i", "he", "she", "they", "them",
    "our", "your", "my", "me", "us", "do", "does", "did", "not", "no", "yes",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "have", "has", "had", "about", "into", "over", "under", "than", "so",
    "such", "some", "any", "all", "each", "very", "just", "also", "how",
    "what", "when", "where", "which", "who", "why", "use", "using", "used",
}

# Frequent Chinese function characters: kept in the document index (they help
# phrase matching) but dropped from queries when better bigrams exist.
_CJK_STOP = set(
    "的了是在和与或及也就都而并且被把对从到为以于之其这那我你他她它们个有没不很更最再又还只"
)


def is_cjk(ch: str) -> bool:
    code = ord(ch)
    for start, end in _CJK_RANGES:
        if start <= code <= end:
            return True
    return False


def _split_word(word: str) -> list[str]:
    """Split identifiers: camelCase, snake_case, and the original."""
    parts: list[str] = [word]
    if "_" in word:
        parts.extend(p for p in word.split("_") if p)
    if "-" in word:
        parts.extend(p for p in word.split("-") if p)
    if any(ch.isupper() for ch in word[1:]):
        parts.extend(p.lower() for p in _CAMEL_RE.findall(word))
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        part = part.lower()
        if not part or part in seen:
            continue
        seen.add(part)
        out.append(part)
    return out


def tokenize(text: str) -> list[str]:
    """Tokenise mixed CJK/Latin text into indexable terms."""
    if not text:
        return []
    out: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        chunk = match.group()
        if chunk[0].isascii():  # ASCII word run
            out.extend(_split_word(chunk))
        else:  # CJK run: unigrams + bigrams
            out.extend(chunk)
            out.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return out


def index_blob(*fields: str) -> str:
    """Build the space-joined token blob stored in FTS5."""
    tokens: list[str] = []
    for field in fields:
        if field:
            tokens.extend(tokenize(field))
    return " ".join(tokens)


def query_tokens(query: str, limit: int = 40) -> list[str]:
    """Tokens for an FTS query, with stopword and redundancy pruning."""
    raw = tokenize(query)
    has_cjk_bigram = any(len(t) == 2 and not t.isascii() for t in raw)

    out: list[str] = []
    seen: set[str] = set()
    for tok in raw:
        if tok in seen:
            continue
        if tok.isascii():
            if tok in _STOPWORDS or (len(tok) < 2 and not tok.isdigit()):
                continue
        elif len(tok) == 1:
            # Drop single CJK chars when bigrams are available; otherwise keep
            # them (a one-character query still has to match something).
            if has_cjk_bigram or tok in _CJK_STOP:
                continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


def fts_expression(tokens: list[str], op: str = "OR") -> str:
    """Quote tokens into a valid FTS5 MATCH expression."""
    quoted = ['"' + tok.replace('"', '""') + '"' for tok in tokens]
    return f" {op} ".join(quoted)


def estimate_tokens(text: str) -> int:
    """Cheap cross-language token estimate (no tokenizer dependency)."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if is_cjk(ch))
    other = len(text) - cjk
    return int(cjk + other / 3.6) + 1


_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+(?=[A-Z0-9])|\n+")


def summarize(text: str, limit_chars: int = 180) -> str:
    """First meaningful sentence(s) of a chunk, for cheap scanning."""
    flat = re.sub(r"\s+", " ", (text or "").strip())
    if not flat:
        return ""
    if len(flat) <= limit_chars:
        return flat
    parts = [p.strip() for p in _SENTENCE_RE.split(flat) if p and p.strip()]
    out = ""
    for part in parts:
        if out and len(out) + len(part) + 1 > limit_chars:
            break
        out = f"{out} {part}".strip() if out else part
        if len(out) >= limit_chars * 0.6:
            break
    if len(out) < limit_chars * 0.4:  # no sentence structure: hard cut
        out = flat[:limit_chars].rsplit(" ", 1)[0]
    if len(out) > limit_chars:
        out = out[:limit_chars]
    return out.rstrip() + ("…" if len(flat) > len(out) else "")


def chunk_text(
    text: str, max_tokens: int = 260, overlap_tokens: int = 48
) -> list[str]:
    """Split long text into overlapping, boundary-aware chunks.

    Preference order: blank-line blocks > lines > sentences > hard cut. Chunks
    stay under ``max_tokens`` (estimated) and carry ``overlap_tokens`` of the
    previous chunk so a fact split across the boundary is still recoverable.
    """
    text = (text or "").strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    units = _to_units(text, max_tokens)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit)
        if current and current_tokens + unit_tokens > max_tokens:
            chunks.append("\n".join(current).strip())
            if overlap_tokens > 0:
                tail: list[str] = []
                tail_tokens = 0
                for prev in reversed(current):
                    prev_tokens = estimate_tokens(prev)
                    if tail_tokens + prev_tokens > overlap_tokens:
                        break
                    tail.insert(0, prev)
                    tail_tokens += prev_tokens
                current, current_tokens = tail, tail_tokens
            else:
                current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        tail_text = "\n".join(current).strip()
        if tail_text and (not chunks or tail_text != chunks[-1]):
            chunks.append(tail_text)
    return [c for c in chunks if c.strip()]


def _to_units(text: str, max_tokens: int) -> list[str]:
    """Break text into units small enough to be packed into chunks."""
    blocks = re.split(r"\n\s*\n", text)
    units: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if estimate_tokens(block) <= max_tokens:
            units.append(block)
            continue
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if estimate_tokens(line) <= max_tokens:
                units.append(line)
                continue
            for piece in _split_sentences(line):
                if estimate_tokens(piece) <= max_tokens:
                    units.append(piece)
                else:
                    units.extend(_hard_split(piece, max_tokens))
    return units


def _split_sentences(line: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(line) if p and p.strip()]
    return parts or [line]


def _hard_split(piece: str, max_tokens: int) -> list[str]:
    """Last resort: slice by character count using the CJK-heavy estimate."""
    approx_chars = max(64, int(max_tokens * 2.2))
    return [
        piece[i : i + approx_chars]
        for i in range(0, len(piece), approx_chars)
        if piece[i : i + approx_chars].strip()
    ]
