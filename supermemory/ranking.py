"""The attention layer.

Every candidate memory is described by a small feature vector:

    lexical         BM25 from FTS5, normalised against the candidate set
    vector          cosine similarity with the query (when an embedder is on)
    importance      author-declared, damped by confidence
    recency         exponential decay on the last touch (write or recall)
    reinforcement   log(1 + times recalled) — memories that proved useful
                    before are more likely to be useful again
    pinned          explicit, permanent boost

The weighted sum is the *score*. ``softmax(score / temperature)`` turns it into
an attention distribution ``alpha`` over the candidate set — the same shape as
attention over a memory bank: a fixed budget of probability mass is split
across chunks according to relevance, and the sharpness of that split is
controlled by one knob.

Final selection uses MMR (maximal marginal relevance) so the top-k is not five
near-identical chunks, and a token budget so recall never blows up the context
window.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_WEIGHTS = {
    "lexical": 0.36,
    "vector": 0.28,
    "importance": 0.13,
    "recency": 0.09,
    "reinforcement": 0.06,
    "pinned": 0.08,
}


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


def score_candidates(
    candidates: list[dict],
    config: dict,
    now: float,
) -> list[dict]:
    """Annotate candidates with ``score`` and ``attention``; sort descending."""
    if not candidates:
        return []

    rank_cfg = config.get("ranking", {})
    weights = {**DEFAULT_WEIGHTS, **(rank_cfg.get("weights") or {})}
    half_life = float(rank_cfg.get("half_life_days", 45.0)) * 86400.0
    temperature = max(0.02, float(rank_cfg.get("temperature", 0.35)))
    confidence_floor = float(rank_cfg.get("confidence_floor", 0.25))

    max_lex = max((c.get("lex", 0.0) for c in candidates), default=0.0) or 1.0

    for cand in candidates:
        lexical = _clamp01(cand.get("lex", 0.0) / max_lex)
        vector = _clamp01(cand.get("cosine", 0.0))
        importance = _clamp01(cand.get("importance", 0.5))
        confidence = _clamp01(cand.get("confidence", 0.8))
        confidence_factor = 0.6 + 0.4 * confidence
        if confidence < confidence_floor:
            confidence_factor *= 0.6

        last_touch = max(
            cand.get("updated_at") or 0.0,
            cand.get("accessed_at") or 0.0,
            cand.get("created_at") or 0.0,
        )
        age = max(0.0, now - last_touch)
        recency = math.exp(-math.log(2.0) * age / half_life) if half_life > 0 else 1.0

        reinforcement = min(1.0, math.log1p(cand.get("access_count", 0)) / math.log1p(20.0))
        pinned = 1.0 if cand.get("pinned") else 0.0

        score = (
            weights["lexical"] * lexical
            + weights["vector"] * vector
            + weights["importance"] * importance * confidence_factor
            + weights["recency"] * recency
            + weights["reinforcement"] * reinforcement
            + weights["pinned"] * pinned
        )
        # never let a pinned/important-but-irrelevant memory outrank an exact match
        if lexical == 0.0 and vector < 0.15:
            score *= 0.55

        cand["features"] = {
            "lexical": round(lexical, 4),
            "vector": round(vector, 4),
            "importance": round(importance * confidence_factor, 4),
            "recency": round(recency, 4),
            "reinforcement": round(reinforcement, 4),
            "pinned": pinned,
        }
        cand["score"] = score

    # softmax attention over the whole candidate pool
    peak = max(c["score"] for c in candidates)
    exps = [math.exp((c["score"] - peak) / temperature) for c in candidates]
    total = sum(exps) or 1.0
    for cand, weight in zip(candidates, exps):
        cand["attention"] = weight / total

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def mmr_select(
    scored: list[dict],
    limit: int,
    max_tokens: int | None,
    config: dict,
    vectors: dict[int, np.ndarray] | None = None,
    costs: dict[int, int] | None = None,
) -> list[dict]:
    """Pick up to ``limit`` results, diverse and within a token budget."""
    if not scored:
        return []
    mmr_lambda = float(config.get("ranking", {}).get("mmr_lambda", 0.22))
    min_score = float(config.get("recall", {}).get("min_score", 0.0))
    vectors = vectors or {}
    costs = costs or {}

    pool = [c for c in scored if c.get("score", 0.0) >= min_score]
    if not pool:
        pool = scored[:1]

    picked: list[dict] = []
    picked_vectors: list[np.ndarray] = []
    used_tokens = 0

    while len(picked) < limit and pool:
        best = None
        best_value = -1e9
        for cand in pool:
            vector = vectors.get(cand["id"])
            penalty = 0.0
            if picked_vectors and vector is not None:
                similarity = max(
                    [0.0] + [float(np.dot(vector, other)) for other in picked_vectors]
                )
                penalty = mmr_lambda * similarity
            value = cand["score"] - penalty
            if value > best_value:
                best, best_value = cand, value

        if best is None:
            break
        cost = costs.get(best["id"], 0)
        if max_tokens and picked and used_tokens + cost > max_tokens:
            break
        picked.append(best)
        pool.remove(best)
        used_tokens += cost
        if best["id"] in vectors:
            picked_vectors.append(vectors[best["id"]])
        if max_tokens and used_tokens >= max_tokens:
            break

    return picked
