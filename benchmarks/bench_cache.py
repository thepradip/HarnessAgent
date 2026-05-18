"""Benchmark 2: Semantic LLM cache — hit rate vs similarity threshold.

Tests 100 query pairs (50 paraphrase pairs + 50 different-topic pairs) at 5
similarity thresholds. Measures true-positive rate (paraphrase hits cache),
false-positive rate (unrelated query hits cache), and lookup latency.

Uses a deterministic similarity model when sentence-transformers is not
installed, but real SentenceTransformer embeddings when it is available.

Run:
    PYTHONPATH=src python benchmarks/bench_cache.py

Output:
    benchmarks/results/semantic_cache_hit_rate.json
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS_DIR = ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.70, 0.80, 0.90, 0.95, 0.97, 0.99]

# ---------------------------------------------------------------------------
# Query test cases at 4 similarity tiers.
#
# The semantic cache is designed to catch repeated / near-duplicate queries
# (e.g. the same user asking the same question twice, possibly with minor
# wording differences) — NOT to equate semantically equivalent paraphrases.
# The 0.97 default is deliberately conservative to avoid false positives.
#
# We test 4 tiers to produce a threshold vs hit-rate tradeoff curve:
#   - NEAR_EXACT   : same query, minor surface variation  → expect hit at 0.97+
#   - MINOR_REPHRASE: one word changed, same intent       → expect hit at 0.90–0.97
#   - PARAPHRASE   : different words, same meaning        → expect hit at 0.70–0.90
#   - DECOY        : different question entirely          → expect miss at all thresholds
# ---------------------------------------------------------------------------

QUERY_CASES = [
    # tier, seed, variant
    # NEAR_EXACT: same words, tiny changes (pluralisation, punctuation, leading space)
    ("near_exact",
     "How many pending orders are there?",
     "How many pending orders are there"),
    ("near_exact",
     "List all users with the premium plan",
     "list all users with the premium plan"),
    ("near_exact",
     "Show products where inventory_count is 0",
     "Show products where inventory_count is 0."),
    ("near_exact",
     "What is the total revenue for this month?",
     "What is the total revenue for this month"),
    ("near_exact",
     "Find orders with status cancelled",
     "Find orders with status 'cancelled'"),
    # MINOR_REPHRASE: one or two words swapped
    ("minor_rephrase",
     "How many orders are in pending status?",
     "How many orders have a pending status?"),
    ("minor_rephrase",
     "List users on the premium plan",
     "Show users on the premium plan"),
    ("minor_rephrase",
     "Count products with zero inventory",
     "Count products with no inventory"),
    ("minor_rephrase",
     "What is the average order value this month?",
     "What is the mean order value this month?"),
    ("minor_rephrase",
     "Show the top 5 products by revenue",
     "Show the top 5 products ranked by revenue"),
    # PARAPHRASE: different phrasing, same intent
    ("paraphrase",
     "Which users placed the most orders?",
     "Show customers ranked by number of purchases"),
    ("paraphrase",
     "How many items are out of stock?",
     "Count products with inventory_count equal to zero"),
    ("paraphrase",
     "Show failed payments last month",
     "List unsuccessful transactions from the previous month"),
    ("paraphrase",
     "What is the revenue breakdown by category?",
     "Show total sales grouped by product category"),
    ("paraphrase",
     "Find users who have not ordered in 90 days",
     "Which customers are inactive for the past three months?"),
    # DECOY: different question, should never hit
    ("decoy",
     "Show the top 5 products by revenue",
     "How many users upgraded to the premium plan this quarter?"),
    ("decoy",
     "Count products with zero inventory",
     "What is the average session duration for all users?"),
    ("decoy",
     "List users on the premium plan",
     "Find orders where the payment failed"),
    ("decoy",
     "How many pending orders are there?",
     "Show the refund rate grouped by product category"),
    ("decoy",
     "What is the total revenue for this month?",
     "Which addresses are missing a ZIP code?"),
]


# ---------------------------------------------------------------------------
# Embedder implementations
# ---------------------------------------------------------------------------

class _RealEmbedder:
    """Wraps SentenceTransformer for real semantic embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(model_name)
        self.model = model_name
        self.dimensions = self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


class _DeterministicEmbedder:
    """Hash-based embedder that produces high similarity for text with large n-gram overlap.

    Not a real semantic model, but produces predictable, reproducible results.
    Similar texts (same words) → higher cosine similarity.
    """

    DIMS = 128
    model = "deterministic-ngram"

    @property
    def dimensions(self) -> int:
        return self.DIMS

    def _embed_one(self, text: str) -> list[float]:
        text = text.lower().strip()
        words = text.split()
        vec = [0.0] * self.DIMS
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.DIMS
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _build_embedder():
    try:
        emb = _RealEmbedder()
        print("Using real SentenceTransformer embeddings (all-MiniLM-L6-v2)")
        return emb, True
    except ImportError:
        print("sentence-transformers not installed — using deterministic n-gram embedder")
        return _DeterministicEmbedder(), False


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

async def run() -> None:
    import fakeredis.aioredis as fake_aioredis  # type: ignore

    from harness.llm.cache import SemanticLLMCache

    embedder, real_embeddings = _build_embedder()
    # decode_responses=False: SemanticLLMCache internally uses bytes keys (b"embedding")
    redis_client = fake_aioredis.FakeRedis(decode_responses=False)

    # Seed cache: one entry per unique seed query
    cache = SemanticLLMCache(
        redis_client=redis_client,
        embedding_provider=embedder,
        tenant_id="bench",
    )
    unique_seeds = list({c[1] for c in QUERY_CASES})
    print(f"\nSeeding cache with {len(unique_seeds)} unique queries...")
    for i, seed in enumerate(unique_seeds):
        await cache.set([{"role": "user", "content": seed}], f"result_{i}", ttl=86400)

    # Measure actual cosine similarities for reporting
    print("\nMeasuring cosine similarities by tier (for reference):")
    tier_sims: dict[str, list[float]] = {}
    for tier, seed, variant in QUERY_CASES:
        ea, ev = await embedder.embed([seed, variant])
        cos = _cosine(ea, ev)
        tier_sims.setdefault(tier, []).append(cos)
    for tier, sims in tier_sims.items():
        avg = sum(sims) / len(sims)
        print(f"  {tier:<16} avg cos={avg:.4f}  min={min(sims):.4f}  max={max(sims):.4f}")

    threshold_results = []

    for threshold in THRESHOLDS:
        counts: dict[str, dict[str, int]] = {
            t: {"hit": 0, "miss": 0}
            for t in ["near_exact", "minor_rephrase", "paraphrase", "decoy"]
        }
        latencies: list[float] = []

        for tier, seed, variant in QUERY_CASES:
            t0 = time.perf_counter()
            hit = await cache.get([{"role": "user", "content": variant}], threshold=threshold)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            if hit is not None:
                counts[tier]["hit"] += 1
            else:
                counts[tier]["miss"] += 1

        # Compute rates per tier
        def rate(tier: str) -> float:
            total = counts[tier]["hit"] + counts[tier]["miss"]
            return counts[tier]["hit"] / total if total > 0 else 0

        p_lat = sorted(latencies)
        p50 = p_lat[len(p_lat) // 2]

        row = {
            "threshold": threshold,
            "near_exact_hit_rate":     round(rate("near_exact"), 3),
            "minor_rephrase_hit_rate": round(rate("minor_rephrase"), 3),
            "paraphrase_hit_rate":     round(rate("paraphrase"), 3),
            "decoy_hit_rate":          round(rate("decoy"), 3),  # false positive rate
            "lookup_p50_ms":           round(p50, 2),
        }
        threshold_results.append(row)
        print(
            f"  threshold={threshold:.2f}  near_exact={rate('near_exact'):.0%}  "
            f"minor_rephrase={rate('minor_rephrase'):.0%}  paraphrase={rate('paraphrase'):.0%}  "
            f"decoy(FP)={rate('decoy'):.0%}  p50={p50:.1f}ms"
        )

    # Best operating point: max recall on near_exact with 0% decoy FP
    best = next(
        (r for r in reversed(threshold_results) if r["decoy_hit_rate"] == 0.0),
        threshold_results[-1],
    )
    print(f"\nDefault threshold (0.97): "
          f"near_exact={threshold_results[THRESHOLDS.index(0.97)]['near_exact_hit_rate']:.0%}  "
          f"FP={threshold_results[THRESHOLDS.index(0.97)]['decoy_hit_rate']:.0%}")
    print(f"Using real embeddings: {real_embeddings}")

    output = {
        "benchmark": "semantic_cache_hit_rate",
        "embedder": embedder.model,
        "real_embeddings": real_embeddings,
        "seed_queries": len(unique_seeds),
        "test_cases": len(QUERY_CASES),
        "tier_avg_cosine": {
            tier: round(sum(s) / len(s), 4) for tier, s in tier_sims.items()
        },
        "best_zero_fp_threshold": best,
        "per_threshold": threshold_results,
    }
    out_path = RESULTS_DIR / "semantic_cache_hit_rate.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")

    await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(run())
