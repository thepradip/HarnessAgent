"""Advantage estimation for RLVR — discounted return + baseline subtraction + normalisation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from harness.improvement.rlvr.buffer import StepReward


@dataclass
class StepAdvantage:
    step: StepReward
    discounted_return: float   # G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
    baseline: float
    advantage: float           # G_t - baseline (raw, un-normalised — decides reinforce/patch)
    weight: float              # z-normalised |advantage| — gradient-scaling weight only
    norm_advantage: float = 0.0  # z-normalised advantage (per-episode), used for weighting


class AdvantageEstimator:
    """
    Computes per-step advantages for an episode.

    Formula (REINFORCE with baseline):
        G[t] = Σ_{k=t}^{T} γ^(k-t) * r[k]     (discounted return from step t)
        A[t] = G[t] - baseline                   (raw advantage — signal)
        A_norm[t] = (A[t] - mean(A)) / (std(A) + ε)   (per-episode z-score — weight only)

    The *raw* advantage (G - baseline) is the signal that decides reinforce vs
    patch in ``split()``: it preserves the rolling baseline's effect, which the
    z-normalised value cancels out (mean(A_norm)=0 always, so the baseline drops
    out of the per-episode centring). The z-score is kept only as a gradient-
    scaling weight, not as the decision signal.

    γ=1.0 gives undiscounted return (no decay — good for short episodes).
    γ<1.0 discounts future rewards (good for long runs where later steps matter less).
    """

    def __init__(self, gamma: float = 0.95, eps: float = 1e-8) -> None:
        self.gamma = gamma
        self.eps = eps

    def compute(
        self,
        episode: list[StepReward],
        baseline: float,
    ) -> list[StepAdvantage]:
        if not episode:
            return []

        rewards = [s.reward for s in episode]

        # Discounted returns (backward pass)
        T = len(rewards)
        returns = [0.0] * T
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            returns[t] = running

        # Raw advantages (G - baseline) — this is the decision signal. It keeps
        # the rolling baseline's effect, unlike the z-normalised value below.
        raw = [g - baseline for g in returns]

        # Z-normalise for gradient scaling ONLY. Note mean(raw) re-centres each
        # episode, which cancels the baseline; so this is never used to decide
        # reinforce/patch — only as a per-step weight.
        mean_a = sum(raw) / len(raw)
        var_a = sum((a - mean_a) ** 2 for a in raw) / len(raw)
        std_a = math.sqrt(var_a + self.eps)
        normalised = [(a - mean_a) / std_a for a in raw]

        return [
            StepAdvantage(
                step=step,
                discounted_return=round(returns[i], 6),
                baseline=baseline,
                advantage=round(raw[i], 6),
                norm_advantage=round(normalised[i], 6),
                weight=round(abs(normalised[i]), 6),
            )
            for i, step in enumerate(episode)
        ]

    def split(
        self,
        advantages: list[StepAdvantage],
        pos_threshold: float = 0.5,
        neg_threshold: float = -0.5,
    ) -> tuple[list[StepAdvantage], list[StepAdvantage]]:
        """
        Split into positive (reinforce) and negative (patch) sets.

        The decision operates on the *raw* advantage (G - baseline), NOT the
        per-episode z-score. Thresholds are in reward units relative to the
        rolling baseline: a step is reinforced when its return is at least
        ``pos_threshold`` above baseline, and patched when at least
        ``-neg_threshold`` below it. This keeps the baseline meaningful — a
        uniformly-good episode (all steps above baseline) yields all-positive,
        none-negative, instead of being split by a zero-mean z-score.

        Returns (positive, negative) where:
          positive: advantage >= pos_threshold  (above-baseline steps — good)
          negative: advantage <= neg_threshold  (below-baseline steps — bad)
        """
        positive = [a for a in advantages if a.advantage >= pos_threshold]
        negative = [a for a in advantages if a.advantage <= neg_threshold]
        return positive, negative
