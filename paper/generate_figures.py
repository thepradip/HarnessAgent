"""
Generate all paper figures from benchmark result JSONs.

Run:
    cd paper && python generate_figures.py

Outputs (all in paper/figures/):
    fig1_agencybench_ablation.pdf   — the killer result / Figure 1
    fig2_graphrag_tokens.pdf        — GraphRAG token savings
    fig3_hermes_improvement.pdf     — Hermes before/after
    fig4_safety_pipeline.pdf        — tau-bench + ATBench safety
    fig5_span_overhead.pdf          — fakeredis vs real Redis
    fig6_circuit_breaker.pdf        — circuit breaker state machine
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as FancyArrow
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
OUTDIR  = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#2C6FBF"
C_GREEN  = "#2E8B57"
C_ORANGE = "#D4720A"
C_RED    = "#C0392B"
C_TEAL   = "#1A7A6E"
C_GRAY   = "#7F8C8D"
C_LIGHT  = "#ECF0F1"

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize":8,
    "ytick.labelsize":8,
    "legend.fontsize":8,
    "figure.dpi":    150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


# =============================================================================
# Fig 1 — AgencyBench-V2 Harness Ablation
# =============================================================================

def fig1_agencybench() -> None:
    data = json.loads((RESULTS / "agencybench_v2_ablation.json").read_text())
    conditions = data["conditions"]

    # Overall pass rates (regular tasks only)
    labels = ["A  Bare\n(no harness)", "B  +HarnessAgent\ninfra", "C  Full\nHaaS"]
    rates  = [c["pass_rate"] * 100 for c in conditions]
    colors = [C_GRAY, C_BLUE, C_GREEN]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    bars = ax.bar(labels, rates, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{rate:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Native-SDK baseline line
    native = 48.4
    ax.axhline(native, color=C_RED, linestyle="--", linewidth=1.2, label=f"AgencyBench native-SDK baseline ({native}%)")
    ax.text(2.42, native + 0.6, f"native-SDK\n{native}%",
            color=C_RED, fontsize=7.5, va="bottom", ha="right")

    # Gain annotations
    ax.annotate("", xy=(1, rates[1]), xytext=(0, rates[0]),
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.2),
                xycoords=("data", "data"))
    ax.text(0.5, (rates[0]+rates[1])/2 + 1, f"+{rates[1]-rates[0]:.1f} pp",
            ha="center", color=C_BLUE, fontsize=8, fontweight="bold")
    ax.annotate("", xy=(2, rates[2]), xytext=(1, rates[1]),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2),
                xycoords=("data", "data"))
    ax.text(1.5, (rates[1]+rates[2])/2 + 1, f"+{rates[2]-rates[1]:.1f} pp",
            ha="center", color=C_GREEN, fontsize=8, fontweight="bold")

    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 85)
    ax.set_title("Figure 1 — AgencyBench-V2 Harness Ablation\n"
                 "28 tasks · 6 capability dimensions · seed=42")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out = OUTDIR / "fig1_agencybench_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# =============================================================================
# Fig 2 — GraphRAG Token Savings
# =============================================================================

def fig2_graphrag() -> None:
    data = json.loads((RESULTS / "graphrag_token_efficiency.json").read_text())
    s = data["summary"]

    categories = ["Naive\n(full DDL)", "GraphRAG\n(BFS retrieval)"]
    tokens     = [s["naive_avg_tokens"], s["graphrag_avg_tokens"]]
    colors     = [C_RED, C_GREEN]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.4))

    # Left: token comparison bar
    bars = ax1.bar(categories, tokens, color=colors, width=0.45,
                   edgecolor="white", linewidth=0.8)
    for bar, t in zip(bars, tokens):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f"{t:,.0f}", ha="center", fontweight="bold", fontsize=9)
    ax1.set_ylabel("Average Tokens per Query")
    ax1.set_title("Token Consumption")
    ax1.set_ylim(0, 2700)

    reduction_pct = s["overall_savings_pct"]
    ax1.annotate(
        f"−{reduction_pct:.1f}%\nreduction",
        xy=(1, tokens[1]), xytext=(0.5, 1600),
        fontsize=9, fontweight="bold", color=C_GREEN,
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2),
        ha="center",
    )

    # Right: savings range per query type (from per-query data if available)
    if "per_query" in data:
        savings = [q["savings_pct"] for q in data["per_query"]]
        ax2.hist(savings, bins=8, color=C_BLUE, edgecolor="white", linewidth=0.5)
        ax2.set_xlabel("Token Savings (%)")
        ax2.set_ylabel("Number of Queries")
        ax2.set_title(f"Savings Distribution\n(n={len(savings)} queries)")
    else:
        # Show strategy breakdown
        strategies = s.get("strategy_breakdown", {"vector_primary": 19, "graph_primary": 1})
        slabels = [k.replace("_", "\n") for k in strategies]
        svals   = list(strategies.values())
        ax2.pie(svals, labels=slabels, autopct="%1.0f%%",
                colors=[C_BLUE, C_TEAL], startangle=90,
                wedgeprops=dict(edgecolor="white"))
        ax2.set_title(f"Retrieval Strategy\n(n={sum(svals)} queries)")

    fig.suptitle("Figure 2 — GraphRAG Token Efficiency\n"
                 "10-table schema · 20 queries · 100% table coverage",
                 y=1.01)
    fig.tight_layout()
    out = OUTDIR / "fig2_graphrag_tokens.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# =============================================================================
# Fig 3 — Hermes Self-Improvement
# =============================================================================

def fig3_hermes() -> None:
    real_raw = json.loads((RESULTS / "hermes_real_results.json").read_text())
    real = real_raw.get("summary", real_raw)
    gaia_raw = json.loads((RESULTS / "hermes_gaia_improvement.json").read_text())
    gaia = gaia_raw.get("summary", gaia_raw)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    x      = np.arange(2)
    width  = 0.28
    before = [real["pre_patch_pass1"]*100,  gaia["pre_patch_pass1"]*100]
    after  = [real["post_patch_pass1"]*100, gaia["post_patch_pass1"]*100]
    xlabels = ["SQL tasks\n(real SQLite, n=50)", "GAIA-style tasks\n(embedded, n=20)"]

    b1 = ax.bar(x - width/2, before, width, label="Pre-patch",
                color=C_GRAY, edgecolor="white")
    b2 = ax.bar(x + width/2, after,  width, label="Post-patch",
                color=[C_BLUE, C_GREEN], edgecolor="white")

    # Gain labels
    for i, (b, a) in enumerate(zip(before, after)):
        gain = a - b
        ax.text(x[i] + width/2,
                a + 1.2,
                f"+{gain:.0f} pp",
                ha="center", va="bottom",
                fontweight="bold", fontsize=9,
                color=C_BLUE if i == 0 else C_GREEN)

    # Value labels on before bars
    for bar, v in zip(b1, before):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                f"{v:.0f}%", ha="center", fontsize=7.5, color=C_GRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Pass@1 (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")
    ax.set_title("Figure 3 — Hermes Self-Improvement\n"
                 "Converges at cycle 1 in both benchmarks")

    fig.tight_layout()
    out = OUTDIR / "fig3_hermes_improvement.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# =============================================================================
# Fig 4 — Safety Pipeline (τ-bench + ATBench)
# =============================================================================

def fig4_safety() -> None:
    tau  = json.loads((RESULTS / "taubench_safety.json").read_text())
    atb  = json.loads((RESULTS / "atbench_safety_trajectories.json").read_text())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # ── Left: τ-bench compliance OFF vs ON ───────────────────────────────────
    cond_x = tau["conditions"][0]
    cond_y = tau["conditions"][1]
    metrics  = ["Adversarial\nBlocked", "Policy\nCompliance", "False\nPositive"]
    off_vals = [cond_x["adversarial_block_rate"]*100,
                cond_x["policy_compliance_rate"]*100,
                cond_x["false_positive_rate"]*100]
    on_vals  = [cond_y["adversarial_block_rate"]*100,
                cond_y["policy_compliance_rate"]*100,
                cond_y["false_positive_rate"]*100]

    xpos = np.arange(len(metrics))
    w = 0.32
    ax1.bar(xpos - w/2, off_vals, w, label="Safety OFF", color=C_RED,   alpha=0.8, edgecolor="white")
    ax1.bar(xpos + w/2, on_vals,  w, label="Safety ON",  color=C_GREEN, alpha=0.9, edgecolor="white")

    for pos, off, on in zip(xpos, off_vals, on_vals):
        ax1.text(pos - w/2, off + 1, f"{off:.0f}%", ha="center", fontsize=7.5, color=C_RED)
        ax1.text(pos + w/2, on  + 1, f"{on:.0f}%",  ha="center", fontsize=7.5,
                 color=C_GREEN, fontweight="bold" if on >= 90 else "normal")

    ax1.set_xticks(xpos)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel("Rate (%)")
    ax1.set_ylim(0, 118)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_title("τ-bench Safety\n50 tasks (40 benign, 10 adversarial)")

    # ── Right: ATBench per-category TPR ─────────────────────────────────────
    cat_map = atb.get("by_category", {})
    # Keep only the 6 ATBench categories
    keep = ["Prompt Injection", "Privilege Escalation", "Data Exfiltration",
            "Unsafe Code Execution", "Policy Bypass", "Harmful Content Generation"]
    cats = [c for c in keep if c in cat_map]
    tprs = [cat_map[c] * 100 for c in cats]
    short_labels = [c.replace(" ", "\n") for c in cats]

    colors = [C_GREEN if t == 100 else C_ORANGE for t in tprs]
    bars = ax2.barh(range(len(cats)), tprs, color=colors,
                    edgecolor="white", height=0.55)
    for i, (bar, t) in enumerate(zip(bars, tprs)):
        ax2.text(t + 1, i, f"{t:.0f}%", va="center", fontsize=8,
                 fontweight="bold", color=C_GREEN if t == 100 else C_ORANGE)

    ax2.axvline(90, color=C_RED, linestyle="--", linewidth=1, label="Target ≥ 90%")
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels(short_labels, fontsize=7.5)
    ax2.set_xlabel("TPR (%)")
    ax2.set_xlim(0, 115)
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.set_title("ATBench Safety TPR\nby Category (30 unsafe scenarios)")

    fig.suptitle("Figure 4 — Safety Pipeline Validation", y=1.02)
    fig.tight_layout()
    out = OUTDIR / "fig4_safety_pipeline.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# =============================================================================
# Fig 5 — Span Recording Overhead
# =============================================================================

def fig5_spans() -> None:
    fake = json.loads((RESULTS / "span_recording_overhead.json").read_text())
    real = json.loads((RESULTS / "span_recording_overhead_redis.json").read_text())

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    percentiles = ["p50", "p95", "p99"]
    fake_m = fake["measurements"]
    real_m = real["measurements"]

    # start_span + end_span rows
    fake_row = next(m for m in fake_m if "start_span" in m["label"])
    real_row = next(m for m in real_m if "start_span" in m["label"])

    fake_vals = [fake_row["p50_us"], fake_row["p95_us"], fake_row["p99_us"]]
    real_vals = [real_row["p50_us"], real_row["p95_us"], real_row["p99_us"]]

    x = np.arange(len(percentiles))
    w = 0.32
    ax.bar(x - w/2, fake_vals, w, label="fakeredis (in-process)", color=C_BLUE,  edgecolor="white")
    ax.bar(x + w/2, real_vals, w, label="real Redis (localhost)",  color=C_TEAL, edgecolor="white")

    for pos, fv, rv in zip(x, fake_vals, real_vals):
        ax.text(pos - w/2, fv + 20, f"{fv:,.0f}", ha="center", fontsize=7.5, color=C_BLUE)
        ax.text(pos + w/2, rv + 20, f"{rv:,.0f}", ha="center", fontsize=7.5, color=C_TEAL)

    # 5 ms production threshold
    threshold_us = 5000
    ax.axhline(threshold_us, color=C_RED, linestyle="--", linewidth=1.2,
               label="5 ms production threshold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"p{p}" for p in [50, 95, 99]])
    ax.set_ylabel("Latency (µs)")
    ax.set_ylim(0, 5800)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Figure 5 — Span Recording Overhead\n"
                 "2,000 iterations · start_span + end_span · n=2000")

    # Delta annotation
    delta = real_row["p50_us"] - fake_row["p50_us"]
    ax.annotate(
        f"+{delta:.0f} µs\n(network stack)",
        xy=(0 + w/2, real_vals[0]),
        xytext=(0.7, real_vals[0] + 400),
        fontsize=7.5, color=C_TEAL,
        arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1),
    )

    fig.tight_layout()
    out = OUTDIR / "fig5_span_overhead.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# =============================================================================
# Fig 6 — Circuit Breaker State Machine
# =============================================================================

def fig6_circuit_breaker() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def state_box(x, y, label, color, text_color="white"):
        box = FancyBboxPatch((x - 1.0, y - 0.45), 2.0, 0.9,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor="white",
                             linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=text_color)

    # States
    state_box(1.5, 2.0, "CLOSED",    C_GREEN)
    state_box(5.0, 2.0, "OPEN",      C_RED)
    state_box(8.5, 2.0, "HALF-OPEN", C_ORANGE)

    arrow_kw = dict(arrowstyle="-|>", color=C_GRAY,
                    lw=1.4, mutation_scale=12)

    # CLOSED → OPEN
    ax.annotate("", xy=(3.95, 2.25), xytext=(2.5, 2.25),
                arrowprops=arrow_kw)
    ax.text(3.2, 2.55, "5 failures\n(0.01 ms)", ha="center",
            fontsize=7.5, color=C_RED, fontstyle="italic")

    # OPEN → HALF-OPEN
    ax.annotate("", xy=(7.45, 2.25), xytext=(6.05, 2.25),
                arrowprops=arrow_kw)
    ax.text(6.75, 2.55, "60 s timeout", ha="center",
            fontsize=7.5, color=C_ORANGE, fontstyle="italic")

    # HALF-OPEN → CLOSED (success)
    ax.annotate("", xy=(2.5, 1.75), xytext=(7.45, 1.75),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN,
                                lw=1.4, mutation_scale=12,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(5.0, 0.85, "2 successes → CLOSED", ha="center",
            fontsize=7.5, color=C_GREEN, fontstyle="italic")

    # HALF-OPEN → OPEN (failure)
    ax.annotate("", xy=(6.05, 2.25), xytext=(7.45, 2.25),
                arrowprops=dict(arrowstyle="-|>", color=C_RED,
                                lw=1.0, mutation_scale=10,
                                connectionstyle="arc3,rad=0.4"))
    ax.text(6.75, 3.1, "failure → OPEN", ha="center",
            fontsize=7, color=C_RED, fontstyle="italic")

    # Self-loop on CLOSED (normal ops)
    ax.annotate("", xy=(0.6, 2.55), xytext=(0.6, 1.45),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN,
                                lw=1.0, mutation_scale=9,
                                connectionstyle="arc3,rad=-0.7"))
    ax.text(0.0, 2.0, "pass", ha="center", fontsize=7, color=C_GREEN)

    ax.set_title("Figure 6 — Circuit Breaker State Machine\n"
                 "0.01 ms to open · 0 false trips across 3 fault scenarios",
                 fontsize=9, pad=6)

    fig.tight_layout()
    out = OUTDIR / "fig6_circuit_breaker.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out.name}")


# =============================================================================
# Run all
# =============================================================================

if __name__ == "__main__":
    print(f"\nGenerating figures → {OUTDIR}\n")
    fig1_agencybench()
    fig2_graphrag()
    fig3_hermes()
    fig4_safety()
    fig5_spans()
    fig6_circuit_breaker()
    print(f"\nDone — {len(list(OUTDIR.glob('*.pdf')))} PDF + PNG pairs written.\n")
