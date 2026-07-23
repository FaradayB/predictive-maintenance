"""
=============================================================================
 src/dashboard.py
 Vehicle Predictive Maintenance — Cost & Performance Report Generator
=============================================================================
 Reads logs/app.jsonl and reports/eval_summary.csv to produce:
   - reports/cost_analysis.md    — per-query cost breakdown
   - reports/dashboard.png       — performance + cost charts

 Usage:
   python -m src.dashboard
=============================================================================
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from predictivecare.logger import read_logs_as_df, COST_INPUT_PER_TOKEN, COST_OUTPUT_PER_TOKEN

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FAULT_LABELS = {
    0: "Normal", 1: "Battery Degradation", 2: "Brake System Issue",
    3: "Cooling System Problem", 4: "Engine Misfire",
    5: "Alternator Failure", 6: "Oil Pressure Issue",
    7: "Transmission Problem",
}
RISK_LABELS = {
    0: "No Risk", 1: "Low Risk", 2: "Medium Risk", 3: "High Risk",
}


# ─────────────────────────────────────────────────────────────────────────────
# COST ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def build_cost_analysis(df: pd.DataFrame) -> dict:
    """Compute cost metrics from the log DataFrame."""
    llm_rows = df[df["llm_called"] == True]
    total_queries     = len(df)
    llm_queries       = len(llm_rows)
    skipped_queries   = total_queries - llm_queries

    total_input_tok   = df["input_tokens"].sum()
    total_output_tok  = df["output_tokens"].sum()
    total_cost        = df["cost_usd"].sum()
    avg_cost_per_query = (total_cost / llm_queries) if llm_queries > 0 else 0

    t1 = df[df["track"] == 1]
    t2 = df[df["track"] == 2]

    return {
        "total_queries":       total_queries,
        "llm_queries":         llm_queries,
        "skipped_queries":     skipped_queries,
        "total_input_tokens":  int(total_input_tok),
        "total_output_tokens": int(total_output_tok),
        "total_cost_usd":      round(float(total_cost), 6),
        "avg_cost_per_query":  round(float(avg_cost_per_query), 6),
        "t1_queries":          len(t1),
        "t1_cost":             round(float(t1["cost_usd"].sum()), 6),
        "t2_queries":          len(t2),
        "t2_cost":             round(float(t2["cost_usd"].sum()), 6),
        "avg_response_ms":     round(float(df[df["response_time_ms"] > 0]["response_time_ms"].mean()), 1) if llm_queries > 0 else 0,
        "p95_response_ms":     round(float(df[df["response_time_ms"] > 0]["response_time_ms"].quantile(0.95)), 1) if llm_queries > 0 else 0,
    }


def write_cost_markdown(stats: dict, eval_summary: pd.DataFrame) -> Path:
    """Write cost_analysis.md to reports/."""
    path = REPORTS_DIR / "cost_analysis.md"
    now  = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# PredictiveCare — Cost & Performance Analysis",
        f"_Generated: {now}_",
        "",
        "---",
        "",
        "## Query Volume",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total queries processed | {stats['total_queries']} |",
        f"| LLM calls made | {stats['llm_queries']} |",
        f"| Skipped (Normal / No Risk) | {stats['skipped_queries']} |",
        f"| Track 1 queries | {stats['t1_queries']} |",
        f"| Track 2 queries | {stats['t2_queries']} |",
        "",
        "---",
        "",
        "## API Cost (Google Gemini)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total input tokens | {stats['total_input_tokens']:,} |",
        f"| Total output tokens | {stats['total_output_tokens']:,} |",
        f"| Total cost | ${stats['total_cost_usd']:.6f} |",
        f"| Average cost per LLM query | ${stats['avg_cost_per_query']:.6f} |",
        f"| Track 1 total cost | ${stats['t1_cost']:.6f} |",
        f"| Track 2 total cost | ${stats['t2_cost']:.6f} |",
        "",
        "**Pricing basis:** gemini-2.0-flash",
        f"- Input: $0.10 / 1M tokens ({COST_INPUT_PER_TOKEN * 1_000_000:.2f} per token)",
        f"- Output: $0.40 / 1M tokens ({COST_OUTPUT_PER_TOKEN * 1_000_000:.2f} per token)",
        "",
        "---",
        "",
        "## Response Time",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Average (LLM queries only) | {stats['avg_response_ms']} ms |",
        f"| p95 latency | {stats['p95_response_ms']} ms |",
        "",
        "---",
        "",
    ]

    if not eval_summary.empty:
        lines += [
            "## ML Model Accuracy",
            "",
            "| Track | Samples | Accuracy | F1 (Weighted) | F1 (Macro) |",
            "|---|---|---|---|---|",
        ]
        for _, row in eval_summary.iterrows():
            lines.append(
                f"| Track {int(row['track'])} | {int(row['valid_samples'])} | "
                f"{row['accuracy']:.4f} | {row['f1_weighted']:.4f} | "
                f"{row['f1_macro']:.4f} |"
            )
        lines += ["", "---", ""]

    lines += [
        "## Cost Projection",
        "",
        "| Scale | Estimated Monthly Cost |",
        "|---|---|",
    ]

    daily_avg = stats["avg_cost_per_query"]
    for vehicles, label in [(100, "100 vehicles"), (500, "500 vehicles"), (1000, "1,000 vehicles")]:
        # 2 queries/day per vehicle (1 T1 + 1 T2), assume 60% trigger LLM
        monthly = vehicles * 2 * 0.6 * 30 * daily_avg
        lines.append(f"| {label} / day | ${monthly:.2f} |")

    lines += ["", "_Note: Projections assume 60% of queries trigger an LLM call._"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD CHART
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard_chart(df: pd.DataFrame, eval_summary: pd.DataFrame) -> Path:
    """Generate a 6-panel dashboard PNG."""
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                          "axes.spines.top": False, "axes.spines.right": False})

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("PredictiveCare — System Performance Dashboard", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    T1_C = ["#4CAF50","#2196F3","#FF5722","#FF9800","#E91E63","#9C27B0","#795548","#00BCD4"]
    T2_C = ["#4CAF50","#FFC107","#FF9800","#F44336"]

    # 1 — Fault class distribution (Track 1)
    t1 = df[df["track"] == 1]
    if not t1.empty and "predicted_class" in t1.columns:
        counts = t1["predicted_class"].value_counts().sort_index()
        labels = [FAULT_LABELS.get(i, str(i)) for i in counts.index]
        ax1.bar([l.replace(" ", "\n") for l in labels], counts.values,
                color=[T1_C[i % len(T1_C)] for i in counts.index], alpha=0.85, edgecolor="white")
    ax1.set_title("Track 1 — Fault Class Distribution", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Count"); ax1.tick_params(axis="x", labelsize=7)

    # 2 — Risk class distribution (Track 2)
    t2 = df[df["track"] == 2]
    if not t2.empty and "predicted_class" in t2.columns:
        counts2 = t2["predicted_class"].value_counts().sort_index()
        labels2 = [RISK_LABELS.get(i, str(i)) for i in counts2.index]
        ax2.bar(labels2, counts2.values,
                color=[T2_C[i % len(T2_C)] for i in counts2.index], alpha=0.85, edgecolor="white")
    ax2.set_title("Track 2 — Risk Class Distribution", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Count"); ax2.tick_params(axis="x", labelsize=8)

    # 3 — Model accuracy bar
    if not eval_summary.empty:
        tracks = [f"Track {int(r['track'])}" for _, r in eval_summary.iterrows()]
        accs   = [r["accuracy"]   for _, r in eval_summary.iterrows()]
        f1s    = [r["f1_weighted"] for _, r in eval_summary.iterrows()]
        x = range(len(tracks)); w = 0.35
        ax3.bar([i - w/2 for i in x], accs, w, color="#1565C0", alpha=0.85, label="Accuracy")
        ax3.bar([i + w/2 for i in x], f1s,  w, color="#43A047", alpha=0.85, label="F1 (Weighted)")
        ax3.set_xticks(list(x)); ax3.set_xticklabels(tracks)
        ax3.set_ylim(0.8, 1.02); ax3.set_ylabel("Score"); ax3.legend(fontsize=8)
    ax3.set_title("ML Model Accuracy", fontsize=9, fontweight="bold")

    # 4 — Response time histogram
    resp = df[df["response_time_ms"] > 0]["response_time_ms"]
    if not resp.empty:
        ax4.hist(resp, bins=15, color="#1565C0", alpha=0.80, edgecolor="white")
        ax4.axvline(resp.mean(),         color="orange", linestyle="--", linewidth=1.2, label=f"Mean {resp.mean():.0f}ms")
        ax4.axvline(resp.quantile(0.95), color="red",    linestyle="--", linewidth=1.2, label=f"p95  {resp.quantile(0.95):.0f}ms")
        ax4.legend(fontsize=7.5)
    ax4.set_title("LLM Response Time Distribution", fontsize=9, fontweight="bold")
    ax4.set_xlabel("ms"); ax4.set_ylabel("Count")

    # 5 — Cost per query over time
    llm_df = df[df["llm_called"] == True].copy()
    if not llm_df.empty and "timestamp" in llm_df.columns:
        try:
            llm_df["timestamp"] = pd.to_datetime(llm_df["timestamp"])
            llm_df = llm_df.sort_values("timestamp")
            ax5.plot(range(len(llm_df)), llm_df["cost_usd"] * 1000,
                     color="#E91E63", linewidth=1.5, alpha=0.8)
            ax5.set_xlabel("Query #"); ax5.set_ylabel("Cost (milli-USD)")
        except Exception:
            ax5.text(0.5, 0.5, "Cost data\nnot available", ha="center", va="center",
                     transform=ax5.transAxes, color="grey")
    ax5.set_title("Cost per LLM Query", fontsize=9, fontweight="bold")

    # 6 — Correct vs incorrect
    correct_counts = df.groupby("track")["correct"].value_counts().unstack(fill_value=0)
    if not correct_counts.empty:
        correct_counts.plot(kind="bar", ax=ax6, color=["#F44336", "#4CAF50"],
                            edgecolor="white", alpha=0.85)
        ax6.set_xticklabels([f"Track {int(t)}" for t in correct_counts.index], rotation=0)
        ax6.set_ylabel("Count")
        ax6.legend(["Incorrect", "Correct"], fontsize=8)
    ax6.set_title("Prediction Accuracy", fontsize=9, fontweight="bold")

    path = REPORTS_DIR / "dashboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  PredictiveCare Dashboard Generator")
    print("=" * 55)

    df = read_logs_as_df()
    if df.empty:
        print("\n  No log data found. Run eval_runner.py first.\n")
        return

    print(f"\n  Log entries loaded: {len(df)}")

    # Load eval summary if exists
    summary_path = REPORTS_DIR / "eval_summary.csv"
    eval_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    # Cost analysis
    stats     = build_cost_analysis(df)
    cost_path = write_cost_markdown(stats, eval_summary)

    # Dashboard chart
    chart_path = build_dashboard_chart(df, eval_summary)

    print(f"\n  Total queries   : {stats['total_queries']}")
    print(f"  LLM calls made  : {stats['llm_queries']}")
    print(f"  Total cost      : ${stats['total_cost_usd']:.6f}")
    print(f"  Avg response    : {stats['avg_response_ms']} ms")
    print(f"  p95 response    : {stats['p95_response_ms']} ms")

    print(f"\n  Saved:")
    print(f"    {cost_path}")
    print(f"    {chart_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
