#!/usr/bin/env python3
"""Generate figures for CEI-ToM DMLR 2026 paper.

Reads directly from data/human-gold/ CSV files and generates all 4 paper figures.
No intermediate JSON files required.

Usage:
    python scripts/generate_figures.py
"""

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication style
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "human-gold"
OUTPUT_DIR = REPO_ROOT / "reports" / "dmlr2026" / "figures"

EMOTIONS = ["anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust"]
EMOTION_LABELS = ["Anger", "Anticip.", "Disgust", "Fear",
                  "Joy", "Sadness", "Surprise", "Trust"]

SUBTYPES = ["sarcasm-irony", "mixed-signals", "strategic-politeness",
            "passive-aggression", "deflection-misdirection"]
SUBTYPE_LABELS_SHORT = ["Sarcasm/\nIrony", "Mixed\nSignals", "Strategic\nPoliteness",
                        "Passive\nAggression", "Deflection"]
SUBTYPE_LABELS_FIG2 = ["Sarcasm", "Mixed\nSignals", "Strategic\nPoliteness",
                       "Passive\nAggression", "Deflection"]


def load_all_data():
    """Load all CSV files and return list of dicts with subtype and annotator labels."""
    all_rows = []
    for csv_path in sorted(DATA_DIR.glob("data_*.csv")):
        subtype = csv_path.stem.replace("data_", "")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            # Find annotator emotion columns
            emotion_cols = [h for h in headers if h.startswith("sl_plutchik_primary_")]
            for row in reader:
                labels = [row[col].strip().lower() for col in emotion_cols if row[col].strip()]
                # Filter to valid emotions only
                labels = [l for l in labels if l in EMOTIONS]
                all_rows.append({
                    "id": row["id"],
                    "subtype": subtype,
                    "labels": labels,
                    "gold_standard": row.get("gold_standard", "").strip().lower(),
                    "situation": row.get("sd_situation", ""),
                    "utterance": row.get("sd_utterance", ""),
                })
    return all_rows


def compute_confusion_matrix(rows):
    """Compute 8x8 pairwise annotator confusion matrix.

    For each scenario with 3 annotators, generates 3 pairs: (A1,A2), (A1,A3), (A2,A3).
    Returns dict with 'overall' and 'by_subtype' matrices.
    """
    def empty_matrix():
        return {e1: {e2: 0 for e2 in EMOTIONS} for e1 in EMOTIONS}

    overall = empty_matrix()
    by_subtype = {st: {"matrix": empty_matrix(), "total_pairs": 0, "agreement_count": 0}
                  for st in SUBTYPES}

    total_pairs = 0
    total_agreement = 0

    for row in rows:
        labels = row["labels"]
        if len(labels) < 2:
            continue
        subtype = row["subtype"]
        for l1, l2 in itertools.combinations(labels, 2):
            if l1 == l2:
                # Diagonal: add once (cell is its own symmetric partner)
                overall[l1][l2] += 1
                by_subtype[subtype]["matrix"][l1][l2] += 1
                by_subtype[subtype]["agreement_count"] += 1
                total_agreement += 1
            else:
                # Off-diagonal: add to both (i,j) and (j,i)
                overall[l1][l2] += 1
                overall[l2][l1] += 1
                by_subtype[subtype]["matrix"][l1][l2] += 1
                by_subtype[subtype]["matrix"][l2][l1] += 1
            by_subtype[subtype]["total_pairs"] += 1
            total_pairs += 1

    # Compute agreement rates
    for st in SUBTYPES:
        tp = by_subtype[st]["total_pairs"]
        by_subtype[st]["agreement_rate"] = by_subtype[st]["agreement_count"] / tp if tp > 0 else 0

    return {
        "overall": {"matrix": overall, "total_pairs": total_pairs,
                     "agreement_count": total_agreement,
                     "agreement_rate": total_agreement / total_pairs if total_pairs > 0 else 0},
        "by_subtype": by_subtype,
    }


def compute_agreement_per_scenario(rows):
    """Classify each scenario as unanimous (3/3), majority (2/3), or split (1/1/1)."""
    results = []
    for row in rows:
        labels = row["labels"]
        if len(labels) < 3:
            # For scenarios with fewer than 3 valid labels, skip
            continue
        unique = set(labels)
        if len(unique) == 1:
            agreement_level = "unanimous"
            agreement_score = 1.0
        elif len(unique) == 2:
            agreement_level = "majority"
            agreement_score = 2 / 3
        else:
            agreement_level = "split"
            agreement_score = 1 / 3
        results.append({
            "subtype": row["subtype"],
            "level": agreement_level,
            "score": agreement_score,
            "context_words": len(row["situation"].split()),
            "utterance_words": len(row["utterance"].split()),
        })
    return results


def fig1_confusion_matrix(confusion):
    """Generate 8x8 emotion confusion matrix heatmap."""
    matrix = np.zeros((8, 8))
    overall = confusion["overall"]["matrix"]

    for i, e1 in enumerate(EMOTIONS):
        for j, e2 in enumerate(EMOTIONS):
            matrix[i, j] = overall[e1][e2]

    # Normalize by row
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    matrix_norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(3.25, 2.8))
    im = ax.imshow(matrix_norm, cmap=plt.cm.Blues, aspect="auto", vmin=0, vmax=0.35)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Proportion", rotation=-90, va="bottom", fontsize=8)

    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(EMOTION_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(EMOTION_LABELS)
    ax.set_xlabel("Annotator 2")
    ax.set_ylabel("Annotator 1")

    for i in range(8):
        for j in range(8):
            val = matrix_norm[i, j]
            if val > 0.15:
                text_color = "white" if val > 0.25 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=6)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_confusion_matrix.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig1_confusion_matrix.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  Generated: fig1_confusion_matrix.pdf/png")


def fig2_difficulty_distribution(agreement_data):
    """Generate scenario difficulty distribution."""
    # Overall counts
    counts = {"unanimous": 0, "majority": 0, "split": 0}
    subtype_counts = {st: {"unanimous": 0, "majority": 0, "split": 0} for st in SUBTYPES}

    for item in agreement_data:
        counts[item["level"]] += 1
        subtype_counts[item["subtype"]][item["level"]] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # Left: pie chart
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = [f'Unanimous\n({counts["unanimous"]})',
              f'Majority\n({counts["majority"]})',
              f'Split\n({counts["split"]})']
    sizes = [counts["unanimous"], counts["majority"], counts["split"]]

    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
            startangle=90, textprops={"fontsize": 8})
    ax1.set_title("(a) Overall Difficulty Distribution", fontsize=9)

    # Right: stacked bar by subtype
    x = np.arange(len(SUBTYPES))
    width = 0.6
    easy = [subtype_counts[st]["unanimous"] for st in SUBTYPES]
    medium = [subtype_counts[st]["majority"] for st in SUBTYPES]
    hard = [subtype_counts[st]["split"] for st in SUBTYPES]

    ax2.bar(x, easy, width, label="Unanimous", color="#2ecc71")
    ax2.bar(x, medium, width, bottom=easy, label="Majority", color="#f39c12")
    ax2.bar(x, hard, width, bottom=np.array(easy) + np.array(medium),
            label="Split", color="#e74c3c")

    ax2.set_ylabel("Number of Scenarios")
    ax2.set_xticks(x)
    ax2.set_xticklabels(SUBTYPE_LABELS_FIG2, fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)
    ax2.set_title("(b) Difficulty by Subtype", fontsize=9)
    ax2.set_ylim(0, 65)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_difficulty_distribution.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig2_difficulty_distribution.png", bbox_inches="tight", dpi=300)
    plt.close()

    total = sum(sizes)
    print(f"  Generated: fig2_difficulty_distribution.pdf/png")
    print(f"    Unanimous: {counts['unanimous']} ({counts['unanimous']/total*100:.1f}%)")
    print(f"    Majority: {counts['majority']} ({counts['majority']/total*100:.1f}%)")
    print(f"    Split: {counts['split']} ({counts['split']/total*100:.1f}%)")


def fig3_linguistic_analysis(agreement_data):
    """Generate linguistic feature analysis (length vs agreement)."""
    context_lengths = np.array([d["context_words"] for d in agreement_data])
    utterance_lengths = np.array([d["utterance_words"] for d in agreement_data])
    agreements = np.array([d["score"] for d in agreement_data])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # Left: context length vs agreement
    ax1.scatter(context_lengths, agreements, alpha=0.4, s=15, c="#3498db")
    z = np.polyfit(context_lengths, agreements, 1)
    p = np.poly1d(z)
    x_line = np.linspace(context_lengths.min(), context_lengths.max(), 100)
    rho_context = spearmanr(context_lengths, agreements).statistic
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1.5,
             label=f"\u03c1={rho_context:.2f}")
    ax1.set_xlabel("Context Length (words)")
    ax1.set_ylabel("Annotator Agreement")
    ax1.set_title("(a) Context Length vs Agreement", fontsize=9)
    ax1.legend(loc="upper right", fontsize=7)
    ax1.set_ylim(0, 1.1)

    # Right: utterance length vs agreement
    ax2.scatter(utterance_lengths, agreements, alpha=0.4, s=15, c="#9b59b6")
    z = np.polyfit(utterance_lengths, agreements, 1)
    p = np.poly1d(z)
    x_line = np.linspace(utterance_lengths.min(), utterance_lengths.max(), 100)
    rho_utterance = spearmanr(utterance_lengths, agreements).statistic
    ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1.5,
             label=f"\u03c1={rho_utterance:.2f}")
    ax2.set_xlabel("Utterance Length (words)")
    ax2.set_ylabel("Annotator Agreement")
    ax2.set_title("(b) Utterance Length vs Agreement", fontsize=9)
    ax2.legend(loc="upper right", fontsize=7)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_linguistic_analysis.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig3_linguistic_analysis.png", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"  Generated: fig3_linguistic_analysis.pdf/png")
    print(f"    Context length Spearman rho: {rho_context:.3f}")
    print(f"    Utterance length Spearman rho: {rho_utterance:.3f}")
    print(f"    Mean context length: {context_lengths.mean():.1f} words")
    print(f"    Mean utterance length: {utterance_lengths.mean():.1f} words")


def fig4_agreement_by_subtype(confusion):
    """Generate agreement rate by subtype bar chart."""
    agreement_rates = []
    for st in SUBTYPES:
        rate = confusion["by_subtype"][st]["agreement_rate"]
        agreement_rates.append(rate)

    overall_rate = confusion["overall"]["agreement_rate"]

    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    x = np.arange(len(SUBTYPES))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(SUBTYPES)))

    bars = ax.bar(x, [r * 100 for r in agreement_rates], color=colors,
                  edgecolor="black", linewidth=0.5)

    for bar, rate in zip(bars, agreement_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate*100:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Agreement Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(SUBTYPE_LABELS_SHORT, fontsize=7)
    ax.set_ylim(0, max(r * 100 for r in agreement_rates) + 10)
    ax.axhline(y=overall_rate * 100, color="red", linestyle="--", linewidth=1,
               label=f"Overall ({overall_rate*100:.1f}%)")
    ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_agreement_by_subtype.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4_agreement_by_subtype.png", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Generated: fig4_agreement_by_subtype.pdf/png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data from CSV files...")
    rows = load_all_data()
    print(f"  Loaded {len(rows)} scenarios")

    print("\nComputing confusion matrix...")
    confusion = compute_confusion_matrix(rows)
    print(f"  Total pairs: {confusion['overall']['total_pairs']}")
    print(f"  Overall agreement rate: {confusion['overall']['agreement_rate']:.4f}")

    # Save confusion matrix JSON for reference
    json_path = OUTPUT_DIR.parent / "confusion_matrix.json"
    with open(json_path, "w") as f:
        json.dump(confusion, f, indent=2)
    print(f"  Saved: {json_path}")

    print("\nComputing per-scenario agreement...")
    agreement_data = compute_agreement_per_scenario(rows)
    print(f"  {len(agreement_data)} scenarios classified")

    print("\nGenerating figures...")
    print("-" * 40)

    print("\n1. Emotion Confusion Matrix:")
    fig1_confusion_matrix(confusion)

    print("\n2. Difficulty Distribution:")
    fig2_difficulty_distribution(agreement_data)

    print("\n3. Linguistic Analysis:")
    fig3_linguistic_analysis(agreement_data)

    print("\n4. Agreement by Subtype:")
    fig4_agreement_by_subtype(confusion)

    print("\n" + "-" * 40)
    print(f"All figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
