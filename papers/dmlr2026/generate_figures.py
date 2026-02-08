#!/usr/bin/env python3
"""Generate figures for CEI DMLR paper."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# Set style for publication
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

# Paths — relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = REPO_ROOT
OUTPUT_PATH = REPO_ROOT / "papers" / "dmlr2026" / "figures"
OUTPUT_PATH.mkdir(exist_ok=True)


def load_data():
    """Load benchmark and QA data."""
    with open(DATASET_PATH / "data/processed/cei_benchmark.json") as f:
        benchmark = json.load(f)

    with open(DATASET_PATH / "data/qa_reports/confusion_matrix.json") as f:
        confusion = json.load(f)

    return benchmark, confusion


def fig1_confusion_matrix(confusion: dict):
    """Generate 8x8 emotion confusion matrix heatmap."""
    emotions = ["anger", "anticipation", "disgust", "fear",
                "joy", "sadness", "surprise", "trust"]
    emotion_labels = ["Anger", "Anticip.", "Disgust", "Fear",
                      "Joy", "Sadness", "Surprise", "Trust"]

    # Build matrix
    matrix = np.zeros((8, 8))
    overall = confusion["overall"]["matrix"]

    for i, e1 in enumerate(emotions):
        for j, e2 in enumerate(emotions):
            matrix[i, j] = overall[e1][e2]

    # Normalize by row (to show confusion proportions)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / row_sums

    # Create figure - single column width (3.25 inches for two-column)
    fig, ax = plt.subplots(figsize=(3.25, 2.8))

    # Custom colormap: white to blue
    cmap = plt.cm.Blues

    im = ax.imshow(matrix_norm, cmap=cmap, aspect='auto', vmin=0, vmax=0.35)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Proportion', rotation=-90, va="bottom", fontsize=8)

    # Set ticks
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels(emotion_labels, rotation=45, ha="right")
    ax.set_yticklabels(emotion_labels)

    # Labels
    ax.set_xlabel('Annotator 2')
    ax.set_ylabel('Annotator 1')

    # Add text annotations for high values
    for i in range(8):
        for j in range(8):
            val = matrix_norm[i, j]
            if val > 0.15:
                text_color = 'white' if val > 0.25 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=text_color, fontsize=6)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_PATH / "fig1_confusion_matrix.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_PATH / "fig1_confusion_matrix.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: fig1_confusion_matrix.pdf/png")


def fig2_difficulty_distribution(benchmark: dict):
    """Generate scenario difficulty distribution based on annotator agreement."""
    scenarios = benchmark["scenarios"]

    # Categorize by agreement level
    unanimous = 0  # 100% agreement (3/3)
    majority = 0   # 67% agreement (2/3)
    split = 0      # 33% agreement (1/3 each)

    for s in scenarios:
        agreement = s["ground_truth"]["annotator_agreement"]
        if agreement >= 0.99:
            unanimous += 1
        elif agreement >= 0.65:
            majority += 1
        else:
            split += 1

    # Also by subtype
    subtypes = ["sarcasm-irony", "mixed-signals", "strategic-politeness",
                "passive-aggression", "deflection-misdirection"]
    subtype_labels = ["Sarcasm", "Mixed\nSignals", "Strategic\nPoliteness",
                      "Passive\nAggression", "Deflection"]

    subtype_difficulty = {st: {"easy": 0, "medium": 0, "hard": 0} for st in subtypes}

    for s in scenarios:
        st = s["subtype"]
        agreement = s["ground_truth"]["annotator_agreement"]
        if agreement >= 0.99:
            subtype_difficulty[st]["easy"] += 1
        elif agreement >= 0.65:
            subtype_difficulty[st]["medium"] += 1
        else:
            subtype_difficulty[st]["hard"] += 1

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # Left: Overall distribution (pie chart)
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
    labels = [f'Unanimous\n({unanimous})', f'Majority\n({majority})', f'Split\n({split})']
    sizes = [unanimous, majority, split]

    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 8})
    ax1.set_title('(a) Overall Difficulty Distribution', fontsize=9)

    # Right: By subtype (stacked bar)
    x = np.arange(len(subtypes))
    width = 0.6

    easy = [subtype_difficulty[st]["easy"] for st in subtypes]
    medium = [subtype_difficulty[st]["medium"] for st in subtypes]
    hard = [subtype_difficulty[st]["hard"] for st in subtypes]

    ax2.bar(x, easy, width, label='Unanimous', color='#2ecc71')
    ax2.bar(x, medium, width, bottom=easy, label='Majority', color='#f39c12')
    ax2.bar(x, hard, width, bottom=np.array(easy)+np.array(medium),
            label='Split', color='#e74c3c')

    ax2.set_ylabel('Number of Scenarios')
    ax2.set_xticks(x)
    ax2.set_xticklabels(subtype_labels, fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    ax2.set_title('(b) Difficulty by Subtype', fontsize=9)
    ax2.set_ylim(0, 65)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_PATH / "fig2_difficulty_distribution.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_PATH / "fig2_difficulty_distribution.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: fig2_difficulty_distribution.pdf/png")

    # Print stats
    print(f"  Unanimous: {unanimous} ({unanimous/300*100:.1f}%)")
    print(f"  Majority: {majority} ({majority/300*100:.1f}%)")
    print(f"  Split: {split} ({split/300*100:.1f}%)")


def fig3_linguistic_analysis(benchmark: dict):
    """Generate linguistic feature analysis (length vs agreement)."""
    scenarios = benchmark["scenarios"]

    # Extract features
    context_lengths = []
    utterance_lengths = []
    agreements = []
    subtypes = []

    for s in scenarios:
        context_lengths.append(len(s["context"].split()))
        utterance_lengths.append(len(s["utterance"].split()))
        agreements.append(s["ground_truth"]["annotator_agreement"])
        subtypes.append(s["subtype"])

    context_lengths = np.array(context_lengths)
    utterance_lengths = np.array(utterance_lengths)
    agreements = np.array(agreements)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.5))

    # Left: Context length vs agreement
    ax1.scatter(context_lengths, agreements, alpha=0.4, s=15, c='#3498db')

    # Add trend line
    z = np.polyfit(context_lengths, agreements, 1)
    p = np.poly1d(z)
    x_line = np.linspace(context_lengths.min(), context_lengths.max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1.5, label=f'r={np.corrcoef(context_lengths, agreements)[0,1]:.2f}')

    ax1.set_xlabel('Context Length (words)')
    ax1.set_ylabel('Annotator Agreement')
    ax1.set_title('(a) Context Length vs Agreement', fontsize=9)
    ax1.legend(loc='upper right', fontsize=7)
    ax1.set_ylim(0, 1.1)

    # Right: Utterance length vs agreement
    ax2.scatter(utterance_lengths, agreements, alpha=0.4, s=15, c='#9b59b6')

    # Add trend line
    z = np.polyfit(utterance_lengths, agreements, 1)
    p = np.poly1d(z)
    x_line = np.linspace(utterance_lengths.min(), utterance_lengths.max(), 100)
    ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=1.5, label=f'r={np.corrcoef(utterance_lengths, agreements)[0,1]:.2f}')

    ax2.set_xlabel('Utterance Length (words)')
    ax2.set_ylabel('Annotator Agreement')
    ax2.set_title('(b) Utterance Length vs Agreement', fontsize=9)
    ax2.legend(loc='upper right', fontsize=7)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_PATH / "fig3_linguistic_analysis.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_PATH / "fig3_linguistic_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: fig3_linguistic_analysis.pdf/png")

    # Print correlations
    r_context = np.corrcoef(context_lengths, agreements)[0,1]
    r_utterance = np.corrcoef(utterance_lengths, agreements)[0,1]
    print(f"  Context length correlation: r={r_context:.3f}")
    print(f"  Utterance length correlation: r={r_utterance:.3f}")
    print(f"  Mean context length: {context_lengths.mean():.1f} words")
    print(f"  Mean utterance length: {utterance_lengths.mean():.1f} words")


def fig4_agreement_by_subtype(confusion: dict):
    """Generate agreement rate by subtype bar chart."""
    subtypes = ["sarcasm-irony", "mixed-signals", "strategic-politeness",
                "passive-aggression", "deflection-misdirection"]
    subtype_labels = ["Sarcasm/\nIrony", "Mixed\nSignals", "Strategic\nPoliteness",
                      "Passive\nAggression", "Deflection"]

    # Get agreement rates from confusion matrix
    agreement_rates = []
    for st in subtypes:
        st_key = st  # Keys match
        if st_key in confusion["by_subtype"]:
            agreement_rates.append(confusion["by_subtype"][st_key]["agreement_rate"])
        else:
            agreement_rates.append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    x = np.arange(len(subtypes))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(subtypes)))

    bars = ax.bar(x, [r * 100 for r in agreement_rates], color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, rate in zip(bars, agreement_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Agreement Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(subtype_labels, fontsize=7)
    ax.set_ylim(0, 30)
    ax.axhline(y=16.5, color='red', linestyle='--', linewidth=1, label='Overall (16.5%)')
    ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout()

    # Save
    fig.savefig(OUTPUT_PATH / "fig4_agreement_by_subtype.pdf", bbox_inches='tight')
    fig.savefig(OUTPUT_PATH / "fig4_agreement_by_subtype.png", bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: fig4_agreement_by_subtype.pdf/png")


def main():
    print("Loading data...")
    benchmark, confusion = load_data()

    print("\nGenerating figures...")
    print("-" * 40)

    print("\n1. Emotion Confusion Matrix:")
    fig1_confusion_matrix(confusion)

    print("\n2. Difficulty Distribution:")
    fig2_difficulty_distribution(benchmark)

    print("\n3. Linguistic Analysis:")
    fig3_linguistic_analysis(benchmark)

    print("\n4. Agreement by Subtype:")
    fig4_agreement_by_subtype(confusion)

    print("\n" + "-" * 40)
    print(f"All figures saved to: {OUTPUT_PATH}")
    print("\nFigures generated:")
    for f in sorted(OUTPUT_PATH.glob("*.pdf")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
