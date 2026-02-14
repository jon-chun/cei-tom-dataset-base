"""Generate model-level confusion matrix figure for DMLR 2026 paper.

Reads baseline_results.json (produced by run_pipeline_dmlr2026.py --stage run_baselines)
and generates an 8x8 confusion matrix aggregated across all models, analogous to
the human annotator confusion matrix (fig1_confusion_matrix.pdf).

Usage:
    python scripts/generate_model_confusion_matrix.py

Outputs:
    reports/dmlr2026/figures/fig8_model_confusion_matrix.pdf
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EMOTIONS = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "reports" / "dmlr2026" / "baseline_results.json"
OUTPUT_PATH = REPO_ROOT / "reports" / "dmlr2026" / "figures" / "fig8_model_confusion_matrix.pdf"


def load_gold_labels() -> dict[str, str]:
    """Load gold-standard emotion labels from human annotation CSVs."""
    import csv

    gold = {}
    data_dir = REPO_ROOT / "data" / "human-gold"
    for csv_path in data_dir.glob("data_*.csv"):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario_id = row.get("id", "")
                gold_label = row.get("gold_standard", "").strip().lower()
                if scenario_id and gold_label:
                    gold[scenario_id] = gold_label
    return gold


def build_confusion_matrix(
    results: dict, gold: dict[str, str]
) -> np.ndarray:
    """Build 8x8 confusion matrix: rows=gold, cols=predicted, aggregated across models."""
    emo_to_idx = {e: i for i, e in enumerate(EMOTIONS)}
    matrix = np.zeros((8, 8), dtype=int)

    for model_name, predictions in results.items():
        for pred in predictions:
            sid = str(pred.get("id", ""))
            raw_predicted = pred.get("predicted")
            if raw_predicted is None:
                continue
            predicted = raw_predicted.strip().lower()
            gold_label = gold.get(sid, pred.get("gold", "").strip().lower())

            if gold_label in emo_to_idx and predicted in emo_to_idx:
                matrix[emo_to_idx[gold_label], emo_to_idx[predicted]] += 1

    return matrix


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    """Plot row-normalized confusion matrix matching the style of the human version."""
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    norm_matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm_matrix, cmap="YlOrRd", vmin=0, vmax=0.5)

    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    labels = [e.capitalize() for e in EMOTIONS]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted Emotion", fontsize=12)
    ax.set_ylabel("Gold Emotion", fontsize=12)

    # Annotate cells
    for i in range(8):
        for j in range(8):
            val = norm_matrix[i, j]
            color = "white" if val > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Proportion", shrink=0.8)
    plt.title("Model Prediction Confusion Matrix\n(aggregated across 7 models, row-normalized)", fontsize=12)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main() -> None:
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found.")
        print("Run baselines first: python scripts/run_pipeline_dmlr2026.py --stage run_baselines")
        sys.exit(1)

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    gold = load_gold_labels()
    if not gold:
        print("WARNING: Could not load gold labels from CSVs, falling back to 'gold' field in results")

    matrix = build_confusion_matrix(results, gold)
    total = matrix.sum()
    print(f"Total predictions: {total} ({total // 300} models x 300 scenarios)")
    print(f"Overall accuracy: {matrix.trace() / total:.3f}")

    plot_confusion_matrix(matrix, OUTPUT_PATH)


if __name__ == "__main__":
    main()
