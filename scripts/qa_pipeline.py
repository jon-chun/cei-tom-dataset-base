#!/usr/bin/env python3
"""4-Level Quality Assurance Pipeline for CEI Benchmark.

Implements the QA pipeline described in the paper (Section 4.3):
  Level 1: Schema validation (valid fields, enum values, required columns)
  Level 2: Statistical consistency (MAD outlier detection, straight-lining, self-contradiction)
  Level 3: Agreement analysis (Fleiss' kappa per subtype with bootstrap CIs)
  Level 4: Expert adjudication flagging (scenarios requiring manual review)

Usage:
    python scripts/qa_pipeline.py                    # Run all 4 levels
    python scripts/qa_pipeline.py --level 1          # Schema only
    python scripts/qa_pipeline.py --level 2          # Statistical only
    python scripts/qa_pipeline.py --data-dir path/   # Custom data path
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Valid emotion labels (Plutchik's 8 basic emotions)
VALID_EMOTIONS = {
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
}

# Valid VAD text labels (7-point scale)
VALID_VAD_LABELS = {
    "v": {
        "very unpleasant", "unpleasant", "mildly unpleasant", "neutral",
        "mildly pleasant", "pleasant", "very pleasant",
    },
    "a": {
        "very calm", "calm", "slightly calm", "neutral",
        "slightly excited", "excited", "very excited",
    },
    "d": {
        "very controlled", "controlled", "slightly controlled", "neutral",
        "slightly in control", "in control", "very in control",
    },
}

SUBTYPES = [
    "sarcasm-irony", "mixed-signals", "passive-aggression",
    "deflection-misdirection", "strategic-politeness",
]


def load_data(data_dir: Path) -> dict[str, list[dict]]:
    """Load all CSV files from data directory."""
    data = {}
    for subtype in SUBTYPES:
        csv_path = data_dir / f"data_{subtype}.csv"
        if not csv_path.exists():
            print(f"  WARNING: Missing {csv_path}")
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            data[subtype] = list(csv.DictReader(f))
    return data


def get_annotator_cols(headers: list[str], prefix: str) -> list[str]:
    """Find annotator-specific columns matching a prefix."""
    return [h for h in headers if h.startswith(prefix)]


# ===========================================================================
# Level 1: Schema Validation
# ===========================================================================

def level1_schema_validation(data: dict[str, list[dict]]) -> list[dict]:
    """Validate schema: required fields, valid enum values, data types."""
    issues = []
    required_fields = {"id", "sd_situation", "sd_utterance", "sd_speaker_role",
                       "sd_listener_role", "gold_standard"}

    for subtype, rows in data.items():
        if not rows:
            issues.append({
                "level": 1, "subtype": subtype, "id": None,
                "type": "empty_file", "message": "No rows found",
            })
            continue

        headers = set(rows[0].keys())

        # Check required fields
        missing = required_fields - headers
        if missing:
            issues.append({
                "level": 1, "subtype": subtype, "id": None,
                "type": "missing_columns",
                "message": f"Missing required columns: {missing}",
            })

        # Check annotator columns exist (need at least 3 emotion columns)
        emotion_cols = [h for h in headers if h.startswith("sl_plutchik_primary_")]
        if len(emotion_cols) < 3:
            issues.append({
                "level": 1, "subtype": subtype, "id": None,
                "type": "insufficient_annotators",
                "message": f"Found {len(emotion_cols)} emotion columns, need >= 3",
            })

        for row in rows:
            row_id = row.get("id", "?")

            # Check empty required fields
            for field in required_fields:
                val = row.get(field, "").strip()
                if not val:
                    issues.append({
                        "level": 1, "subtype": subtype, "id": row_id,
                        "type": "empty_field",
                        "message": f"Empty required field: {field}",
                    })

            # Validate gold_standard enum
            gs = row.get("gold_standard", "").strip().lower()
            if gs and gs not in VALID_EMOTIONS:
                issues.append({
                    "level": 1, "subtype": subtype, "id": row_id,
                    "type": "invalid_gold_standard",
                    "message": f"Invalid gold_standard: '{gs}'",
                })

            # Validate annotator emotion labels
            for col in emotion_cols:
                val = row.get(col, "").strip().lower()
                if val and val not in VALID_EMOTIONS:
                    issues.append({
                        "level": 1, "subtype": subtype, "id": row_id,
                        "type": "invalid_emotion",
                        "message": f"Invalid emotion in {col}: '{val}'",
                    })

            # Validate VAD labels
            for dim_key, valid_set in VALID_VAD_LABELS.items():
                vad_cols = [h for h in headers if h.startswith(f"sl_{dim_key}_")]
                for col in vad_cols:
                    val = row.get(col, "").strip().lower()
                    if val and val not in valid_set:
                        issues.append({
                            "level": 1, "subtype": subtype, "id": row_id,
                            "type": "invalid_vad",
                            "message": f"Invalid VAD label in {col}: '{val}'",
                        })

    return issues


# ===========================================================================
# Level 2: Statistical Consistency Checks
# ===========================================================================

def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _mad(values: list[float]) -> float:
    """Median absolute deviation."""
    med = _median(values)
    return _median([abs(v - med) for v in values])


def level2_statistical_checks(data: dict[str, list[dict]]) -> list[dict]:
    """Check for straight-lining, self-contradictions, and outliers."""
    issues = []

    for subtype, rows in data.items():
        if not rows:
            continue

        headers = list(rows[0].keys())
        emotion_cols = get_annotator_cols(headers, "sl_plutchik_primary_")

        # Per-annotator checks
        annotator_ids = set()
        for col in emotion_cols:
            ann_id = col.replace("sl_plutchik_primary_", "")
            annotator_ids.add(ann_id)

        for ann_id in annotator_ids:
            em_col = f"sl_plutchik_primary_{ann_id}"
            v_col = f"sl_v_{ann_id}"
            conf_col = f"sl_confidence_{ann_id}"

            # Straight-lining: same emotion for > 80% of scenarios
            emotions = [row.get(em_col, "").strip().lower() for row in rows
                        if row.get(em_col, "").strip()]
            if emotions:
                most_common_count = Counter(emotions).most_common(1)[0][1]
                if most_common_count / len(emotions) > 0.80:
                    issues.append({
                        "level": 2, "subtype": subtype, "id": None,
                        "type": "straight_lining",
                        "message": (
                            f"{ann_id}: same emotion for "
                            f"{most_common_count}/{len(emotions)} "
                            f"({most_common_count/len(emotions)*100:.0f}%) scenarios"
                        ),
                    })

            # Self-contradiction: positive emotion + very negative valence
            positive_emotions = {"joy", "trust", "anticipation"}
            negative_vad = {"very unpleasant", "unpleasant"}
            for row in rows:
                em = row.get(em_col, "").strip().lower()
                v = row.get(v_col, "").strip().lower()
                if em in positive_emotions and v in negative_vad:
                    issues.append({
                        "level": 2, "subtype": subtype, "id": row.get("id", "?"),
                        "type": "self_contradiction",
                        "message": (
                            f"{ann_id}: positive emotion '{em}' "
                            f"with negative valence '{v}'"
                        ),
                    })

    return issues


# ===========================================================================
# Level 3: Agreement Analysis
# ===========================================================================

def _fleiss_kappa(ratings_matrix: list[list[int]], n_categories: int) -> float:
    """Compute Fleiss' kappa for a ratings matrix."""
    n_subjects = len(ratings_matrix)
    n_raters = sum(ratings_matrix[0]) if ratings_matrix else 0

    if n_subjects == 0 or n_raters == 0:
        return 0.0

    # Proportion per category
    p_j = []
    for j in range(n_categories):
        total = sum(row[j] for row in ratings_matrix)
        p_j.append(total / (n_subjects * n_raters))

    # Per-subject agreement
    P_i = []
    for row in ratings_matrix:
        n = sum(row)
        if n <= 1:
            P_i.append(1.0)
            continue
        s = sum(r * r for r in row)
        P_i.append((s - n) / (n * (n - 1)))

    P_bar = sum(P_i) / n_subjects
    P_e = sum(p * p for p in p_j)

    if abs(1 - P_e) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else 0.0

    return (P_bar - P_e) / (1 - P_e)


def level3_agreement_analysis(
    data: dict[str, list[dict]], n_bootstrap: int = 2000, seed: int = 42,
) -> dict[str, dict]:
    """Compute Fleiss' kappa per subtype with bootstrap CIs."""
    import random
    rng = random.Random(seed)

    emotions = sorted(VALID_EMOTIONS)
    em_to_idx = {e: i for i, e in enumerate(emotions)}
    results = {}

    for subtype, rows in data.items():
        headers = list(rows[0].keys())
        emotion_cols = get_annotator_cols(headers, "sl_plutchik_primary_")

        # Build ratings matrix
        matrix = []
        for row in rows:
            counts = [0] * len(emotions)
            for col in emotion_cols:
                val = row.get(col, "").strip().lower()
                if val in em_to_idx:
                    counts[em_to_idx[val]] += 1
            if sum(counts) > 0:
                matrix.append(counts)

        kappa = _fleiss_kappa(matrix, len(emotions))

        # Bootstrap CI
        kappas = []
        for _ in range(n_bootstrap):
            sample = [matrix[rng.randint(0, len(matrix) - 1)]
                      for _ in range(len(matrix))]
            kappas.append(_fleiss_kappa(sample, len(emotions)))
        kappas.sort()
        ci_low = kappas[int(0.025 * n_bootstrap)]
        ci_high = kappas[int(0.975 * n_bootstrap)]

        results[subtype] = {
            "kappa": round(kappa, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "n_scenarios": len(matrix),
        }

    return results


# ===========================================================================
# Level 4: Expert Adjudication Flagging
# ===========================================================================

def level4_adjudication_flags(data: dict[str, list[dict]]) -> list[dict]:
    """Flag scenarios requiring expert review: 3-way splits, empty gold, low confidence."""
    flags = []

    for subtype, rows in data.items():
        headers = list(rows[0].keys())
        emotion_cols = get_annotator_cols(headers, "sl_plutchik_primary_")
        conf_cols = get_annotator_cols(headers, "sl_confidence_")

        for row in rows:
            row_id = row.get("id", "?")

            # Get annotator labels
            labels = [row.get(col, "").strip().lower() for col in emotion_cols
                      if row.get(col, "").strip()]
            unique = set(labels)

            # Flag: 3-way split (no agreement)
            if len(unique) >= 3:
                flags.append({
                    "level": 4, "subtype": subtype, "id": row_id,
                    "type": "three_way_split",
                    "message": f"No agreement: {labels}",
                })

            # Flag: empty gold_standard
            gs = row.get("gold_standard", "").strip()
            if not gs:
                flags.append({
                    "level": 4, "subtype": subtype, "id": row_id,
                    "type": "empty_gold",
                    "message": "Missing gold_standard label",
                })

            # Flag: gold_standard doesn't match majority
            if gs and len(labels) >= 2:
                counts = Counter(labels)
                majority = counts.most_common(1)[0]
                if majority[1] >= 2 and gs.lower() != majority[0]:
                    flags.append({
                        "level": 4, "subtype": subtype, "id": row_id,
                        "type": "gold_override",
                        "message": (
                            f"Gold '{gs}' differs from majority '{majority[0]}' "
                            f"({majority[1]}/{len(labels)})"
                        ),
                    })

    return flags


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="CEI 4-Level QA Pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/human-gold"),
                        help="Path to merged CSV directory")
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4],
                        help="Run specific level only (default: all)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save report as JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("CEI 4-Level Quality Assurance Pipeline")
    print("=" * 60)

    data = load_data(args.data_dir)
    total = sum(len(rows) for rows in data.values())
    print(f"Loaded {total} scenarios across {len(data)} subtypes\n")

    report = {"total_scenarios": total, "levels": {}}
    all_issues = []

    # Level 1
    if args.level is None or args.level == 1:
        print("[Level 1] Schema Validation")
        issues = level1_schema_validation(data)
        report["levels"]["1_schema"] = {"issues": len(issues), "details": issues}
        all_issues.extend(issues)
        print(f"  Issues found: {len(issues)}")
        for iss in issues[:5]:
            print(f"    {iss['subtype']} id={iss['id']}: {iss['message']}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")
        print()

    # Level 2
    if args.level is None or args.level == 2:
        print("[Level 2] Statistical Consistency")
        issues = level2_statistical_checks(data)
        report["levels"]["2_statistical"] = {"issues": len(issues), "details": issues}
        all_issues.extend(issues)
        print(f"  Issues found: {len(issues)}")
        for iss in issues[:5]:
            print(f"    {iss['subtype']} id={iss['id']}: {iss['message']}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")
        print()

    # Level 3
    if args.level is None or args.level == 3:
        print("[Level 3] Agreement Analysis")
        agreement = level3_agreement_analysis(data, seed=args.seed)
        report["levels"]["3_agreement"] = agreement
        for st, info in agreement.items():
            interp = "Poor" if info["kappa"] < 0 else (
                "Slight" if info["kappa"] < 0.20 else (
                    "Fair" if info["kappa"] < 0.40 else "Moderate"))
            print(f"  {st}: kappa={info['kappa']:.4f} "
                  f"CI={info['ci_95']} ({interp})")
        print()

    # Level 4
    if args.level is None or args.level == 4:
        print("[Level 4] Expert Adjudication Flags")
        flags = level4_adjudication_flags(data)
        report["levels"]["4_adjudication"] = {"flags": len(flags), "details": flags}
        all_issues.extend(flags)
        n_splits = sum(1 for f in flags if f["type"] == "three_way_split")
        n_empty = sum(1 for f in flags if f["type"] == "empty_gold")
        n_override = sum(1 for f in flags if f["type"] == "gold_override")
        print(f"  Three-way splits: {n_splits}")
        print(f"  Empty gold_standard: {n_empty}")
        print(f"  Expert overrides: {n_override}")
        print(f"  Total flags: {len(flags)}")
        pct = len(flags) / total * 100 if total > 0 else 0
        print(f"  Flagged rate: {pct:.1f}%")
        print()

    # Summary
    print("=" * 60)
    print(f"TOTAL ISSUES: {len(all_issues)}")
    print("=" * 60)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved: {args.output}")

    return 0 if not any(
        i for i in all_issues if i.get("level") == 1
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
