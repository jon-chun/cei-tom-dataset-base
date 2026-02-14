#!/usr/bin/env python3
"""DMLR 2026 Pipeline Runner — CEI-ToM Dataset Paper.

Implements all computational tasks for the DMLR dataset paper (R0–R9):
  Phase 0 (local):  Data verification, κ, power distribution, human performance,
                     VAD analysis, scale justification, stratified splits, examples
  Phase 1 (API):    CogSci-safe baseline inference (models loaded from
                     config/config-dmlr.yml — none overlap with CogSci 2026)
  Phase 2 (local):  Baseline analysis, LaTeX tables, figures

Usage:
    python scripts/run_pipeline_dmlr2026.py --stage all_local
    python scripts/run_pipeline_dmlr2026.py --stage run_baselines --dry-run
    python scripts/run_pipeline_dmlr2026.py --stage generate_outputs
    python scripts/run_pipeline_dmlr2026.py --stage all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBTYPES = [
    "deflection-misdirection",
    "mixed-signals",
    "passive-aggression",
    "sarcasm-irony",
    "strategic-politeness",
]

PLUTCHIK_EMOTIONS = [
    "joy", "trust", "fear", "surprise",
    "sadness", "disgust", "anger", "anticipation",
]

VAD_DIMENSIONS = ["v", "a", "d", "confidence"]

VAD_LABEL_MAP: dict[str, dict[str, float]] = {
    "v": {
        "very unpleasant": -1.0, "unpleasant": -0.67, "mildly unpleasant": -0.33,
        "neutral": 0.0, "mildly pleasant": 0.33, "pleasant": 0.67, "very pleasant": 1.0,
    },
    "a": {
        "very calm": -1.0, "calm": -0.67, "slightly calm": -0.33,
        "neutral": 0.0, "slightly excited": 0.33, "excited": 0.67, "very excited": 1.0,
    },
    "d": {
        "very controlled": -1.0, "controlled": -0.67, "slightly controlled": -0.33,
        "neutral": 0.0, "slightly in control": 0.33, "in control": 0.67, "very in control": 1.0,
    },
    "confidence": {
        "very unsure": -1.0, "unsure": -0.67, "mildly unsure": -0.33,
        "neutral": 0.0, "mildly confident": 0.33, "confident": 0.67, "very confident": 1.0,
    },
}

POWER_KEYWORDS_HIGH_TO_LOW = [
    "manager", "boss", "supervisor", "parent", "mother", "father", "teacher",
    "professor", "older sibling", "senior", "mentor", "coach", "adult",
    "elder", "doctor", "principal",
]

POWER_KEYWORDS_LOW_TO_HIGH = [
    "employee", "subordinate", "student", "child", "younger sibling",
    "junior", "intern", "mentee", "son", "daughter", "patient",
]

# DMLR-specific prompt: asks about the SPEAKER's emotion (distinct from CogSci's
# LISTENER framing).  This is the ground-truth annotation target.
DMLR_BASELINE_PROMPT = """You are evaluating a communication scenario.  Based on the \
context, determine the primary emotion the SPEAKER is most likely experiencing when \
they make this utterance.

SCENARIO
Situation: {situation}
Speaker ({speaker_role}) says to Listener ({listener_role}):
"{utterance}"

Choose exactly one of: joy, trust, fear, surprise, sadness, disgust, anger, anticipation

Respond with ONLY a JSON object — no extra text:
{{"emotion": "<one of the 8 emotions>"}}"""

# Chain-of-thought prompt: asks model to reason step-by-step before answering.
DMLR_COT_PROMPT = """You are analyzing a social interaction to determine the speaker's \
emotional state.

SCENARIO
Situation: {situation}
Speaker ({speaker_role}) says to Listener ({listener_role}):
"{utterance}"

Think through this step by step:
1. Literal meaning: What does the utterance literally say?
2. Contextual cues: What does the situation suggest about what is really going on?
3. Pragmatic interpretation: Is there a gap between what is said and what is meant?
4. Speaker's internal state: What is the speaker likely feeling beneath the surface?
5. Primary emotion: Which single emotion best captures the speaker's state?

Choose ONE emotion from: joy, trust, fear, surprise, sadness, disgust, anger, anticipation

First, provide your step-by-step reasoning.
Then on the final line, respond with ONLY a JSON object:
{{"emotion": "<one of the 8 emotions>"}}"""

# Few-shot prompt: provides 3 worked examples before the target scenario.
DMLR_FEWSHOT_PROMPT = """You are evaluating communication scenarios to determine the \
primary emotion the SPEAKER is most likely experiencing.

Here are three examples:

EXAMPLE 1 (Sarcasm/Irony):
Situation: After a colleague takes credit for a project the speaker largely completed, \
the speaker responds in a team meeting.
Speaker (team member) says to Listener (colleague): "Oh, congratulations on YOUR \
amazing work. I'm sure all those late nights I spent were just for fun."
Answer: {{"emotion": "anger"}}

EXAMPLE 2 (Deflection/Misdirection):
Situation: A friend asks the speaker how they're coping after a recent breakup.
Speaker (friend) says to Listener (friend): "Have you tried the new coffee place on \
Main Street? Their lattes are incredible."
Answer: {{"emotion": "sadness"}}

EXAMPLE 3 (Strategic Politeness):
Situation: A junior employee is asked to work overtime for the third weekend in a row \
by their manager.
Speaker (employee) says to Listener (manager): "Of course, I'm happy to help. \
Whatever the team needs."
Answer: {{"emotion": "anger"}}

Now evaluate this scenario:

SCENARIO
Situation: {situation}
Speaker ({speaker_role}) says to Listener ({listener_role}):
"{utterance}"

Choose exactly one of: joy, trust, fear, surprise, sadness, disgust, anger, anticipation

Respond with ONLY a JSON object — no extra text:
{{"emotion": "<one of the 8 emotions>"}}"""

# Map prompt mode names to templates
PROMPT_TEMPLATES = {
    "zero-shot": DMLR_BASELINE_PROMPT,
    "cot": DMLR_COT_PROMPT,
    "few-shot": DMLR_FEWSHOT_PROMPT,
}

# Provider → API key environment variable
PROVIDER_API_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
}

# OpenAI-compatible chat/completions endpoints (all except anthropic and google)
OPENAI_COMPAT_ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
    "together": "https://api.together.xyz/v1/chat/completions",
    "fireworks": "https://api.fireworks.ai/inference/v1/chat/completions",
}


def _short_model_name(model_id: str) -> str:
    """Derive a CLI-friendly short name from a full model ID.

    Examples:
        "gpt-5-mini"                                    → "gpt-5-mini"
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" → "llama-3.1-70b"
        "accounts/fireworks/models/deepseek-v3p1"   → "deepseek-v3p1"
        "microsoft/phi-4"                           → "phi-4"
    """
    name = model_id.split("/")[-1].lower()
    # Simplify common Together/Fireworks naming patterns
    name = name.replace("meta-llama-", "llama-")
    name = name.replace("-instruct-turbo", "")
    name = name.replace("-instruct", "")
    return name


def load_dmlr_config(config_path: Path) -> dict[str, Any]:
    """Load config-dmlr.yml and return parsed config."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model_registry(
    config: dict[str, Any],
    mode: str = "complete",
) -> dict[str, dict[str, str]]:
    """Build model registry from config, keyed by short name.

    Returns dict like:
        {"gpt-5-mini": {"provider": "openai", "model_id": "gpt-5-mini",
                     "api_key_env": "OPENAI_API_KEY"}, ...}
    """
    models = config.get("llm_inference", {}).get("models", {}).get(mode, [])
    registry: dict[str, dict[str, str]] = {}
    for entry in models:
        model_id = entry["id"]
        provider = entry["provider"]
        short = _short_model_name(model_id)
        registry[short] = {
            "provider": provider,
            "model_id": model_id,
            "api_key_env": PROVIDER_API_KEYS.get(provider, ""),
        }
    return registry


def get_pricing(config: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """Extract pricing section from config, keyed by provider → model → {input, output}."""
    return config.get("pricing_usd_per_1m_tokens", {})


# ---------------------------------------------------------------------------
# Utility classes (patterned after run_pipeline_facct2026.py)
# ---------------------------------------------------------------------------


@dataclass
class APICallTracker:
    """Track API calls per model and total, with safety limits."""

    model_counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    model_max: int = 500
    total_max: int = 2000

    def increment(self, model_id: str) -> bool:
        self.model_counts[model_id] = self.model_counts.get(model_id, 0) + 1
        self.total_count += 1
        return (
            self.model_counts[model_id] <= self.model_max
            and self.total_count <= self.total_max
        )

    def check_limits(self, model_id: str) -> tuple[bool, str]:
        if self.model_counts.get(model_id, 0) >= self.model_max:
            return False, f"Model {model_id} reached limit ({self.model_max})"
        if self.total_count >= self.total_max:
            return False, f"Total calls reached limit ({self.total_max})"
        return True, ""

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total_count,
            "limit": self.total_max,
            "per_model": dict(self.model_counts),
        }


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    jitter_factor: float = 0.25

    def delay(self, attempt: int) -> float:
        d = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = d * self.jitter_factor * (2 * random.random() - 1)
        return max(0.1, d + jitter)


class DualLogger:
    """Log to both terminal and file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("dmlr2026_pipeline")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(ch)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_merged_data(data_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Load merged CSVs from data/human-gold/ keyed by subtype."""
    data: dict[str, list[dict[str, str]]] = {}
    for subtype in SUBTYPES:
        csv_path = data_dir / f"data_{subtype}.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        data[subtype] = rows
    return data


def detect_annotator_names(rows: list[dict[str, str]]) -> list[str]:
    """Detect annotator names from column headers like sl_plutchik_primary_<Name>."""
    if not rows:
        return []
    names: list[str] = []
    for col in rows[0]:
        if col.startswith("sl_plutchik_primary_") and col != "sl_plutchik_primary":
            names.append(col.replace("sl_plutchik_primary_", ""))
    return sorted(names)


def classify_power(speaker_role: str, listener_role: str) -> str:
    """Classify power relation as peer / high→low / low→high."""
    sp = speaker_role.lower()
    lr = listener_role.lower()
    sp_high = any(kw in sp for kw in POWER_KEYWORDS_HIGH_TO_LOW)
    sp_low = any(kw in sp for kw in POWER_KEYWORDS_LOW_TO_HIGH)
    lr_high = any(kw in lr for kw in POWER_KEYWORDS_HIGH_TO_LOW)
    lr_low = any(kw in lr for kw in POWER_KEYWORDS_LOW_TO_HIGH)
    if sp_high and lr_low:
        return "high_to_low"
    if sp_low and lr_high:
        return "low_to_high"
    return "peer"


def vad_to_numeric(label: str, dimension: str) -> Optional[float]:
    """Convert a VAD text label to numeric value."""
    mapping = VAD_LABEL_MAP.get(dimension, {})
    val = mapping.get(label.strip().lower()) if isinstance(label, str) else None
    return val


# ---------------------------------------------------------------------------
# R0: Fleiss' κ agreement
# ---------------------------------------------------------------------------


def compute_fleiss_kappa(
    annotation_matrix: list[list[int]],
) -> float:
    """Compute Fleiss' κ for a matrix of (N scenarios × K categories).

    Each cell = number of annotators who chose that category for that scenario.
    """
    n_subjects = len(annotation_matrix)
    if n_subjects == 0:
        return 0.0
    n_categories = len(annotation_matrix[0])
    n_raters = sum(annotation_matrix[0])  # same for all rows

    if n_raters <= 1:
        return 0.0

    # p_j: proportion of all assignments to category j
    total_assignments = n_subjects * n_raters
    p_j = [
        sum(row[j] for row in annotation_matrix) / total_assignments
        for j in range(n_categories)
    ]

    # P_i: extent of agreement for subject i
    P_i = []
    for row in annotation_matrix:
        s = sum(nij * (nij - 1) for nij in row)
        P_i.append(s / (n_raters * (n_raters - 1)))

    P_bar = sum(P_i) / n_subjects
    Pe = sum(pj ** 2 for pj in p_j)

    if abs(1 - Pe) < 1e-10:
        return 1.0 if abs(P_bar - 1.0) < 1e-10 else 0.0

    kappa = (P_bar - Pe) / (1 - Pe)
    return kappa


def bootstrap_kappa_ci(
    annotation_matrix: list[list[int]],
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Fleiss' κ."""
    n = len(annotation_matrix)
    if n <= 1:
        return (0.0, 0.0)
    kappas: list[float] = []
    for _ in range(n_boot):
        indices = [random.randint(0, n - 1) for _ in range(n)]
        sample = [annotation_matrix[i] for i in indices]
        kappas.append(compute_fleiss_kappa(sample))
    kappas.sort()
    lo = kappas[int(n_boot * alpha / 2)]
    hi = kappas[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)


def build_annotation_matrix(
    rows: list[dict[str, str]],
    annotator_names: list[str],
) -> list[list[int]]:
    """Build Fleiss matrix from merged CSV rows.

    Returns list of lists: each inner list has len(PLUTCHIK_EMOTIONS) counts.
    """
    cat_index = {e: i for i, e in enumerate(PLUTCHIK_EMOTIONS)}
    matrix: list[list[int]] = []
    for row in rows:
        counts = [0] * len(PLUTCHIK_EMOTIONS)
        for name in annotator_names:
            col = f"sl_plutchik_primary_{name}"
            label = row.get(col, "").strip().lower()
            if label in cat_index:
                counts[cat_index[label]] += 1
        if sum(counts) > 0:
            matrix.append(counts)
    return matrix


def stage_verify_agreement(
    data: dict[str, list[dict[str, str]]],
    logger: DualLogger,
) -> dict[str, Any]:
    """R0: Recompute Fleiss' κ per subtype + weighted overall with CIs."""
    logger.info("[R0] Verifying inter-annotator agreement (Fleiss' κ)")
    results: dict[str, Any] = {"per_subtype": {}, "overall": {}}
    all_matrices: list[list[int]] = []

    for subtype in SUBTYPES:
        rows = data.get(subtype, [])
        if not rows:
            logger.warning(f"  {subtype}: no data")
            continue
        names = detect_annotator_names(rows)
        if len(names) < 2:
            logger.warning(f"  {subtype}: fewer than 2 annotators detected")
            continue

        matrix = build_annotation_matrix(rows, names)
        kappa = compute_fleiss_kappa(matrix)
        ci_lo, ci_hi = bootstrap_kappa_ci(matrix)
        n = len(matrix)

        results["per_subtype"][subtype] = {
            "kappa": round(kappa, 4),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "n_scenarios": n,
            "annotators": names,
        }
        all_matrices.extend(matrix)
        logger.info(
            f"  {subtype}: κ = {kappa:.4f}  "
            f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  (n={n}, raters={names})"
        )

    # Overall (pooled — same as one big matrix since triads are disjoint)
    if all_matrices:
        overall_k = compute_fleiss_kappa(all_matrices)
        overall_ci = bootstrap_kappa_ci(all_matrices)
        results["overall"] = {
            "kappa": round(overall_k, 4),
            "ci_95": [round(overall_ci[0], 4), round(overall_ci[1], 4)],
            "n_scenarios": len(all_matrices),
            "method": "pooled (annotators assigned by subtype)",
        }
        logger.info(
            f"  OVERALL: κ = {overall_k:.4f}  "
            f"95% CI [{overall_ci[0]:.4f}, {overall_ci[1]:.4f}]  (n={len(all_matrices)})"
        )

    return results


# ---------------------------------------------------------------------------
# R0: Power distribution verification
# ---------------------------------------------------------------------------


def stage_verify_power(
    data: dict[str, list[dict[str, str]]],
    logger: DualLogger,
) -> dict[str, Any]:
    """R0: Verify power relation distribution (peer / high→low / low→high)."""
    logger.info("[R0] Verifying power distribution")
    overall_counts: Counter[str] = Counter()
    per_subtype: dict[str, dict[str, int]] = {}

    for subtype in SUBTYPES:
        rows = data.get(subtype, [])
        counts: Counter[str] = Counter()
        for row in rows:
            sp = row.get("sd_speaker_role", "")
            lr = row.get("sd_listener_role", "")
            pwr = classify_power(sp, lr)
            counts[pwr] += 1
        per_subtype[subtype] = dict(counts)
        overall_counts.update(counts)
        logger.info(f"  {subtype}: {dict(counts)}")

    logger.info(f"  TOTAL: {dict(overall_counts)}  (sum={sum(overall_counts.values())})")
    return {"per_subtype": per_subtype, "overall": dict(overall_counts)}


# ---------------------------------------------------------------------------
# R0: Human performance (agreement patterns)
# ---------------------------------------------------------------------------


def stage_human_performance(
    data: dict[str, list[dict[str, str]]],
    logger: DualLogger,
) -> dict[str, Any]:
    """R0: Compute human agreement patterns (unanimous / majority / split)."""
    logger.info("[R0] Computing human performance / agreement patterns")
    results: dict[str, Any] = {"per_subtype": {}, "overall": {}}
    total_unanimous = 0
    total_majority = 0
    total_split = 0
    total_n = 0

    for subtype in SUBTYPES:
        rows = data.get(subtype, [])
        if not rows:
            continue
        names = detect_annotator_names(rows)
        unanimous = majority = split_count = 0
        for row in rows:
            labels = []
            for nm in names:
                col = f"sl_plutchik_primary_{nm}"
                lbl = row.get(col, "").strip().lower()
                if lbl:
                    labels.append(lbl)
            if len(labels) < 2:
                continue
            c = Counter(labels)
            most_common_count = c.most_common(1)[0][1]
            if most_common_count == len(labels):
                unanimous += 1
            elif most_common_count >= 2:
                majority += 1
            else:
                split_count += 1

        n = unanimous + majority + split_count
        per = {
            "unanimous": unanimous,
            "majority": majority,
            "split": split_count,
            "total": n,
            "pct_unanimous": round(unanimous / n * 100, 1) if n else 0,
            "pct_majority": round(majority / n * 100, 1) if n else 0,
            "pct_split": round(split_count / n * 100, 1) if n else 0,
        }
        results["per_subtype"][subtype] = per
        total_unanimous += unanimous
        total_majority += majority
        total_split += split_count
        total_n += n
        logger.info(
            f"  {subtype}: unan={unanimous} maj={majority} split={split_count} "
            f"({per['pct_unanimous']}% / {per['pct_majority']}% / {per['pct_split']}%)"
        )

    if total_n:
        results["overall"] = {
            "unanimous": total_unanimous,
            "majority": total_majority,
            "split": total_split,
            "total": total_n,
            "pct_unanimous": round(total_unanimous / total_n * 100, 1),
            "pct_majority": round(total_majority / total_n * 100, 1),
            "pct_split": round(total_split / total_n * 100, 1),
        }
        logger.info(
            f"  OVERALL: unan={total_unanimous} maj={total_majority} split={total_split} "
            f"({results['overall']['pct_unanimous']}% / "
            f"{results['overall']['pct_majority']}% / "
            f"{results['overall']['pct_split']}%)"
        )
    return results


# ---------------------------------------------------------------------------
# R4: VAD Analysis — ICC and distributions
# ---------------------------------------------------------------------------


def _icc_2_1(ratings: list[list[Optional[float]]]) -> Optional[float]:
    """Compute ICC(2,1) for a matrix of (n_subjects × k_raters).

    Two-way random, single measures.  Returns None if computation fails.
    """
    # Filter out rows with missing values
    valid = [row for row in ratings if all(v is not None for v in row)]
    if len(valid) < 3:
        return None
    n = len(valid)
    k = len(valid[0])
    if k < 2:
        return None

    grand_mean = sum(v for row in valid for v in row) / (n * k)  # type: ignore[operator]
    row_means = [sum(row) / k for row in valid]  # type: ignore[arg-type]
    col_means = [sum(valid[i][j] for i in range(n)) / n for j in range(k)]  # type: ignore[operator]

    # Sum of squares
    ss_total = sum(
        (valid[i][j] - grand_mean) ** 2  # type: ignore[operator]
        for i in range(n) for j in range(k)
    )
    ss_rows = k * sum((rm - grand_mean) ** 2 for rm in row_means)
    ss_cols = n * sum((cm - grand_mean) ** 2 for cm in col_means)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1) if n > 1 else 0
    ms_cols = ss_cols / (k - 1) if k > 1 else 0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0

    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if abs(denom) < 1e-10:
        return None
    icc = (ms_rows - ms_error) / denom
    return icc


def stage_vad_analysis(
    data: dict[str, list[dict[str, str]]],
    logger: DualLogger,
) -> dict[str, Any]:
    """R4: VAD ICC computation, distribution stats, emotion-VAD consistency."""
    logger.info("[R4] VAD Analysis (ICC + distributions)")
    results: dict[str, Any] = {
        "icc_per_subtype": {},
        "icc_overall": {},
        "distributions": {},
        "consistency": {},
    }

    # Collect all VAD ratings per dimension for ICC
    for dim in VAD_DIMENSIONS:
        all_ratings: list[list[Optional[float]]] = []

        for subtype in SUBTYPES:
            rows = data.get(subtype, [])
            if not rows:
                continue
            names = detect_annotator_names(rows)
            subtype_ratings: list[list[Optional[float]]] = []

            for row in rows:
                vals: list[Optional[float]] = []
                for nm in names:
                    col = f"sl_{dim}_{nm}"
                    raw = row.get(col, "")
                    vals.append(vad_to_numeric(raw, dim))
                subtype_ratings.append(vals)

            # Subtype-level ICC
            icc_val = _icc_2_1(subtype_ratings)
            if subtype not in results["icc_per_subtype"]:
                results["icc_per_subtype"][subtype] = {}
            results["icc_per_subtype"][subtype][dim] = (
                round(icc_val, 4) if icc_val is not None else None
            )
            all_ratings.extend(subtype_ratings)

        # Overall ICC for this dimension
        icc_overall = _icc_2_1(all_ratings)
        results["icc_overall"][dim] = (
            round(icc_overall, 4) if icc_overall is not None else None
        )
        logger.info(
            f"  ICC(2,1) {dim}: overall={results['icc_overall'][dim]}  "
            + "  ".join(
                f"{st}={results['icc_per_subtype'].get(st, {}).get(dim)}"
                for st in SUBTYPES
            )
        )

    # Distribution statistics (mean, sd per subtype per dimension)
    for subtype in SUBTYPES:
        rows = data.get(subtype, [])
        if not rows:
            continue
        names = detect_annotator_names(rows)
        dist: dict[str, Any] = {}
        for dim in VAD_DIMENSIONS:
            vals: list[float] = []
            for row in rows:
                for nm in names:
                    col = f"sl_{dim}_{nm}"
                    v = vad_to_numeric(row.get(col, ""), dim)
                    if v is not None:
                        vals.append(v)
            if vals:
                mean = sum(vals) / len(vals)
                sd = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
                dist[dim] = {"mean": round(mean, 3), "sd": round(sd, 3), "n": len(vals)}
        results["distributions"][subtype] = dist

    # Emotion-VAD consistency: mean valence per gold-standard emotion
    emotion_valence: dict[str, list[float]] = {e: [] for e in PLUTCHIK_EMOTIONS}
    for subtype in SUBTYPES:
        rows = data.get(subtype, [])
        if not rows:
            continue
        names = detect_annotator_names(rows)
        for row in rows:
            gs = row.get("gold_standard", "").strip().lower()
            if gs not in emotion_valence:
                continue
            for nm in names:
                v = vad_to_numeric(row.get(f"sl_v_{nm}", ""), "v")
                if v is not None:
                    emotion_valence[gs].append(v)

    for emo in PLUTCHIK_EMOTIONS:
        vals = emotion_valence[emo]
        if vals:
            mean_v = sum(vals) / len(vals)
            results["consistency"][emo] = {
                "mean_valence": round(mean_v, 3),
                "n": len(vals),
            }
            logger.info(f"  Emotion-VAD: {emo} → mean_v={mean_v:.3f} (n={len(vals)})")

    return results


# ---------------------------------------------------------------------------
# R3: Scale justification (power analysis)
# ---------------------------------------------------------------------------


def stage_scale_justification(
    data: dict[str, list[dict[str, str]]],
    agreement_results: dict[str, Any],
    logger: DualLogger,
) -> dict[str, Any]:
    """R3: Power analysis, CI widths, benchmark comparison."""
    logger.info("[R3] Scale justification / power analysis")
    n_total = sum(len(rows) for rows in data.values())

    # CI width for κ (already computed)
    overall_ci = agreement_results.get("overall", {}).get("ci_95", [0, 0])
    ci_width = round(overall_ci[1] - overall_ci[0], 4) if len(overall_ci) == 2 else None

    # Benchmark comparisons (published dataset sizes)
    benchmarks = [
        {"name": "iSarcasm", "n": 4484, "type": "tweets"},
        {"name": "SARC", "n": 533000, "type": "Reddit comments"},
        {"name": "SocialIQa", "n": 38000, "type": "multiple-choice QA"},
        {"name": "FANToM", "n": 10000, "type": "dialogue turns"},
        {"name": "MELD", "n": 13708, "type": "utterances"},
        {"name": "IEMOCAP", "n": 10039, "type": "utterances"},
        {"name": "CEI-ToM (ours)", "n": n_total, "type": "expert-authored scenarios"},
    ]

    # Effect size detectable with N=300 at α=.05, power=.80 (Cohen's w)
    # For 8-category chi-squared test: df = 7
    # w = sqrt(chi2_critical / N)
    # chi2(.05, 7) ≈ 14.067
    chi2_crit = 14.067
    detectable_w = round(math.sqrt(chi2_crit / n_total), 3) if n_total > 0 else None

    results = {
        "n_scenarios": n_total,
        "n_annotations": n_total * 3,
        "kappa_ci_width": ci_width,
        "detectable_effect_size_w": detectable_w,
        "benchmarks": benchmarks,
        "notes": [
            f"N={n_total} with 3 annotators/scenario yields {n_total * 3} annotations",
            f"κ 95% CI width = {ci_width} (narrower = more precise estimate)",
            f"Detectable Cohen's w = {detectable_w} (small ≈ 0.1, medium ≈ 0.3)",
            "Expert-authored scenarios enable higher signal density than mined corpora",
        ],
    }
    logger.info(f"  N={n_total} scenarios, {n_total * 3} annotations")
    logger.info(f"  κ CI width: {ci_width}")
    logger.info(f"  Detectable effect size (w): {detectable_w}")
    return results


# ---------------------------------------------------------------------------
# R5: Stratified splits
# ---------------------------------------------------------------------------


def stage_create_splits(
    data: dict[str, list[dict[str, str]]],
    logger: DualLogger,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, Any]:
    """R5: Create stratified train/val/test splits balanced by subtype and power."""
    logger.info("[R5] Creating stratified splits")
    rng = random.Random(seed)
    test_frac = 1.0 - train_frac - val_frac
    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    split_assignments: list[dict[str, Any]] = []

    for subtype in SUBTYPES:
        rows = data.get(subtype, [])
        if not rows:
            continue
        # Group by power relation for stratification
        by_power: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            pwr = classify_power(
                row.get("sd_speaker_role", ""),
                row.get("sd_listener_role", ""),
            )
            by_power.setdefault(pwr, []).append(row)

        for pwr, pwr_rows in by_power.items():
            rng.shuffle(pwr_rows)
            n = len(pwr_rows)
            n_train = max(1, round(n * train_frac))
            n_val = max(1, round(n * val_frac))
            # Ensure we don't exceed total
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)

            for i, row in enumerate(pwr_rows):
                if i < n_train:
                    s = "train"
                elif i < n_train + n_val:
                    s = "val"
                else:
                    s = "test"
                splits[s].append(row)
                split_assignments.append({
                    "id": row.get("id", ""),
                    "subtype": subtype,
                    "power": pwr,
                    "split": s,
                })

    # Verify proportions
    proportions: dict[str, dict[str, int]] = {}
    for split_name, rows_list in splits.items():
        proportions[split_name] = {
            "total": len(rows_list),
        }
        st_counts: Counter[str] = Counter()
        for asgn in split_assignments:
            if asgn["split"] == split_name:
                st_counts[asgn["subtype"]] += 1
        proportions[split_name]["by_subtype"] = dict(st_counts)

    for s in ["train", "val", "test"]:
        logger.info(f"  {s}: {proportions[s]['total']} scenarios  {proportions[s].get('by_subtype', {})}")

    return {
        "seed": seed,
        "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "proportions": proportions,
        "assignments": split_assignments,
    }


# ---------------------------------------------------------------------------
# R7: Extract worked examples
# ---------------------------------------------------------------------------


def stage_extract_examples(
    data: dict[str, list[dict[str, str]]],
    logger: DualLogger,
    n_examples: int = 5,
) -> dict[str, Any]:
    """R7: Select candidate worked examples (prefer mixed signals & deflection)."""
    logger.info("[R7] Extracting candidate worked examples")
    # Prefer subtypes NOT used by CogSci paper (strategic-politeness is CogSci's example)
    preferred = ["deflection-misdirection", "mixed-signals", "passive-aggression"]
    candidates: list[dict[str, Any]] = []

    for subtype in preferred:
        rows = data.get(subtype, [])
        if not rows:
            continue
        names = detect_annotator_names(rows)
        for row in rows:
            labels = [
                row.get(f"sl_plutchik_primary_{nm}", "").strip().lower()
                for nm in names
            ]
            labels = [l for l in labels if l]
            if len(labels) < 2:
                continue
            c = Counter(labels)
            # Interesting examples: majority but not unanimous (shows discussion value)
            most = c.most_common(1)[0][1]
            if most == 2 and len(labels) == 3:
                candidates.append({
                    "subtype": subtype,
                    "id": row.get("id", ""),
                    "situation": row.get("sd_situation", ""),
                    "utterance": row.get("sd_utterance", ""),
                    "speaker_role": row.get("sd_speaker_role", ""),
                    "listener_role": row.get("sd_listener_role", ""),
                    "annotations": labels,
                    "gold_standard": row.get("gold_standard", ""),
                })

    # Select up to n_examples, spread across subtypes
    selected: list[dict[str, Any]] = []
    by_sub: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        by_sub.setdefault(c["subtype"], []).append(c)
    idx = 0
    while len(selected) < n_examples:
        added = False
        for sub_list in by_sub.values():
            if idx < len(sub_list) and len(selected) < n_examples:
                selected.append(sub_list[idx])
                added = True
        if not added:
            break
        idx += 1

    for i, ex in enumerate(selected):
        logger.info(
            f"  Example {i + 1}: [{ex['subtype']}] id={ex['id']}  "
            f"labels={ex['annotations']}  gold={ex['gold_standard']}"
        )
        logger.info(f"    Situation: {ex['situation'][:80]}...")
        logger.info(f"    Utterance: {ex['utterance'][:80]}...")

    return {"candidates": len(candidates), "selected": selected}


# ---------------------------------------------------------------------------
# R1: Baseline inference
# ---------------------------------------------------------------------------

def _call_openai_compat(
    endpoint: str, model_id: str, prompt: str, api_key: str,
    max_tokens: int = 60,
) -> Optional[str]:
    """Call any OpenAI-compatible chat/completions endpoint.

    Works for: OpenAI, xAI (Grok), Together AI, Fireworks AI.
    """
    import urllib.request

    is_openai_new = endpoint == OPENAI_COMPAT_ENDPOINTS.get("openai") and (
        model_id.startswith("gpt-5") or model_id.startswith("o")
    )
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    if is_openai_new:
        # GPT-5+ uses internal reasoning tokens; ensure enough room for output
        payload["max_completion_tokens"] = max(max_tokens, 200)
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.0
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "CEI-Pipeline/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _call_anthropic(model_id: str, prompt: str, api_key: str, max_tokens: int = 60) -> Optional[str]:
    """Call Anthropic Messages API."""
    import urllib.request

    body = json.dumps({
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def _call_google(model_id: str, prompt: str, api_key: str, max_tokens: int = 60) -> Optional[str]:
    """Call Google Gemini generateContent API."""
    import urllib.request

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_model(
    model_name: str,
    prompt: str,
    model_registry: dict[str, dict[str, str]],
    max_tokens: int = 60,
) -> Optional[str]:
    """Dispatch to the appropriate provider based on model registry."""
    cfg = model_registry.get(model_name)
    if not cfg:
        return None
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        return None
    provider = cfg["provider"]
    model_id = cfg["model_id"]

    if provider == "anthropic":
        return _call_anthropic(model_id, prompt, api_key, max_tokens=max_tokens)
    elif provider == "google":
        return _call_google(model_id, prompt, api_key, max_tokens=max_tokens)
    else:
        # OpenAI-compatible: openai, xai, together, fireworks
        endpoint = OPENAI_COMPAT_ENDPOINTS.get(provider)
        if not endpoint:
            return None
        return _call_openai_compat(endpoint, model_id, prompt, api_key, max_tokens=max_tokens)


def _parse_emotion_response(text: str) -> Optional[str]:
    """Extract emotion from model response (expects JSON with 'emotion' key).

    Handles multiple response formats:
    - Direct JSON: {"emotion": "anger"}
    - Markdown-wrapped JSON: ```json\n{"emotion": "anger"}\n```
    - CoT with JSON at end: reasoning...\n{"emotion": "anger"}
    - CoT with "Answer: emotion" format
    """
    if not text:
        return None

    # Try to find JSON anywhere in the text (handles CoT where JSON is at the end)
    import re
    json_matches = re.findall(r'\{[^{}]*"emotion"\s*:\s*"[^"]*"[^{}]*\}', text)
    for match in reversed(json_matches):  # prefer last match (CoT puts it at end)
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and "emotion" in obj:
                emo = obj["emotion"].strip().lower()
                if emo in PLUTCHIK_EMOTIONS:
                    return emo
        except (json.JSONDecodeError, AttributeError):
            pass

    # Try full text as JSON (with markdown cleanup)
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "emotion" in obj:
            emo = obj["emotion"].strip().lower()
            if emo in PLUTCHIK_EMOTIONS:
                return emo
    except (json.JSONDecodeError, AttributeError):
        pass

    # Try "Answer: emotion" format (CoT)
    answer_match = re.search(r'[Aa]nswer:\s*(\w+)', text)
    if answer_match:
        emo = answer_match.group(1).strip().lower()
        if emo in PLUTCHIK_EMOTIONS:
            return emo

    # Fallback: look for emotion keywords in last line or full text
    lines = text.strip().split("\n")
    for search_text in [lines[-1], text]:
        text_lower = search_text.lower()
        for emo in PLUTCHIK_EMOTIONS:
            if emo in text_lower:
                return emo
    return None


def _run_single_scenario(
    model_name: str,
    sc: dict[str, str],
    model_registry: dict[str, dict[str, str]],
    retry_config: RetryConfig,
    prompt_template: str = DMLR_BASELINE_PROMPT,
    prompt_mode: str = "zero-shot",
) -> tuple[str, dict[str, Any], str, bool, Optional[str]]:
    """Run a single scenario for a single model. Returns (key, result, raw, success, error)."""
    key = f"{model_name}::{sc['subtype']}::{sc['id']}"
    prompt = prompt_template.format(
        situation=sc["situation"],
        utterance=sc["utterance"],
        speaker_role=sc["speaker_role"],
        listener_role=sc["listener_role"],
    )
    max_tokens = 2048 if prompt_mode == "cot" else 256
    raw_response = None
    error_msg = None
    for attempt in range(retry_config.max_retries):
        try:
            raw_response = _call_model(model_name, prompt, model_registry, max_tokens=max_tokens)
            break
        except Exception as e:
            error_msg = str(e)
            delay = retry_config.delay(attempt)
            time.sleep(delay)

    emotion = _parse_emotion_response(raw_response or "")
    result = {
        "id": sc["id"],
        "subtype": sc["subtype"],
        "gold": sc["gold_standard"],
        "predicted": emotion,
        "raw_response": raw_response,
        "cached": False,
    }
    success = raw_response is not None
    return key, result, raw_response or "", success, error_msg


def stage_run_baselines(
    data: dict[str, list[dict[str, str]]],
    logger: DualLogger,
    call_tracker: APICallTracker,
    retry_config: RetryConfig,
    output_dir: Path,
    model_registry: dict[str, dict[str, str]],
    pricing: dict[str, Any],
    models: Optional[list[str]] = None,
    dry_run: bool = False,
    resume: bool = False,
    workers_per_model: int = 10,
    prompt_mode: str = "zero-shot",
) -> dict[str, Any]:
    """R1: Run baseline models on all scenarios (parallelized across and within models)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    prompt_template = PROMPT_TEMPLATES.get(prompt_mode, DMLR_BASELINE_PROMPT)
    logger.info(f"[R1] Baseline inference (parallelized, mode={prompt_mode})")
    if models is None:
        models = list(model_registry.keys())

    if dry_run:
        n_scenarios = sum(len(rows) for rows in data.values())
        total_calls = n_scenarios * len(models)
        logger.info(f"  DRY RUN: would make {total_calls} API calls")
        logger.info(f"  Models: {models}")
        est_input_tokens = n_scenarios * 800
        est_output_tokens = n_scenarios * 200
        total_cost = 0.0
        for m in models:
            cfg = model_registry.get(m, {})
            provider = cfg.get("provider", "")
            model_id = cfg.get("model_id", "")
            provider_pricing = pricing.get(provider, {})
            model_pricing = (
                provider_pricing.get(model_id)
                or provider_pricing.get(model_id.split("/")[-1])
                or provider_pricing.get("_default", {})
            )
            inp = model_pricing.get("input", 0)
            out = model_pricing.get("output", 0)
            cost = (est_input_tokens * inp + est_output_tokens * out) / 1_000_000
            total_cost += cost
            has_key = "OK" if os.environ.get(cfg.get("api_key_env", ""), "") else "NO KEY"
            logger.info(f"    {m}: ~${cost:.2f} ({inp}/{out} per 1M tokens) [{has_key}]")
        logger.info(f"  Estimated total cost: ${total_cost:.2f}")
        return {
            "status": "dry_run", "models": models,
            "scenarios": n_scenarios, "est_cost_usd": round(total_cost, 2),
        }

    # Check which models have API keys
    available: list[str] = []
    for m in models:
        cfg = model_registry.get(m)
        if cfg and os.environ.get(cfg["api_key_env"], ""):
            available.append(m)
        else:
            key_env = cfg["api_key_env"] if cfg else "?"
            logger.warning(f"  {m}: API key not set ({key_env}), skipping")

    if not available:
        logger.warning("  No models available (no API keys set). Skipping inference.")
        return {"status": "skipped", "reason": "no_api_keys"}

    # Build scenario list
    scenarios: list[dict[str, str]] = []
    for subtype in SUBTYPES:
        for row in data.get(subtype, []):
            scenarios.append({
                "id": row.get("id", ""),
                "subtype": subtype,
                "situation": row.get("sd_situation", ""),
                "utterance": row.get("sd_utterance", ""),
                "speaker_role": row.get("sd_speaker_role", ""),
                "listener_role": row.get("sd_listener_role", ""),
                "gold_standard": row.get("gold_standard", ""),
            })

    n_scenarios = len(scenarios)
    logger.info(f"  {len(available)} models × {n_scenarios} scenarios = "
                f"{len(available) * n_scenarios} total calls")
    logger.info(f"  Parallelism: {len(available)} models concurrent, "
                f"{workers_per_model} workers/model")

    # Load checkpoint if resuming (new key format: model::subtype::id)
    mode_suffix = f"_{prompt_mode}" if prompt_mode != "zero-shot" else ""
    checkpoint_path = output_dir / f"baseline_checkpoint{mode_suffix}.json"
    completed: dict[str, str] = {}
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        logger.info(f"  Resumed from checkpoint: {len(completed)} results loaded")

    # Thread-safe progress tracking
    lock = threading.Lock()
    stats: dict[str, dict[str, int]] = {
        m: {"success": 0, "failure": 0, "cached": 0, "total": n_scenarios}
        for m in available
    }
    all_results: dict[str, list[dict[str, Any]]] = {m: [] for m in available}
    start_time = time.monotonic()

    def _progress_report() -> str:
        elapsed = time.monotonic() - start_time
        elapsed_m = elapsed / 60
        lines = [f"\n  === Progress Report (elapsed {elapsed_m:.1f} min) ==="]
        total_done = 0
        total_all = 0
        for m in available:
            s = stats[m]
            done = s["success"] + s["failure"] + s["cached"]
            total_done += done
            total_all += s["total"]
            pct_done = done / s["total"] * 100 if s["total"] else 0
            pct_ok = s["success"] / max(done - s["cached"], 1) * 100 if (done - s["cached"]) > 0 else 100.0
            lines.append(
                f"    {m:30s}  "
                f"API {s['success']:>3d}ok/{s['failure']:>3d}fail  "
                f"{pct_ok:5.1f}%ok  "
                f"{done:>3d}/{s['total']} ({pct_done:5.1f}% complete)"
            )
        overall_pct = total_done / total_all * 100 if total_all else 0
        if total_done > 0 and overall_pct < 100:
            est_total = elapsed / (total_done / total_all)
            est_remain = (est_total - elapsed) / 60
            lines.append(f"  ALL MODELS: {overall_pct:.1f}% complete | "
                         f"elapsed {elapsed_m:.1f} min | "
                         f"est. remaining {est_remain:.1f} min")
        else:
            lines.append(f"  ALL MODELS: {overall_pct:.1f}% complete | "
                         f"elapsed {elapsed_m:.1f} min")
        lines.append("  " + "=" * 50)
        return "\n".join(lines)

    # Background progress reporter (every 60s)
    stop_reporter = threading.Event()

    def _reporter_loop() -> None:
        while not stop_reporter.wait(60):
            with lock:
                report = _progress_report()
            logger.info(report)

    reporter_thread = threading.Thread(target=_reporter_loop, daemon=True)
    reporter_thread.start()

    def _checkpoint_save() -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with lock:
            snapshot = dict(completed)
        with open(checkpoint_path, "w") as f:
            json.dump(snapshot, f)

    def _run_model(model_name: str) -> None:
        """Run all scenarios for one model using a thread pool."""
        # Identify which scenarios still need to be run
        todo: list[dict[str, str]] = []
        for sc in scenarios:
            key = f"{model_name}::{sc['subtype']}::{sc['id']}"
            if key in completed:
                # Restore cached result
                raw = completed[key]
                emotion = _parse_emotion_response(raw)
                with lock:
                    all_results[model_name].append({
                        "id": sc["id"], "subtype": sc["subtype"],
                        "gold": sc["gold_standard"], "predicted": emotion,
                        "raw_response": raw, "cached": True,
                    })
                    stats[model_name]["cached"] += 1
            else:
                todo.append(sc)

        if not todo:
            logger.info(f"  {model_name}: all {n_scenarios} cached, skipping")
            return

        logger.info(f"  {model_name}: {len(todo)} to run, "
                     f"{n_scenarios - len(todo)} cached")

        with ThreadPoolExecutor(max_workers=workers_per_model) as pool:
            futures = {
                pool.submit(
                    _run_single_scenario, model_name, sc,
                    model_registry, retry_config,
                    prompt_template, prompt_mode,
                ): sc
                for sc in todo
            }
            done_count = 0
            for future in as_completed(futures):
                key, result, raw, success, error_msg = future.result()
                with lock:
                    all_results[model_name].append(result)
                    completed[key] = raw
                    if success:
                        stats[model_name]["success"] += 1
                    else:
                        stats[model_name]["failure"] += 1
                done_count += 1
                # Checkpoint every 50 completions
                if done_count % 50 == 0:
                    _checkpoint_save()

        # Final checkpoint for this model
        _checkpoint_save()

    # Run all models concurrently
    with ThreadPoolExecutor(max_workers=len(available)) as model_pool:
        model_futures = {
            model_pool.submit(_run_model, m): m for m in available
        }
        for future in as_completed(model_futures):
            m = model_futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"  {m}: FAILED — {e}")

    # Stop reporter and print final report
    stop_reporter.set()
    reporter_thread.join(timeout=2)
    logger.info(_progress_report())

    # Save full results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"baseline_results{mode_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"  Results saved to {results_path}")

    return {
        "status": "complete",
        "models": available,
        "api_calls": call_tracker.summary(),
        "results_path": str(results_path),
    }


# ---------------------------------------------------------------------------
# R1: Baseline analysis (accuracy + macro-F1)
# ---------------------------------------------------------------------------


def stage_analyze_baselines(
    output_dir: Path,
    logger: DualLogger,
    prompt_mode: str = "zero-shot",
) -> dict[str, Any]:
    """R1: Compute accuracy and macro-F1 from baseline results."""
    mode_suffix = f"_{prompt_mode}" if prompt_mode != "zero-shot" else ""
    logger.info(f"[R1] Analyzing baseline results (mode={prompt_mode})")
    results_path = output_dir / f"baseline_results{mode_suffix}.json"
    if not results_path.exists():
        logger.warning(f"  Results file not found: {results_path}")
        return {"status": "no_data"}

    with open(results_path) as f:
        results = json.load(f)

    analysis: dict[str, Any] = {}
    for model_name, preds in results.items():
        n_total = len(preds)
        n_correct = sum(1 for p in preds if p.get("predicted") == p.get("gold", "").strip().lower())
        n_parsed = sum(1 for p in preds if p.get("predicted") is not None)

        # Macro F1
        per_class: dict[str, dict[str, int]] = {
            e: {"tp": 0, "fp": 0, "fn": 0} for e in PLUTCHIK_EMOTIONS
        }
        for p in preds:
            gold = p.get("gold", "").strip().lower()
            pred = p.get("predicted")
            if pred is None:
                if gold in per_class:
                    per_class[gold]["fn"] += 1
                continue
            if pred == gold:
                if pred in per_class:
                    per_class[pred]["tp"] += 1
            else:
                if pred in per_class:
                    per_class[pred]["fp"] += 1
                if gold in per_class:
                    per_class[gold]["fn"] += 1

        f1s: list[float] = []
        for emo in PLUTCHIK_EMOTIONS:
            tp = per_class[emo]["tp"]
            fp = per_class[emo]["fp"]
            fn = per_class[emo]["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1s.append(f1)

        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        analysis[model_name] = {
            "accuracy": round(n_correct / n_total, 4) if n_total else 0.0,
            "macro_f1": round(macro_f1, 4),
            "n_total": n_total,
            "n_correct": n_correct,
            "n_parsed": n_parsed,
            "n_parse_fail": n_total - n_parsed,
        }
        logger.info(
            f"  {model_name}: acc={analysis[model_name]['accuracy']:.4f}  "
            f"F1={analysis[model_name]['macro_f1']:.4f}  "
            f"({n_correct}/{n_total}, {n_total - n_parsed} parse failures)"
        )

    # Save analysis
    analysis_path = output_dir / f"baseline_analysis{mode_suffix}.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"  Analysis saved to {analysis_path}")

    return {"status": "complete", "analysis": analysis}


# ---------------------------------------------------------------------------
# Output generation: LaTeX tables and figures
# ---------------------------------------------------------------------------


def stage_generate_outputs(
    all_results: dict[str, Any],
    output_dir: Path,
    logger: DualLogger,
) -> dict[str, Any]:
    """Generate LaTeX tables and optional figures for the DMLR paper."""
    logger.info("[Output] Generating paper outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # --- Table 2: Inter-Annotator Agreement (κ per subtype) ---
    agreement = all_results.get("agreement", {})
    if agreement.get("per_subtype"):
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Inter-annotator agreement (Fleiss' $\kappa$) by CEI subtype.}",
            r"\label{tab:kappa}",
            r"\begin{tabular}{lccl}",
            r"\toprule",
            r"Subtype & $\kappa$ & 95\% CI & $n$ \\",
            r"\midrule",
        ]
        for st in SUBTYPES:
            info = agreement["per_subtype"].get(st, {})
            k = info.get("kappa", "—")
            ci = info.get("ci_95", [])
            n = info.get("n_scenarios", "—")
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if len(ci) == 2 else "—"
            k_str = f"{k:.3f}" if isinstance(k, float) else str(k)
            st_display = st.replace("-", " ").title()
            lines.append(f"{st_display} & {k_str} & {ci_str} & {n} \\\\")
        ov = agreement.get("overall", {})
        if ov:
            ov_k = f"{ov['kappa']:.3f}" if isinstance(ov.get("kappa"), float) else "—"
            ov_ci = ov.get("ci_95", [])
            ov_ci_str = f"[{ov_ci[0]:.3f}, {ov_ci[1]:.3f}]" if len(ov_ci) == 2 else "—"
            lines.append(r"\midrule")
            lines.append(f"Overall (pooled) & {ov_k} & {ov_ci_str} & {ov.get('n_scenarios', '—')} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        table_path = output_dir / "table_kappa.tex"
        table_path.write_text("\n".join(lines))
        generated.append(str(table_path))
        logger.info(f"  Generated: {table_path}")

    # --- Table: Human Performance (agreement patterns) ---
    human_perf = all_results.get("human_performance", {})
    if human_perf.get("per_subtype"):
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Human agreement patterns by CEI subtype.}",
            r"\label{tab:human-agreement}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Subtype & Unanimous & Majority & Three-way Split \\",
            r"\midrule",
        ]
        for st in SUBTYPES:
            info = human_perf["per_subtype"].get(st, {})
            u = info.get("pct_unanimous", 0)
            m = info.get("pct_majority", 0)
            s = info.get("pct_split", 0)
            st_display = st.replace("-", " ").title()
            lines.append(f"{st_display} & {u:.1f}\\% & {m:.1f}\\% & {s:.1f}\\% \\\\")
        ov = human_perf.get("overall", {})
        if ov:
            lines.append(r"\midrule")
            lines.append(
                f"Overall & {ov.get('pct_unanimous', 0):.1f}\\% & "
                f"{ov.get('pct_majority', 0):.1f}\\% & "
                f"{ov.get('pct_split', 0):.1f}\\% \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        table_path = output_dir / "table_human_agreement.tex"
        table_path.write_text("\n".join(lines))
        generated.append(str(table_path))
        logger.info(f"  Generated: {table_path}")

    # --- Table: Power Distribution ---
    power = all_results.get("power", {})
    if power.get("overall"):
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Power relation distribution across CEI scenarios.}",
            r"\label{tab:power-dist}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Subtype & Peer & High$\to$Low & Low$\to$High & Total \\",
            r"\midrule",
        ]
        for st in SUBTYPES:
            info = power.get("per_subtype", {}).get(st, {})
            p = info.get("peer", 0)
            h2l = info.get("high_to_low", 0)
            l2h = info.get("low_to_high", 0)
            t = p + h2l + l2h
            st_display = st.replace("-", " ").title()
            lines.append(f"{st_display} & {p} & {h2l} & {l2h} & {t} \\\\")
        ov = power["overall"]
        lines.append(r"\midrule")
        lines.append(
            f"Total & {ov.get('peer', 0)} & {ov.get('high_to_low', 0)} & "
            f"{ov.get('low_to_high', 0)} & "
            f"{ov.get('peer', 0) + ov.get('high_to_low', 0) + ov.get('low_to_high', 0)} \\\\"
        )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        table_path = output_dir / "table_power_distribution.tex"
        table_path.write_text("\n".join(lines))
        generated.append(str(table_path))
        logger.info(f"  Generated: {table_path}")

    # --- Table: VAD ICC ---
    vad = all_results.get("vad", {})
    if vad.get("icc_overall"):
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{ICC(2,1) for VAD dimensions by CEI subtype.}",
            r"\label{tab:vad-icc}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Subtype & Valence & Arousal & Dominance & Confidence \\",
            r"\midrule",
        ]
        for st in SUBTYPES:
            info = vad.get("icc_per_subtype", {}).get(st, {})
            vals = [
                f"{info.get(d, '—'):.3f}" if isinstance(info.get(d), (int, float)) else "—"
                for d in VAD_DIMENSIONS
            ]
            st_display = st.replace("-", " ").title()
            lines.append(f"{st_display} & {' & '.join(vals)} \\\\")
        lines.append(r"\midrule")
        ov_vals = [
            f"{vad['icc_overall'].get(d, '—'):.3f}"
            if isinstance(vad["icc_overall"].get(d), (int, float)) else "—"
            for d in VAD_DIMENSIONS
        ]
        lines.append(f"Overall & {' & '.join(ov_vals)} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        table_path = output_dir / "table_vad_icc.tex"
        table_path.write_text("\n".join(lines))
        generated.append(str(table_path))
        logger.info(f"  Generated: {table_path}")

    # --- Table: Baseline Results (if available) ---
    baseline_analysis_path = output_dir / "baseline_analysis.json"
    if baseline_analysis_path.exists():
        with open(baseline_analysis_path) as f:
            baseline_analysis = json.load(f)
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{LLM baseline performance on CEI-ToM speaker emotion prediction.}",
            r"\label{tab:baselines}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Model & Accuracy & Macro-F1 \\",
            r"\midrule",
        ]
        for model_name, info in baseline_analysis.items():
            acc = info.get("accuracy", 0)
            f1 = info.get("macro_f1", 0)
            display_name = model_name.replace("-", " ").title()
            lines.append(f"{display_name} & {acc:.3f} & {f1:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        table_path = output_dir / "table_baselines.tex"
        table_path.write_text("\n".join(lines))
        generated.append(str(table_path))
        logger.info(f"  Generated: {table_path}")

    # --- Figure: VAD distributions (requires matplotlib) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vad_dist = vad.get("distributions", {})
        if vad_dist:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            dim_labels = {"v": "Valence", "a": "Arousal", "d": "Dominance"}
            for idx, (dim, dim_label) in enumerate(dim_labels.items()):
                means = []
                labels = []
                for st in SUBTYPES:
                    d = vad_dist.get(st, {}).get(dim, {})
                    if "mean" in d:
                        means.append(d["mean"])
                        labels.append(st.replace("-", "\n").title())
                if means:
                    bars = axes[idx].bar(range(len(means)), means, color=f"C{idx}")
                    axes[idx].set_xticks(range(len(means)))
                    axes[idx].set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
                    axes[idx].set_ylabel(f"Mean {dim_label}")
                    axes[idx].set_title(dim_label)
                    axes[idx].axhline(0, color="gray", linewidth=0.5, linestyle="--")
            fig.suptitle("VAD Dimension Means by CEI Subtype", fontsize=12)
            fig.tight_layout()
            for fmt in ["pdf", "png"]:
                fig_path = output_dir / f"fig_vad_distributions.{fmt}"
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                generated.append(str(fig_path))
            plt.close(fig)
            logger.info(f"  Generated: fig_vad_distributions.pdf/png")
    except ImportError:
        logger.warning("  matplotlib not available — skipping figures")

    # --- Figure: Emotion-VAD consistency ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        consistency = vad.get("consistency", {})
        if consistency:
            emotions = [e for e in PLUTCHIK_EMOTIONS if e in consistency]
            vals = [consistency[e]["mean_valence"] for e in emotions]
            colors = [
                "#2ecc71" if v > 0.1 else "#e74c3c" if v < -0.1 else "#95a5a6"
                for v in vals
            ]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(range(len(emotions)), vals, color=colors)
            ax.set_yticks(range(len(emotions)))
            ax.set_yticklabels([e.title() for e in emotions])
            ax.set_xlabel("Mean Valence")
            ax.set_title("Mean Valence by Gold-Standard Emotion")
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
            fig.tight_layout()
            for fmt in ["pdf", "png"]:
                fig_path = output_dir / f"fig_emotion_valence.{fmt}"
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                generated.append(str(fig_path))
            plt.close(fig)
            logger.info(f"  Generated: fig_emotion_valence.pdf/png")
    except ImportError:
        pass

    # Save all results as JSON master file
    master_path = output_dir / "dmlr2026_all_results.json"
    with open(master_path, "w") as f:
        # Remove non-serializable items
        serializable = {}
        for k, v in all_results.items():
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
        json.dump(serializable, f, indent=2, default=str)
    generated.append(str(master_path))
    logger.info(f"  Master results: {master_path}")

    return {"generated": generated, "count": len(generated)}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

ALL_LOCAL_STAGES = [
    "verify_agreement",
    "verify_power",
    "human_performance",
    "vad_analysis",
    "scale_justification",
    "create_splits",
    "extract_examples",
]

ALL_STAGES = ALL_LOCAL_STAGES + [
    "run_baselines",
    "analyze_baselines",
    "generate_outputs",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DMLR 2026 Pipeline — CEI-ToM Dataset Paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Stages:
  all_local           Run all local stages (no API calls, $0 cost)
  all                 Run everything (local + baselines + outputs)
  verify_agreement    R0: Recompute Fleiss' κ per subtype
  verify_power        R0: Verify power distribution
  human_performance   R0: Compute agreement patterns
  vad_analysis        R4: VAD ICC, distributions, consistency
  scale_justification R3: Power analysis, benchmark comparison
  create_splits       R5: Stratified train/val/test splits
  extract_examples    R7: Select candidate worked examples
  run_baselines       R1: Run LLM baselines (requires API keys)
  analyze_baselines   R1: Compute accuracy + macro-F1
  generate_outputs    Generate LaTeX tables + figures

Examples:
  python scripts/run_pipeline_dmlr2026.py --stage all_local
  python scripts/run_pipeline_dmlr2026.py --stage run_baselines --dry-run
  python scripts/run_pipeline_dmlr2026.py --stage all --model gpt-5-mini --model llama-3.1-70b
""",
    )
    parser.add_argument(
        "--stage",
        default="all_local",
        help="Stage to run (default: all_local)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/human-gold"),
        help="Path to merged CSV directory (default: data/human-gold)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/dmlr2026"),
        help="Output directory (default: reports/dmlr2026)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config-dmlr.yml"),
        help="Configuration file (default: config/config-dmlr.yml)",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Baseline model to run (can be repeated; default: all available)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--prompt-mode",
        choices=["zero-shot", "cot", "few-shot"],
        default="zero-shot",
        help="Prompt mode: zero-shot (default), cot (chain-of-thought), few-shot (3-shot)",
    )
    args = parser.parse_args()

    # Seed for reproducibility (bootstrap CIs, stratified splits, etc.)
    random.seed(args.seed)

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs")
    log_path = logs_dir / f"log_dmlr2026_pipeline_{timestamp}.txt"
    logger = DualLogger(log_path)

    # Load DMLR config
    if args.config.exists():
        dmlr_config = load_dmlr_config(args.config)
    else:
        logger.warning(f"Config not found: {args.config} — using empty config")
        dmlr_config = {}

    model_registry = build_model_registry(dmlr_config)
    pricing = get_pricing(dmlr_config)

    logger.info("=" * 60)
    logger.info("DMLR 2026 Pipeline — CEI-ToM Dataset Paper")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Data dir: {args.data_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    if model_registry:
        logger.info(f"Models ({len(model_registry)}): {list(model_registry.keys())}")
    logger.info("=" * 60)

    # Load data
    logger.info("")
    logger.info("[Data] Loading merged CSVs...")
    data = load_merged_data(args.data_dir)
    n_loaded = sum(len(rows) for rows in data.values())
    logger.info(f"  Loaded {n_loaded} scenarios across {len(data)} subtypes")
    for st, rows in data.items():
        names = detect_annotator_names(rows)
        logger.info(f"  {st}: {len(rows)} scenarios, annotators={names}")

    if n_loaded == 0:
        logger.error(f"No data found in {args.data_dir}. Exiting.")
        return 1

    # Determine stages to run
    stage = args.stage.lower()
    if stage == "all_local":
        stages = ALL_LOCAL_STAGES
    elif stage == "all":
        stages = ALL_STAGES
    else:
        stages = [stage]

    # Trackers
    call_tracker = APICallTracker()
    retry_config = RetryConfig()

    # Results accumulator
    all_results: dict[str, Any] = {}

    # Run stages
    logger.info("")
    for s in stages:
        logger.info("")
        if s == "verify_agreement":
            all_results["agreement"] = stage_verify_agreement(data, logger)
        elif s == "verify_power":
            all_results["power"] = stage_verify_power(data, logger)
        elif s == "human_performance":
            all_results["human_performance"] = stage_human_performance(data, logger)
        elif s == "vad_analysis":
            all_results["vad"] = stage_vad_analysis(data, logger)
        elif s == "scale_justification":
            agreement = all_results.get("agreement", {})
            all_results["scale"] = stage_scale_justification(data, agreement, logger)
        elif s == "create_splits":
            all_results["splits"] = stage_create_splits(data, logger, seed=args.seed)
        elif s == "extract_examples":
            all_results["examples"] = stage_extract_examples(data, logger)
        elif s == "run_baselines":
            all_results["baselines"] = stage_run_baselines(
                data, logger, call_tracker, retry_config,
                args.output_dir, model_registry, pricing,
                args.models, args.dry_run, args.resume,
                prompt_mode=args.prompt_mode,
            )
        elif s == "analyze_baselines":
            all_results["baseline_analysis"] = stage_analyze_baselines(
                args.output_dir, logger,
                prompt_mode=args.prompt_mode,
            )
        elif s == "generate_outputs":
            all_results["outputs"] = stage_generate_outputs(
                all_results, args.output_dir, logger,
            )
        else:
            logger.error(f"Unknown stage: {s}")
            return 1

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Stages run: {stages}")
    logger.info(f"API calls: {call_tracker.total_count}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Log: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
