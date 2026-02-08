#!/usr/bin/env python3
"""DMLR 2026 Pipeline Runner — CEI-ToM Dataset Paper.

Implements all computational tasks for the DMLR dataset paper (R0–R9):
  Phase 0 (local):  Data verification, κ, power distribution, human performance,
                     VAD analysis, scale justification, stratified splits, examples
  Phase 1 (API):    Baseline inference (models loaded from
                     config/config-dmlr.yml)
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
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

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

# DMLR-specific prompt: asks about the SPEAKER's emotion.
# This is the ground-truth annotation target.
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

# Provider → API key environment variable
PROVIDER_API_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# OpenAI-compatible chat/completions endpoints (all except anthropic and google)
OPENAI_COMPAT_ENDPOINTS: dict[str, str] = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
    "together": "https://api.together.xyz/v1/chat/completions",
    "fireworks": "https://api.fireworks.ai/inference/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}

# Per-provider max concurrent requests (shared across all models from same provider)
PROVIDER_CONCURRENCY: dict[str, int] = {
    "openai": 10,
    "anthropic": 5,
    "xai": 5,
    "google": 10,
    "together": 4,    # Llama-3.1-70B
    "fireworks": 4,
    "openrouter": 4,
}


def _short_model_name(model_id: str) -> str:
    """Derive a CLI-friendly short name from a full model ID.

    Examples:
        "gpt-4o"                                    → "gpt-4o"
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
        {"gpt-4o": {"provider": "openai", "model_id": "gpt-4o",
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
# Utility classes
# ---------------------------------------------------------------------------


@dataclass
class APICallTracker:
    """Track API calls per model and total, with safety limits (thread-safe)."""

    model_counts: dict[str, int] = field(default_factory=dict)
    total_count: int = 0
    model_max: int = 500
    total_max: int = 2000

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def increment(self, model_id: str) -> bool:
        with self._lock:
            self.model_counts[model_id] = self.model_counts.get(model_id, 0) + 1
            self.total_count += 1
            return (
                self.model_counts[model_id] <= self.model_max
                and self.total_count <= self.total_max
            )

    def check_limits(self, model_id: str) -> tuple[bool, str]:
        with self._lock:
            if self.model_counts.get(model_id, 0) >= self.model_max:
                return False, f"Model {model_id} reached limit ({self.model_max})"
            if self.total_count >= self.total_max:
                return False, f"Total calls reached limit ({self.total_max})"
            return True, ""

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total": self.total_count,
                "limit": self.total_max,
                "per_model": dict(self.model_counts),
            }


@dataclass
class RetryConfig:
    max_retries: int = 5
    base_delay: float = 2.0
    max_delay: float = 60.0
    jitter_factor: float = 0.25

    def delay(self, attempt: int) -> float:
        d = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = d * self.jitter_factor * (2 * random.random() - 1)
        return max(0.1, d + jitter)

    def delay_rate_limit(
        self, attempt: int, retry_after: Optional[float] = None,
    ) -> float:
        """Longer backoff for HTTP 429 rate-limit responses."""
        if retry_after and retry_after > 0:
            return retry_after + self.jitter_factor * random.random()
        d = min(self.base_delay * (2 ** (attempt + 1)), self.max_delay)
        jitter = d * self.jitter_factor * (2 * random.random() - 1)
        return max(1.0, d + jitter)


class ProviderThrottler:
    """Per-provider concurrency limiter with adaptive rate-limit backoff.

    Each provider gets a semaphore controlling max concurrent requests
    and an adaptive delay that increases on 429s and decays on success.
    """

    def __init__(self, max_concurrent: int = 10, min_delay: float = 0.05) -> None:
        self.semaphore = threading.Semaphore(max_concurrent)
        self._delay = min_delay
        self._min_delay = min_delay
        self._lock = threading.Lock()

    @property
    def delay(self) -> float:
        with self._lock:
            return self._delay

    def on_rate_limit(self, retry_after: Optional[float] = None) -> None:
        """Increase inter-request delay after a 429 response."""
        with self._lock:
            if retry_after and retry_after > 0:
                self._delay = max(self._delay, retry_after / 2)
            else:
                self._delay = min(self._delay * 2 + 0.5, 30.0)

    def on_success(self) -> None:
        """Gradually reduce delay after successful requests."""
        with self._lock:
            self._delay = max(self._delay * 0.95, self._min_delay)


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

    # Emotion-VAD consistency: mean valence per annotator's own emotion label
    # Each annotator's valence rating is grouped by that annotator's own
    # emotion label (not the gold standard), giving a cleaner measure of
    # whether annotators' dimensional and categorical ratings align.
    emotion_valence: dict[str, list[float]] = {e: [] for e in PLUTCHIK_EMOTIONS}
    for subtype in SUBTYPES:
        rows = data.get(subtype, [])
        if not rows:
            continue
        names = detect_annotator_names(rows)
        for row in rows:
            for nm in names:
                emo = row.get(f"sl_plutchik_primary_{nm}", "").strip().lower()
                if emo not in emotion_valence:
                    continue
                v = vad_to_numeric(row.get(f"sl_v_{nm}", ""), "v")
                if v is not None:
                    emotion_valence[emo].append(v)

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

    # Save splits to disk for reproducibility
    splits_path = Path("reports/dmlr2026/splits.json")
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps({
        "seed": seed,
        "fractions": {"train": train_frac, "val": val_frac, "test": round(test_frac, 2)},
        "assignments": split_assignments,
    }, indent=2))
    logger.info(f"  Saved split assignments: {splits_path}")

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
    # Prefer subtypes that best showcase pragmatic diversity
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
) -> Optional[str]:
    """Call any OpenAI-compatible chat/completions endpoint.

    Works for: OpenAI, xAI (Grok), Together AI, Fireworks AI.
    """
    import urllib.request

    body = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "CEI-ToM/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _call_anthropic(model_id: str, prompt: str, api_key: str) -> Optional[str]:
    """Call Anthropic Messages API."""
    import urllib.request

    body = json.dumps({
        "model": model_id,
        "max_tokens": 60,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "User-Agent": "CEI-ToM/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


def _call_google(model_id: str, prompt: str, api_key: str) -> Optional[str]:
    """Call Google Gemini generateContent API."""
    import urllib.request

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 1024,
            "temperature": 0.0,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "CEI-ToM/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError):
        # Gemini may block content or return empty responses
        return None


def _call_model(
    model_name: str,
    prompt: str,
    model_registry: dict[str, dict[str, str]],
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
        return _call_anthropic(model_id, prompt, api_key)
    elif provider == "google":
        return _call_google(model_id, prompt, api_key)
    else:
        # OpenAI-compatible: openai, xai, together, fireworks
        endpoint = OPENAI_COMPAT_ENDPOINTS.get(provider)
        if not endpoint:
            return None
        return _call_openai_compat(endpoint, model_id, prompt, api_key)


def _parse_emotion_response(text: str) -> Optional[str]:
    """Extract emotion from model response (expects JSON with 'emotion' key)."""
    if not text:
        return None
    # Try JSON parse first
    try:
        # Handle markdown-wrapped JSON
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

    # Fallback: look for emotion keywords in text
    text_lower = text.lower()
    for emo in PLUTCHIK_EMOTIONS:
        if emo in text_lower:
            return emo
    return None


class _ProgressTracker:
    """Real-time progress tracking for parallel baseline execution.

    Thread-safe counters for completed/cached/failed tasks, retry attempts,
    rate-limit hits, and per-model cost estimation.  Used by a background
    reporter thread to print status every 60 seconds.
    """

    def __init__(
        self,
        total_tasks: int,
        model_registry: dict[str, dict[str, str]],
        pricing: dict[str, Any],
        est_input_tokens: int = 800,
        est_output_tokens: int = 200,
    ) -> None:
        self._lock = threading.Lock()
        self.total_tasks = total_tasks
        self.completed = 0
        self.cached = 0
        self.api_success = 0
        self.api_failed = 0
        self.retries = 0
        self.rate_limits = 0
        self.per_model: dict[str, dict[str, int]] = {}
        self.start_time = time.time()
        self._model_registry = model_registry
        self._pricing = pricing
        self._est_input = est_input_tokens
        self._est_output = est_output_tokens

    def record_complete(self, model_name: str, *, cached: bool = False) -> None:
        with self._lock:
            self.completed += 1
            m = self.per_model.setdefault(
                model_name, {"success": 0, "failed": 0, "cached": 0},
            )
            if cached:
                self.cached += 1
                m["cached"] += 1
            else:
                self.api_success += 1
                m["success"] += 1

    def record_failure(self, model_name: str) -> None:
        with self._lock:
            self.completed += 1
            m = self.per_model.setdefault(
                model_name, {"success": 0, "failed": 0, "cached": 0},
            )
            self.api_failed += 1
            m["failed"] += 1

    def record_retry(self, *, is_rate_limit: bool = False) -> None:
        with self._lock:
            self.retries += 1
            if is_rate_limit:
                self.rate_limits += 1

    def _model_cost_usd(self, model_name: str, n_calls: int) -> float:
        cfg = self._model_registry.get(model_name, {})
        provider = cfg.get("provider", "")
        model_id = cfg.get("model_id", "")
        pp = self._pricing.get(provider, {})
        mp = (
            pp.get(model_id)
            or pp.get(model_id.split("/")[-1])
            or pp.get("_default", {})
        )
        inp = mp.get("input", 0)
        out = mp.get("output", 0)
        return n_calls * (self._est_input * inp + self._est_output * out) / 1_000_000

    def estimated_cost(self) -> float:
        with self._lock:
            return sum(
                self._model_cost_usd(m, c["success"])
                for m, c in self.per_model.items()
            )

    def status_line(self) -> str:
        with self._lock:
            elapsed = time.time() - self.start_time
            pct = (
                self.completed / self.total_tasks * 100
                if self.total_tasks else 0
            )
            rate = self.completed / elapsed if elapsed > 0 else 0
            remaining = (
                (self.total_tasks - self.completed) / rate if rate > 0 else 0
            )

            elapsed_m, elapsed_s = divmod(int(elapsed), 60)
            eta_m, eta_s = divmod(int(remaining), 60)
            cost = sum(
                self._model_cost_usd(m, c["success"])
                for m, c in self.per_model.items()
            )

            return (
                f"  [{elapsed_m:02d}:{elapsed_s:02d}] "
                f"{self.completed}/{self.total_tasks} ({pct:.0f}%) | "
                f"API: {self.api_success} ok, {self.api_failed} fail, "
                f"{self.cached} cached, {self.retries} retries "
                f"({self.rate_limits} 429s) | "
                f"~${cost:.2f} | "
                f"ETA {eta_m}m{eta_s:02d}s"
            )

    def final_summary(self) -> str:
        with self._lock:
            elapsed = time.time() - self.start_time
            elapsed_m, elapsed_s = divmod(int(elapsed), 60)
            cost = sum(
                self._model_cost_usd(m, c["success"])
                for m, c in self.per_model.items()
            )
            lines = [
                f"  Completed {self.completed}/{self.total_tasks} in "
                f"{elapsed_m}m{elapsed_s:02d}s",
                f"  API calls: {self.api_success} success, "
                f"{self.api_failed} failed, {self.cached} cached",
                f"  Retries: {self.retries} total "
                f"({self.rate_limits} rate-limited)",
                f"  Estimated cost: ~${cost:.2f}",
            ]
            for m in sorted(self.per_model):
                c = self.per_model[m]
                mc = self._model_cost_usd(m, c["success"])
                lines.append(
                    f"    {m}: {c['success']} ok, {c['failed']} fail, "
                    f"{c['cached']} cached (~${mc:.3f})"
                )
            return "\n".join(lines)


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
) -> dict[str, Any]:
    """R1: Run baseline models on all scenarios with parallel execution.

    Architecture:
      - All models run concurrently via ThreadPoolExecutor
      - Per-provider semaphores limit concurrent requests (e.g., Together=8
        shared by Llama-70B + Phi-4)
      - Adaptive throttling: 429 responses increase per-provider delay;
        successful requests decay it back
      - Tasks interleaved round-robin across models for balanced utilization
      - Thread-safe checkpointing every 50 completions per model
    """
    logger.info("[R1] Baseline inference (parallel)")
    if models is None:
        models = list(model_registry.keys())

    if dry_run:
        # Dry-run shows cost estimates for all requested models (no keys needed)
        n_scenarios = sum(len(rows) for rows in data.values())
        total_calls = n_scenarios * len(models)
        logger.info(f"  DRY RUN: would make {total_calls} API calls")
        logger.info(f"  Models: {models}")
        # Estimate cost from config pricing
        est_input_tokens = n_scenarios * 800  # ~800 input tokens/scenario
        est_output_tokens = n_scenarios * 200  # ~200 output tokens/scenario
        total_cost = 0.0
        for m in models:
            cfg = model_registry.get(m, {})
            provider = cfg.get("provider", "")
            model_id = cfg.get("model_id", "")
            provider_pricing = pricing.get(provider, {})
            # Try full model_id, then last path segment, then _default
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

    # Load checkpoint if resuming
    checkpoint_path = output_dir / "baseline_checkpoint.json"
    completed: dict[str, str] = {}  # key = "model::id" → response text
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = json.load(f)
        logger.info(f"  Resumed from checkpoint: {len(completed)} results loaded")

    # Create per-provider throttlers for rate limiting
    provider_throttlers: dict[str, ProviderThrottler] = {}
    for model_name in available:
        provider = model_registry[model_name]["provider"]
        if provider not in provider_throttlers:
            max_conc = PROVIDER_CONCURRENCY.get(provider, 5)
            provider_throttlers[provider] = ProviderThrottler(max_conc)

    # Thread-safe shared state
    results: dict[str, list[dict[str, Any]]] = {m: [] for m in available}
    results_lock = threading.Lock()
    checkpoint_lock = threading.Lock()
    progress_counts: dict[str, int] = {m: 0 for m in available}
    errors: list[str] = []
    errors_lock = threading.Lock()
    n_scenarios = len(scenarios)
    total_tasks = n_scenarios * len(available)

    # Progress tracker for periodic status reports (every 60s)
    tracker = _ProgressTracker(total_tasks, model_registry, pricing)

    def _process_one(model_name: str, sc: dict[str, str]) -> None:
        """Process a single (model, scenario) pair with provider throttling."""
        key = f"{model_name}::{sc['subtype']}::{sc['id']}"

        # Check cache
        with checkpoint_lock:
            cached_response = completed.get(key)
        if cached_response is not None:
            emotion = _parse_emotion_response(cached_response)
            with results_lock:
                results[model_name].append({
                    "id": sc["id"], "subtype": sc["subtype"],
                    "gold": sc["gold_standard"], "predicted": emotion,
                    "raw_response": cached_response, "cached": True,
                })
            tracker.record_complete(model_name, cached=True)
            return

        # Check safety limits
        ok, reason = call_tracker.check_limits(model_name)
        if not ok:
            logger.error(f"    Safety limit reached: {reason}")
            tracker.record_failure(model_name)
            return

        prompt = DMLR_BASELINE_PROMPT.format(
            situation=sc["situation"], utterance=sc["utterance"],
            speaker_role=sc["speaker_role"], listener_role=sc["listener_role"],
        )

        provider = model_registry[model_name]["provider"]
        throttler = provider_throttlers[provider]
        raw_response = None

        for attempt in range(retry_config.max_retries):
            wait = 0.0
            success = False
            # Acquire provider semaphore — limits concurrent requests to this
            # provider across all its models (e.g., Together: Llama + Phi)
            with throttler.semaphore:
                # Adaptive inter-request delay (increases on 429s, decays on success)
                d = throttler.delay
                if d > 0:
                    time.sleep(d)
                try:
                    raw_response = _call_model(model_name, prompt, model_registry)
                    call_tracker.increment(model_name)
                    throttler.on_success()
                    success = True
                except Exception as e:
                    is_rate_limit = (
                        hasattr(e, "code") and getattr(e, "code", 0) in (429, 403)
                    )
                    tracker.record_retry(is_rate_limit=is_rate_limit)
                    if is_rate_limit:
                        retry_after = None
                        if hasattr(e, "headers"):
                            ra = getattr(e, "headers", {}).get("Retry-After")
                            if ra:
                                try:
                                    retry_after = float(ra)
                                except (ValueError, TypeError):
                                    pass
                        throttler.on_rate_limit(retry_after)
                        wait = retry_config.delay_rate_limit(
                            attempt, retry_after,
                        )
                        logger.warning(
                            f"    429 {model_name} scenario {sc['id']} "
                            f"(attempt {attempt + 1}, waiting {wait:.1f}s)"
                        )
                    else:
                        wait = retry_config.delay(attempt)
                        logger.warning(
                            f"    Retry {attempt + 1}/{retry_config.max_retries} "
                            f"for {model_name} scenario {sc['id']}: {e}  "
                            f"(waiting {wait:.1f}s)"
                        )
            # Semaphore released — sleep outside to free the slot for others
            if success:
                break
            time.sleep(wait)

        emotion = _parse_emotion_response(raw_response or "")
        with results_lock:
            results[model_name].append({
                "id": sc["id"], "subtype": sc["subtype"],
                "gold": sc["gold_standard"], "predicted": emotion,
                "raw_response": raw_response, "cached": False,
            })

        if raw_response is not None:
            tracker.record_complete(model_name)
        else:
            tracker.record_failure(model_name)

        # Update checkpoint
        with checkpoint_lock:
            completed[key] = raw_response or ""
            progress_counts[model_name] = progress_counts.get(model_name, 0) + 1
            if progress_counts[model_name] % 50 == 0:
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w") as f:
                    json.dump(completed, f)
                logger.info(
                    f"    {model_name}: "
                    f"{progress_counts[model_name]}/{n_scenarios} done"
                )

    # Build interleaved task list (round-robin across models for balanced
    # provider utilization — prevents all threads piling on one provider)
    tasks: list[tuple[str, dict[str, str]]] = []
    for i in range(n_scenarios):
        for model_name in available:
            tasks.append((model_name, scenarios[i]))

    # Worker count = sum of active provider concurrency limits (capped)
    active_providers = {model_registry[m]["provider"] for m in available}
    max_workers = min(
        sum(PROVIDER_CONCURRENCY.get(p, 5) for p in active_providers), 60,
    )

    logger.info(
        f"  Parallel execution: {len(available)} models, "
        f"{n_scenarios} scenarios, {total_tasks} total tasks, "
        f"{max_workers} workers"
    )
    providers_str = ", ".join(
        f"{p}={PROVIDER_CONCURRENCY.get(p, 5)}"
        for p in sorted(active_providers)
    )
    logger.info(f"  Provider concurrency: {providers_str}")

    # Start periodic status reporter (fires every 60 seconds)
    reporter_stop = threading.Event()

    def _status_reporter() -> None:
        while not reporter_stop.wait(timeout=60):
            logger.info(tracker.status_line())

    reporter_thread = threading.Thread(target=_status_reporter, daemon=True)
    reporter_thread.start()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_one, m, sc) for m, sc in tasks
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                with errors_lock:
                    errors.append(str(e))
                logger.error(f"    Unhandled error: {e}")

    # Stop reporter and print final summary
    reporter_stop.set()
    reporter_thread.join(timeout=2)
    logger.info(tracker.final_summary())
    if errors:
        logger.warning(f"  {len(errors)} errors encountered")

    # Final checkpoint + results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f)
    results_path = output_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved to {results_path}")

    elapsed = time.time() - tracker.start_time
    return {
        "status": "complete",
        "models": available,
        "api_calls": call_tracker.summary(),
        "results_path": str(results_path),
        "elapsed_seconds": round(elapsed, 1),
        "estimated_cost_usd": round(tracker.estimated_cost(), 2),
        "errors": len(errors),
    }


# ---------------------------------------------------------------------------
# R1: Baseline analysis (accuracy + macro-F1)
# ---------------------------------------------------------------------------


def stage_analyze_baselines(
    output_dir: Path,
    logger: DualLogger,
) -> dict[str, Any]:
    """R1: Compute accuracy and macro-F1 from baseline results."""
    logger.info("[R1] Analyzing baseline results")
    results_path = output_dir / "baseline_results.json"
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
    analysis_path = output_dir / "baseline_analysis.json"
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

    # If all_results is empty (standalone run), load from saved master JSON
    if not any(k for k in all_results if k != "outputs"):
        master_path = output_dir / "dmlr2026_all_results.json"
        if master_path.exists():
            logger.info(f"  Loading cached results from {master_path}")
            with open(master_path) as f:
                cached = json.load(f)
            all_results.update(cached)
        else:
            logger.warning("  No cached results found. Run analysis stages first.")
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
    "generate_outputs",
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
  python scripts/run_pipeline_dmlr2026.py --stage all --model gpt-4o --model llama-3.1-70b
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
    args = parser.parse_args()

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
            )
        elif s == "analyze_baselines":
            all_results["baseline_analysis"] = stage_analyze_baselines(
                args.output_dir, logger,
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
