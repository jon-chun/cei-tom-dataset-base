# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CEI-ToM (Contextual Emotional Inference and Theory of Mind) is a benchmark dataset for evaluating LLM emotion inference from pragmatically complex utterances. It contains 300 expert-authored scenarios across 5 communication subtypes (sarcasm-irony, mixed-signals, passive-aggression, deflection-misdirection, strategic-politeness), each annotated by 3 human raters using Plutchik's 8 basic emotions and VAD scales. This repo is the anonymous DMLR 2026 submission.

## Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e "."                    # core (pyyaml only)
uv pip install -e ".[figures]"           # adds matplotlib + numpy for figures

# Run all local analysis (no API calls, ~30s)
python scripts/run_pipeline_dmlr2026.py --stage all_local

# Run a single stage
python scripts/run_pipeline_dmlr2026.py --stage verify_agreement
python scripts/run_pipeline_dmlr2026.py --stage vad_analysis

# Dry-run baselines (cost estimate, no API keys needed)
python scripts/run_pipeline_dmlr2026.py --stage run_baselines --dry-run

# Run baselines (requires API keys as env vars)
python scripts/run_pipeline_dmlr2026.py --stage run_baselines
python scripts/run_pipeline_dmlr2026.py --stage run_baselines --model gpt-4o --model phi-4
python scripts/run_pipeline_dmlr2026.py --stage run_baselines --resume  # resume from checkpoint

# Generate paper figures (confusion matrix, difficulty, linguistic, agreement)
python scripts/generate_figures.py

# Generate LaTeX tables + figures from pipeline
python scripts/run_pipeline_dmlr2026.py --stage generate_outputs
```

## Architecture

The entire pipeline is a single self-contained script (`scripts/run_pipeline_dmlr2026.py`) with no imports from a `src/` package. It implements all analysis stages as functions prefixed with `stage_*`.

**Pipeline stages (in order):**
- `verify_agreement` — Fleiss' kappa per subtype + bootstrap CIs
- `verify_power` — Power relation distribution (peer/high-to-low/low-to-high) via keyword matching
- `human_performance` — Unanimous/majority/split agreement patterns
- `vad_analysis` — ICC(2,1) per VAD dimension, distribution stats, emotion-valence consistency
- `scale_justification` — Power analysis, CI widths, benchmark dataset comparisons
- `create_splits` — Stratified train/val/test (70/15/15) balanced by subtype and power relation
- `extract_examples` — Candidate worked examples (majority-vote scenarios from preferred subtypes)
- `run_baselines` — LLM inference across 7 models via provider APIs (checkpoints every 50 calls)
- `analyze_baselines` — Accuracy + macro-F1 from saved results
- `generate_outputs` — LaTeX tables + matplotlib figures to `reports/dmlr2026/`

Shortcut stages: `all_local` (all except baselines), `all` (everything).

**Figure generation:** `scripts/generate_figures.py` reads CSVs directly and generates all 4 paper figures (confusion matrix, difficulty distribution, linguistic analysis, agreement by subtype) to `reports/dmlr2026/figures/`.

**Key data flow:** Merged CSVs in `data/human-gold/` → pipeline stages → JSON results + LaTeX tables in `reports/dmlr2026/`.

## Important Constraints

- **Model exclusion:** DMLR baselines must use only the 7 models listed in `config/config-dmlr.yml` under `models.complete`. Models listed under `excluded_models` must not be used. Never add models from that exclusion list to the DMLR baseline set.
- **Speaker framing:** The DMLR prompt asks about the **speaker's** emotion. The constant `DMLR_BASELINE_PROMPT` encodes this.
- **Anonymization:** This is a blind-review repo. Annotator names are replaced with `Annotator_1` through `Annotator_15`. No author names, institutional references, or identifying URLs should be added. See `docs/copy_postprocessing.md` for the full PII audit.
- **API safety limits:** The `APICallTracker` caps at 500 calls/model and 2000 total, with checkpoints every 50 calls.
- **Reproducibility:** All stochastic operations use `--seed 42` by default. VAD text labels map to [-1.0, +1.0] at equal intervals via `VAD_LABEL_MAP`.

## Repo Structure

- `data/human-gold/` — 5 merged CSVs (60 scenarios each, 3 annotators)
- `config/config-dmlr.yml` — Pipeline configuration, model lists, model exclusions
- `scripts/run_pipeline_dmlr2026.py` — Main analysis pipeline (all stages)
- `scripts/generate_figures.py` — Paper figure generation from CSV data
- `prompts/` — Standardized prompt templates (zero-shot, chain-of-thought, few-shot examples)
- `DATASHEET.md` — Gebru et al. datasheet
- `REPRODUCIBILITY.md` — Pineau reproducibility checklist
- `CHANGELOG.md` — Version history
- `LICENSE` — Dual CC-BY-4.0 (data) + MIT (code)

## Data Schema

Each CSV in `data/human-gold/` has 60 rows (scenarios) with columns: `id`, `sd_situation`, `sd_utterance`, `sd_speaker_role`, `sd_listener_role`, `gold_standard`, plus per-annotator columns `sl_plutchik_primary_<Annotator_N>`, `sl_v_<Annotator_N>`, `sl_a_<Annotator_N>`, `sl_d_<Annotator_N>`, `sl_confidence_<Annotator_N>`.

## LLM Provider Dispatch

API calls route through `_call_model()` which dispatches based on provider: Anthropic and Google have dedicated handlers; OpenAI, xAI, Together, and Fireworks use a shared OpenAI-compatible endpoint handler (`_call_openai_compat`). API keys are read from environment variables defined in `PROVIDER_API_KEYS`.
