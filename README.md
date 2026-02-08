# CEI Dataset: DMLR 2026 Replication Guide

**Paper:** *CEI: A Benchmark for Evaluating Pragmatic Reasoning in Language Models*
**Venue:** Journal of Data-centric Machine Learning Research (DMLR)

---

## Overview

The CEI-ToM benchmark evaluates how well language models can infer a **speaker's emotional state** from contextually complex utterances requiring Theory of Mind reasoning.  The dataset contains **300 expert-authored scenarios** across 5 pragmatic communication subtypes, each annotated by 3 independent human raters.

### Dataset at a Glance

| Property | Value |
|----------|-------|
| Scenarios | 300 (60 per subtype) |
| Subtypes | sarcasm-irony, mixed-signals, passive-aggression, deflection-misdirection, strategic-politeness |
| Annotators | 15 total (3 per subtype, assigned by subtype) |
| Emotion labels | Plutchik's 8 basic emotions |
| VAD scales | 7-point Valence, Arousal, Dominance, Confidence |
| Gold standard | Adjudicated per-scenario label |

---

## Quick Start

### 1. Install

```bash
# Clone and set up environment
git clone <anonymous> && cd cei-tom-dataset-base
uv venv && source .venv/bin/activate
uv pip install -e "."
```

**Requirements:** Python 3.10+, `pyyaml` (included).  Matplotlib is optional (for figures).

### 2. Run All Local Analysis (no API calls, $0 cost)

```bash
python scripts/run_pipeline_dmlr2026.py --stage all_local
```

This runs in ~30 seconds and produces:
- Fleiss' kappa per subtype + 95% bootstrap CIs
- Power distribution counts (peer / high-to-low / low-to-high)
- Human agreement patterns (unanimous / majority / split)
- VAD ICC(2,1) per dimension per subtype
- Scale justification (power analysis, benchmark comparison)
- Stratified train/val/test splits (70/15/15)
- Candidate worked examples

### 3. Generate Paper Tables & Figures

```bash
python scripts/run_pipeline_dmlr2026.py --stage all
```

Outputs to `reports/dmlr2026/`:
- `table_kappa.tex` -- Inter-annotator agreement
- `table_human_agreement.tex` -- Agreement patterns
- `table_power_distribution.tex` -- Power relation distribution
- `table_vad_icc.tex` -- VAD ICC values
- `table_baselines.tex` -- LLM baseline results (if baselines have been run)
- `fig_vad_distributions.pdf` -- VAD dimension means by subtype
- `fig_emotion_valence.pdf` -- Mean valence by gold-standard emotion
- `dmlr2026_all_results.json` -- All computed results (machine-readable)

---

## Running LLM Baselines

The paper includes baseline results from 7 models.  Models and pricing are defined in `config/config-dmlr.yml`.

### Baseline Models

| Model | Provider | API Key Env Var | Est. Cost/300 |
|-------|----------|-----------------|---------------|
| GPT-4o | OpenAI | `OPENAI_API_KEY` | $1.20 |
| Claude Sonnet 4.5 | Anthropic | `ANTHROPIC_API_KEY` | $1.62 |
| Grok-4 | xAI | `XAI_API_KEY` | $1.62 |
| Gemini 2.5 Flash | Google | `GOOGLE_API_KEY` | $0.22 |
| Llama-3.1-70B | Together | `TOGETHER_API_KEY` | $0.26 |
| DeepSeek V3.1 | Fireworks | `FIREWORKS_API_KEY` | $0.08 |
| Phi-4 | Together | `TOGETHER_API_KEY` | $0.06 |

### Configure API Keys

Set one or more of:

```bash
export OPENAI_API_KEY="sk-..."          # For GPT-4o
export ANTHROPIC_API_KEY="sk-ant-..."   # For Claude Sonnet 4.5
export XAI_API_KEY="xai-..."            # For Grok-4
export GOOGLE_API_KEY="..."             # For Gemini 2.5 Flash
export TOGETHER_API_KEY="..."           # For Llama-3.1-70B and Phi-4
export FIREWORKS_API_KEY="..."          # For DeepSeek V3.1
```

The pipeline runs whichever models have keys configured and skips the rest.

### Dry Run (estimate cost, no API calls)

```bash
python scripts/run_pipeline_dmlr2026.py --stage run_baselines --dry-run
```

Shows per-model cost estimates and API key status -- no keys required.

### Run Baselines

```bash
# All available models (those with API keys set)
python scripts/run_pipeline_dmlr2026.py --stage run_baselines

# Specific models only
python scripts/run_pipeline_dmlr2026.py --stage run_baselines \
    --model gpt-4o --model llama-3.1-70b --model phi-4

# Resume from checkpoint after interruption
python scripts/run_pipeline_dmlr2026.py --stage run_baselines --resume
```

**Estimated API costs:** ~$5.07 for all 7 models (2,100 calls), ~$3.14 for recommended 4 (GPT-4o + Claude Sonnet + Llama-70B + Phi-4).

### Analyze Baseline Results

```bash
python scripts/run_pipeline_dmlr2026.py --stage analyze_baselines
```

Computes per-model accuracy and macro-F1, saved to `reports/dmlr2026/baseline_analysis.json`.

---

## Pipeline Stages Reference

| Stage | R# | API Calls | Description |
|-------|----|-----------|-------------|
| `verify_agreement` | R0 | 0 | Fleiss' kappa per subtype + overall |
| `verify_power` | R0 | 0 | Power relation distribution |
| `human_performance` | R0 | 0 | Unanimous/majority/split patterns |
| `vad_analysis` | R4 | 0 | ICC(2,1), distributions, emotion-VAD consistency |
| `scale_justification` | R3 | 0 | Power analysis, CI widths, benchmarks |
| `create_splits` | R5 | 0 | Stratified train/val/test (70/15/15) |
| `extract_examples` | R7 | 0 | Candidate worked examples |
| `run_baselines` | R1 | ~2,100 | LLM inference (7 models x 300 scenarios) |
| `analyze_baselines` | R1 | 0 | Accuracy + macro-F1 |
| `generate_outputs` | -- | 0 | LaTeX tables + figures |

**Shortcut stages:**
- `all_local` -- All stages except baselines (default)
- `all` -- Everything including baselines

---

## Data Layout

```
data/human-gold/                     # Merged CSVs (pipeline input)
  data_deflection-misdirection.csv   #   60 scenarios, 3 annotators
  data_mixed-signals.csv
  data_passive-aggression.csv
  data_sarcasm-irony.csv
  data_strategic-politeness.csv

reports/dmlr2026/                    # Pipeline outputs
  table_*.tex                        #   LaTeX tables
  fig_*.pdf                          #   Figures
  dmlr2026_all_results.json          #   All results (JSON)
  baseline_results.json              #   Raw model responses
  baseline_analysis.json             #   Accuracy/F1 metrics
```

### Merged CSV Schema

| Column | Description |
|--------|-------------|
| `id` | Scenario ID (1-60 within subtype) |
| `sd_situation` | Situational context |
| `sd_utterance` | Speaker's utterance |
| `sd_speaker_role` | Speaker's role/relationship |
| `sd_listener_role` | Listener's role/relationship |
| `sl_plutchik_primary_<Annotator_N>` | Annotator's emotion label |
| `gold_standard` | Adjudicated ground truth emotion |
| `sl_v_<Annotator_N>`, `sl_a_<Annotator_N>`, `sl_d_<Annotator_N>` | Per-annotator VAD ratings (7-point text labels) |
| `sl_confidence_<Annotator_N>` | Per-annotator confidence |

---

## Configuration

The pipeline reads `config/config-dmlr.yml` for model definitions and pricing.  Override data and output paths via CLI:

```bash
python scripts/run_pipeline_dmlr2026.py \
    --data-dir /path/to/merged/csvs \
    --output-dir /path/to/output \
    --config /path/to/custom-config.yml \
    --seed 42
```

---

## Reproducibility Notes

- **Random seed:** All stochastic operations (bootstrap CIs, stratified splits) use `--seed 42` by default
- **VAD mapping:** 7 text labels per dimension mapped to [-1.0, +1.0] at equal intervals
- **Baseline prompt:** Asks about the **speaker's** emotion
- **Safety limits:** API call tracker caps at 500/model, 2000 total; checkpoints every 50 calls
- **Annotation target:** Labels describe the speaker's emotional state, not the listener's response
- **Model selection:** All 7 baseline models are defined in `config/config-dmlr.yml` under `models.complete`

---

## Project Structure

```
scripts/run_pipeline_dmlr2026.py     # DMLR pipeline (all stages)
scripts/generate_figures.py          # Paper figure generation
config/config-dmlr.yml               # Model ensemble + pricing
data/human-gold/                     # Merged annotated CSVs
prompts/                             # Prompt templates (zero-shot, CoT, few-shot)
reports/dmlr2026/                    # Pipeline outputs (tables, figures, JSON)
```
