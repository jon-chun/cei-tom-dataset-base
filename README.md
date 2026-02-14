# CEI Benchmark: Replication Guide

**Paper:** *CEI: A Benchmark for Evaluating Pragmatic Reasoning in Language Models*
**Venue:** Journal of Data-centric Machine Learning Research (DMLR)

---

## Overview

The CEI-ToM benchmark evaluates how well language models can infer a **speaker's emotional state** from contextually complex utterances requiring Theory of Mind reasoning. The dataset contains **300 expert-authored scenarios** across 5 pragmatic communication subtypes, each annotated by 3 independent human raters.

| Property | Value |
|----------|-------|
| Scenarios | 300 (60 per subtype) |
| Subtypes | sarcasm-irony, mixed-signals, passive-aggression, deflection-misdirection, strategic-politeness |
| Annotators | 15 total (3 per subtype) |
| Emotion labels | Plutchik's 8 basic emotions |
| VAD scales | 7-point Valence, Arousal, Dominance, Confidence |
| Gold standard | Adjudicated per-scenario label |

---

## Quick Start

### 1. Install

```bash
git clone <anonymous> && cd cei-tom-dataset-base
uv venv && source .venv/bin/activate
uv pip install -e "."                    # core (pyyaml + python-dotenv)
uv pip install -e ".[figures]"           # adds matplotlib, numpy, scipy
```

**Requirements:** Python 3.10+. Alternative: `pip install -r requirements.txt`.

### 2. Run All Local Analysis (~30s, no API calls)

```bash
python scripts/run_pipeline_dmlr2026.py --stage all_local
```

Produces inter-annotator agreement (Fleiss' kappa + bootstrap CIs), power distributions, human agreement patterns, VAD ICC(2,1), scale justification, stratified splits (70/15/15), and worked examples.

### 3. Generate Paper Figures

```bash
python scripts/generate_figures.py
```

Outputs 4 figures to `reports/dmlr2026/figures/`: confusion matrix, difficulty distribution, linguistic analysis, agreement by subtype.

---

## Running LLM Baselines

Models and pricing are defined in `config/config.yml`.

### Baseline Models

| Model | Provider | API Key Env Var | Est. Cost/300 |
|-------|----------|-----------------|---------------|
| GPT-5-mini | OpenAI | `OPENAI_API_KEY` | $0.15 |
| Claude Sonnet 4.5 | Anthropic | `ANTHROPIC_API_KEY` | $1.53 |
| Grok-4.1-fast | xAI | `XAI_API_KEY` | $0.09 |
| Gemini 2.5 Flash | Google | `GOOGLE_API_KEY` | $0.18 |
| Llama-3.1-70B | Together | `TOGETHER_API_KEY` | $0.34 |
| DeepSeek V3.1 | Fireworks | `FIREWORKS_API_KEY` | $0.08 |
| Qwen2.5-7B | Together | `TOGETHER_API_KEY` | $0.12 |

**Estimated total:** ~$2.49 for all 7 models (2,100 calls). Use `--dry-run` for exact estimates.

### Configure API Keys

```bash
export OPENAI_API_KEY="sk-..."          # GPT-5-mini
export ANTHROPIC_API_KEY="sk-ant-..."   # Claude Sonnet 4.5
export XAI_API_KEY="xai-..."            # Grok-4.1-fast
export GOOGLE_API_KEY="..."             # Gemini 2.5 Flash
export TOGETHER_API_KEY="..."           # Llama-3.1-70B + Qwen2.5-7B
export FIREWORKS_API_KEY="..."          # DeepSeek V3.1
```

The pipeline runs whichever models have keys configured and skips the rest.

### Run Baselines

```bash
# Dry run (cost estimate only, no API keys needed)
python scripts/run_pipeline_dmlr2026.py --stage run_baselines --dry-run

# All available models
python scripts/run_pipeline_dmlr2026.py --stage run_baselines

# Specific models only
python scripts/run_pipeline_dmlr2026.py --stage run_baselines \
    --model gpt-5-mini --model llama-3.1-70b

# Resume from checkpoint after interruption
python scripts/run_pipeline_dmlr2026.py --stage run_baselines --resume

# Chain-of-thought or few-shot prompting
python scripts/run_pipeline_dmlr2026.py --stage run_baselines --prompt-mode cot
python scripts/run_pipeline_dmlr2026.py --stage run_baselines --prompt-mode few-shot
```

### Analyze Results

```bash
python scripts/run_pipeline_dmlr2026.py --stage analyze_baselines
```

Computes per-model accuracy and macro-F1, saved to `reports/dmlr2026/baseline_analysis.json`.

---

## Pipeline Stages

| Stage | API Calls | Description |
|-------|-----------|-------------|
| `verify_agreement` | 0 | Fleiss' kappa per subtype + overall |
| `verify_power` | 0 | Power relation distribution |
| `human_performance` | 0 | Unanimous/majority/split patterns |
| `vad_analysis` | 0 | ICC(2,1), distributions, emotion-VAD consistency |
| `scale_justification` | 0 | Power analysis, CI widths, benchmarks |
| `create_splits` | 0 | Stratified train/val/test (70/15/15) |
| `extract_examples` | 0 | Candidate worked examples |
| `run_baselines` | ~2,100 | LLM inference (7 models x 300 scenarios) |
| `analyze_baselines` | 0 | Accuracy + macro-F1 |
| `generate_outputs` | 0 | LaTeX tables + figures |

**Shortcuts:** `all_local` (all except baselines, default), `all` (everything).

---

## Data

### Layout

```
data/human-gold/                     # Input: 5 merged CSVs
  data_sarcasm-irony.csv             #   60 scenarios, 3 annotators each
  data_mixed-signals.csv
  data_passive-aggression.csv
  data_deflection-misdirection.csv
  data_strategic-politeness.csv
```

### CSV Schema

| Column | Description |
|--------|-------------|
| `id` | Scenario ID (1-60 within subtype) |
| `sd_situation` | Situational context |
| `sd_utterance` | Speaker's utterance |
| `sd_speaker_role` | Speaker's role/relationship |
| `sd_listener_role` | Listener's role/relationship |
| `gold_standard` | Adjudicated ground truth emotion |
| `sl_plutchik_primary_<Annotator_N>` | Per-annotator emotion label |
| `sl_v_<N>`, `sl_a_<N>`, `sl_d_<N>` | Per-annotator VAD ratings (7-point text labels) |
| `sl_confidence_<Annotator_N>` | Per-annotator confidence |

---

## Configuration

The pipeline reads `config/config.yml` for model definitions and pricing. Override paths via CLI:

```bash
python scripts/run_pipeline_dmlr2026.py \
    --data-dir /path/to/csvs \
    --output-dir /path/to/output \
    --config /path/to/config.yml \
    --seed 42
```

---

## Reproducibility

- **Random seed:** All stochastic operations use `--seed 42` by default
- **VAD mapping:** 7 text labels per dimension mapped to [-1.0, +1.0] at equal intervals
- **Baseline prompt:** Asks about the **speaker's** emotion (the annotation target)
- **Safety limits:** API call tracker caps at 500/model, 2000 total; checkpoints every 50 calls
- **Model selection:** All 7 baseline models defined in `config/config.yml` under `models.complete`
- **Prompt modes:** Zero-shot (default), chain-of-thought, and few-shot (3 examples)

---

## Project Structure

```
config/config.yml                    # Model definitions + pricing
data/human-gold/                     # Annotated scenario CSVs (5 files)
prompts/                             # Prompt templates (zero-shot, CoT, few-shot)
scripts/
  run_pipeline_dmlr2026.py           # Main pipeline (all stages)
  generate_figures.py                # Paper figures from CSV data
  generate_model_confusion_matrix.py # Model confusion matrix from baseline results
  qa_pipeline.py                     # 4-level QA (schema, stats, agreement, adjudication)
reports/dmlr2026/                    # Pipeline outputs (tables, figures, JSON)
DATASHEET.md                         # Gebru et al. datasheet
REPRODUCIBILITY.md                   # Pineau reproducibility checklist
CHANGELOG.md                         # Version history
LICENSE                              # CC-BY-4.0 (data) + MIT (code)
```
