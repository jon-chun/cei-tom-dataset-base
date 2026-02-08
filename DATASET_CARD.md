---
license: cc-by-4.0
task_categories:
  - text-classification
language:
  - en
tags:
  - pragmatic-reasoning
  - emotion-inference
  - benchmark
  - social-reasoning
  - theory-of-mind
  - annotation
size_categories:
  - n<1K
---

# CEI Benchmark: Contextual Emotional Inference

## Dataset Summary

The CEI Benchmark evaluates how well language models can infer a speaker's emotional state from pragmatically complex utterances requiring Theory of Mind reasoning. The dataset contains 300 expert-authored scenarios across 5 pragmatic communication subtypes, each annotated by 3 independent human raters using Plutchik's 8 basic emotions and Valence-Arousal-Dominance (VAD) scales.

## Supported Tasks

- **Emotion Classification**: Predict the speaker's primary emotion from Plutchik's 8 categories (joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
- **VAD Regression**: Predict continuous Valence, Arousal, and Dominance ratings

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Scenario ID (1-60 within subtype) |
| `sd_situation` | str | Situational context (2-4 sentences) |
| `sd_utterance` | str | Speaker's pragmatically ambiguous utterance |
| `sd_speaker_role` | str | Speaker's role/relationship |
| `sd_listener_role` | str | Listener's role/relationship |
| `gold_standard` | str | Adjudicated ground truth emotion |
| `sl_plutchik_primary_*` | str | Per-annotator emotion label (3 per scenario) |
| `sl_v_*` | str | Per-annotator valence (7-point text scale) |
| `sl_a_*` | str | Per-annotator arousal (7-point text scale) |
| `sl_d_*` | str | Per-annotator dominance (7-point text scale) |
| `sl_confidence_*` | str | Per-annotator confidence rating |

### Data Splits

| Split | Count | Fraction |
|-------|-------|----------|
| Train | 211 | 70% |
| Validation | 48 | 15% |
| Test | 41 | 15% |

Splits are stratified by subtype and power relation (seed=42). See `reports/dmlr2026/splits.json`.

### Pragmatic Subtypes (60 scenarios each)

| Subtype | Description |
|---------|-------------|
| Sarcasm/Irony | Surface meaning contradicts intended meaning |
| Mixed Signals | Verbal and contextual cues conflict |
| Strategic Politeness | Face-saving language masks negative affect |
| Passive Aggression | Indirect hostility through apparent compliance |
| Deflection/Misdirection | Topic change to avoid addressing issues |

## Dataset Creation

### Annotation Process

- 15 undergraduate annotators (3 per subtype)
- Plutchik's 8 basic emotions + 7-point VAD scales
- 4-level QA pipeline (schema, statistical, agreement, expert adjudication)
- Gold standard via majority vote + expert override
- Inter-annotator agreement: Fleiss' kappa = 0.06-0.25 by subtype

### Source Data

All scenarios are researcher-authored (synthetic) to enable controlled variation across subtypes and power relations. No naturalistic data was collected.

## Considerations

### Limitations

- English only; pragmatic conventions vary across languages
- Synthetic scenarios may not capture full communication complexity
- Fair agreement (kappa=0.21) reflects genuine pragmatic ambiguity

### Ethical Considerations

- IRB-exempt (Category 2)
- All scenarios are synthetic and de-identified
- No sensitive personal information

## Citation

```bibtex
@article{chun2026cei,
  title={CEI: A Benchmark for Evaluating Pragmatic Reasoning in Language Models},
  author={Chun, Jon},
  journal={Journal of Data-centric Machine Learning Research},
  year={2026}
}
```

## License

- Data: CC-BY-4.0
- Code: MIT
