# Prompt Templates for CEI Benchmark Evaluation

This directory contains standardized prompt templates for evaluating language models on the CEI Benchmark.

## Overview

The CEI Benchmark evaluates models' ability to infer the **speaker's emotional state** from pragmatically ambiguous utterances. Models receive scenario context and must predict the speaker's primary emotion.

## Files

| File | Description |
|------|-------------|
| `zero_shot.txt` | Zero-shot evaluation prompt |
| `chain_of_thought.txt` | Chain-of-thought reasoning prompt |
| `few_shot_examples.json` | Few-shot examples (2 per subtype) |

## Usage

### Python Example

```python
import csv
from pathlib import Path

# Load prompt template
with open("prompts/zero_shot.txt") as f:
    template = f.read()

# Load scenario from dataset CSV
with open("data/human-gold/data_sarcasm-irony.csv") as f:
    reader = csv.DictReader(f)
    row = next(reader)

# Format prompt
prompt = template.format(
    context=row["sd_situation"],
    speaker_role=row["sd_speaker_role"],
    listener_role=row["sd_listener_role"],
    utterance=row["sd_utterance"],
)

# Send to model API and parse response
# response = model.generate(prompt)
```

## Evaluation Protocol

### Ground Truth

Ground truth labels are established via majority vote among 3 annotators. For scenarios with 3-way splits (~31%), expert adjudication determined the label.

### Metrics

- **Accuracy**: Exact match with ground truth emotion
- **Macro-F1**: F1 averaged across 8 emotion classes
- **Stratified Analysis**: Report separately by pragmatic subtype

### Valid Emotion Labels

Models must output exactly one of these 8 Plutchik emotions:
- joy
- trust
- fear
- surprise
- sadness
- disgust
- anger
- anticipation

### Stratification

Results should be reported for:
1. **Overall** accuracy/F1
2. **By subtype**: sarcasm-irony, mixed-signals, strategic-politeness, passive-aggression, deflection-misdirection
3. **By power relation** (optional): peer, high-to-low, low-to-high
