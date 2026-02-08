# Datasheet for CEI Benchmark

*Following the framework of [Gebru et al. (2021)](https://arxiv.org/abs/1803.09010)*

## Motivation

### For what purpose was the dataset created?

The Contextual Emotional Inference (CEI) Benchmark was created to evaluate pragmatic reasoning capabilities in large language models. Specifically, it tests models' ability to infer the speaker's emotional state from pragmatically ambiguous utterances---situations where intended meaning diverges from literal semantics.

### Who created the dataset and on behalf of which entity?

The dataset was created by an academic research team. The annotation was performed by trained undergraduate annotators recruited through university channels.

### Who funded the creation of the dataset?

University research funds supported the dataset creation and annotation process.

## Composition

### What do the instances represent?

Each instance is a **scenario** consisting of:
- A situational context (2–4 sentences establishing setting and background)
- Speaker and listener roles with explicit power relations
- An ambiguous utterance requiring pragmatic interpretation
- Three independent annotations of the speaker's emotional state

### How many instances are there?

| Attribute | Value |
|-----------|-------|
| Total scenarios | 300 |
| Annotations per scenario | 3 |
| Total annotations | 900 |
| Pragmatic subtypes | 5 |
| Scenarios per subtype | 60 |
| Power relations | 3 |
| Social contexts | 4 (workplace, family, social, service) |

### What data does each instance consist of?

Each scenario contains:

**Scenario fields:**
- `id`: Unique identifier
- `sd_situation`: Situational context (2–4 sentences)
- `sd_utterance`: The speaker's pragmatically ambiguous statement
- `sd_speaker_role`: The speaker's social role
- `sd_listener_role`: The listener's social role
**Annotation fields (per annotator, suffixed with `_Annotator_N`):**
- `sl_plutchik_primary`: Speaker's primary emotion (8 Plutchik categories)
- `sl_v`: Valence rating (7-point text scale)
- `sl_a`: Arousal rating (7-point text scale)
- `sl_d`: Dominance rating (7-point text scale)
- `sl_confidence`: Annotator confidence (7-point text scale)

### Is there a label or target associated with each instance?

Yes. The target is the **speaker's emotional state**, encoded as:
1. Primary emotion from Plutchik's 8 basic emotions
2. VAD (Valence-Arousal-Dominance) dimensional ratings

Ground truth is established via majority vote among 3 annotators. For 3-way splits (31.3% of dataset), expert adjudication determined the label.

### Is any information missing from individual instances?

No. All 300 scenarios have complete annotations from all 3 assigned annotators.

### Are there any errors, sources of noise, or redundancies?

The documented inter-annotator agreement (Fleiss' κ = 0.21, fair) reflects the **genuine ambiguity** inherent in pragmatic inference, not annotation noise. This is a deliberate feature of the benchmark—pragmatically ambiguous utterances naturally admit multiple valid interpretations.

Quality control measures applied:
- 4-level automated QA pipeline
- 15.7% of scenarios received expert adjudication
- Timing outliers and straight-lining patterns flagged and reviewed

### Does the dataset contain data that might be considered confidential?

No. All scenarios are synthetic (researcher-authored), containing no personally identifiable information or confidential content.

### Does the dataset contain data that might be considered offensive or inappropriate?

The dataset contains scenarios depicting interpersonal conflict situations (sarcasm, passive aggression, etc.). These are synthetic examples designed for research purposes and do not contain hate speech, explicit content, or content targeting protected groups.

## Collection Process

### How was the data associated with each instance acquired?

**Scenarios**: Template-based synthesis by research team members, followed by expert curation for naturalness and pragmatic validity.

**Annotations**: Crowdsourced via trained university annotators using Label Studio annotation platform.

### Who was involved in the data collection process?

- Research team members: Scenario authoring and curation
- 15 trained undergraduate annotators: Emotion and VAD annotations
- Expert meta-annotator: Quality review and adjudication

### Over what timeframe was the data collected?

Annotations were collected in December 2025.

### Were any ethical review processes conducted?

The annotation study was reviewed and approved by the university IRB. Annotators provided informed consent and were compensated at university standard rates.

## Preprocessing/Cleaning/Labeling

### Was any preprocessing/cleaning/labeling of the data done?

Yes. A 4-level quality control pipeline was applied:

1. **Schema Validation**: JSON structure, required fields, enum value validation
2. **Statistical Consistency**: Timing outlier detection (MAD-based), straight-lining detection, self-contradiction checks
3. **Inter-Annotator Agreement**: Fleiss' κ computation with bootstrap confidence intervals
4. **Expert Adjudication**: Human review of flagged scenarios (15.7% of total)

### Is the software used to preprocess/clean/label the data available?

Yes. The analysis pipeline is included in this repository as `scripts/run_pipeline_dmlr2026.py`. See [README.md](README.md) for usage instructions.

## Uses

### What tasks has the dataset been used for?

The dataset is designed for evaluating language models on:
- Pragmatic disambiguation
- Contextual emotional inference
- Social reasoning in ambiguous communication

### Is there anything about the composition or collection that might impact future uses?

1. **English only**: Pragmatic conventions vary across languages and cultures
2. **Synthetic scenarios**: May not capture full complexity of naturalistic communication
3. **Power relation imbalance**: 72% peer-to-peer interactions limits power-stratified analysis
4. **Low agreement is informative**: The κ = 0.21 (fair) reflects genuine ambiguity, not noise

### Are there tasks for which the dataset should not be used?

- High-stakes decision-making without human oversight
- Surveillance or manipulation of vulnerable populations
- Training emotion recognition systems for deceptive purposes

## Distribution

### How will the dataset be distributed?

The dataset is distributed via:
- Anonymous review repository during peer review
- Public GitHub repository post-publication
- Accompanying the DMLR 2026 paper

### When will the dataset be released?

The dataset is available now for peer review and will be publicly released upon paper acceptance.

### Will the dataset be distributed under a copyright or intellectual property license?

- **Data**: CC-BY-4.0 (Creative Commons Attribution 4.0 International)
- **Code**: MIT License

### Have any third parties imposed restrictions on the data?

No.

## Maintenance

### Who will be maintaining the dataset?

The research team will maintain the dataset for at least 5 years post-publication.

### How can the maintainer be contacted?

See contact information in the published paper.

### Will the dataset be updated?

Error corrections will be documented in [CHANGELOG.md](CHANGELOG.md). The dataset follows semantic versioning (current: v1.0.0).

### Will older versions of the dataset continue to be available?

Yes. All versions will remain available via Git tags and releases.

### How will updates be communicated?

- CHANGELOG.md in the repository
- GitHub releases
- Updates to the paper's supplementary materials where applicable

## Additional Documentation

- [README.md](README.md): Installation, usage, and dataset overview
- [CHANGELOG.md](CHANGELOG.md): Version history and changes
- [LICENSE](LICENSE): Full license text
- [config/config-dmlr.yml](config/config-dmlr.yml): Pipeline configuration
- Paper Appendix B: Complete annotation guidelines
