# Changelog

All notable changes to the CEI Benchmark dataset will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-02

### Added
- Initial release of CEI Benchmark dataset
- 300 scenarios across 5 pragmatic subtypes
- 900 annotations (3 per scenario) from 15 trained annotators
- 4-level quality control pipeline (`scripts/run_pipeline_dmlr2026.py`)
- Pipeline outputs (`reports/dmlr2026/`)
- Figure generation (`scripts/generate_figures.py`)
- Complete annotation guidelines (paper Appendix B)
- Datasheet documentation (`DATASHEET.md`)
- Prompt templates for model evaluation (`prompts/`)

### Dataset Statistics
- **Scenarios**: 300 total (60 per subtype)
- **Annotations**: 900 total (3 per scenario)
- **Pragmatic Subtypes**: sarcasm-irony, mixed-signals, strategic-politeness, passive-aggression, deflection-misdirection
- **Inter-Annotator Agreement**: Fleiss' κ = 0.21 (overall), ranging from 0.06 to 0.25 by subtype
- **Expert Adjudication Rate**: 15.7% of scenarios

## Versioning Policy

This dataset follows semantic versioning:

- **Major version** (X.0.0): Changes that alter ground truth labels or remove scenarios
- **Minor version** (0.X.0): Addition of new scenarios, annotations, or metadata fields
- **Patch version** (0.0.X): Documentation updates, error corrections, code fixes

## Maintenance

This dataset will be maintained for at least 5 years post-publication (until 2031).

Error reports and corrections should be submitted via GitHub Issues.
