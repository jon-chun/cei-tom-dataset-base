# PII Audit & Postprocessing Log

This document records all personally identifying information (PII) found during preparation of the anonymous DMLR 2026 submission repo, and the actions taken to remove or redact it.

---

## 1. CSV Header Anonymization

All 15 annotator names appearing in CSV column headers were replaced with anonymous identifiers.

### Mapping

| File | Original Name | Replacement |
|------|--------------|-------------|
| `data_sarcasm-irony.csv` | Hannah | Annotator_1 |
| `data_sarcasm-irony.csv` | Andre | Annotator_2 |
| `data_sarcasm-irony.csv` | Gwen | Annotator_3 |
| `data_mixed-signals.csv` | Abhigya | Annotator_4 |
| `data_mixed-signals.csv` | Anna | Annotator_5 |
| `data_mixed-signals.csv` | Peter | Annotator_6 |
| `data_passive-aggression.csv` | Adrian | Annotator_7 |
| `data_passive-aggression.csv` | Eli | Annotator_8 |
| `data_passive-aggression.csv` | Godwin | Annotator_9 |
| `data_deflection-misdirection.csv` | Ann | Annotator_10 |
| `data_deflection-misdirection.csv` | Kirill | Annotator_11 |
| `data_deflection-misdirection.csv` | Tiffanie | Annotator_12 |
| `data_strategic-politeness.csv` | Morty | Annotator_13 |
| `data_strategic-politeness.csv` | Mous | Annotator_14 |
| `data_strategic-politeness.csv` | Wisdom | Annotator_15 |

### Columns affected per file (5 columns per annotator, 3 annotators = 15 columns)

- `sl_plutchik_primary_<Name>`
- `sl_v_<Name>`
- `sl_a_<Name>`
- `sl_d_<Name>`
- `sl_confidence_<Name>`

### Data cell verification

No PII found in CSV data cells (only in headers).

---

## 2. Paper .tex Redactions

File: `papers/dmlr2026/cei-tom_dataset.tex`

| Location | PII Found | Action |
|----------|-----------|--------|
| Line 20 (`\dmlrheading`) | `Chun \textit{et al.}` | Replaced with `Anonymous` |
| Line 23 (`\ShortHeadings`) | `Chun \textit{et al.}` | Replaced with `Anonymous` |
| Lines 30-46 (`\author` block) | 16 author names, emails, institution | Replaced with `\author{\name Anonymous Authors}` |
| Line 149 (annotation section) | `Kenyon College`, `IPHS 391` | Replaced with `a US liberal arts college`, removed course code |
| Line 313 (maintenance) | `chunj@kenyon.edu` | Replaced with `[redacted for blind review]` |
| Line 322 (conclusion URL) | `https://github.com/jon-chun/cei-benchmark` | Replaced with `<anonymous>` |
| Lines 329-331 (acknowledgments) | All 15 annotator names + institution | Replaced with `[Redacted for blind review]` |
| Line 381 (datasheet URL) | `https://github.com/jon-chun/cei-benchmark` | Replaced with `<anonymous>` |
| Line 387 (datasheet contact) | `chunj@kenyon.edu` | Replaced with `[redacted for blind review]` |

---

## 3. Path Fixes

| File | Issue | Fix |
|------|-------|-----|
| `papers/dmlr2026/generate_figures.py` | Hardcoded `/Users/jonc/code/cei-tom-dataset` and `/Users/jonc/code/cei-tom-benchmark` | Replaced with `Path(__file__).resolve().parent.parent.parent` (repo-relative) |
| `pyproject.toml` | Created fresh (no author names/emails/URLs) | Minimal version with project name + pyyaml dependency only |

---

## 4. Files NOT Copied (excluded from anonymous repo)

- `src/` (full pipeline source code)
- `tests/` (test suite)
- `Makefile`
- `docs/annotation/` (annotation guidelines)
- `data-human/` (raw Label Studio exports with annotator filenames)
- CogSci 2026 paper materials
- ICML/FAccT 2026 code
- `README-DMLR.md` (replaced by anonymized `README.md`)

---

## 5. Verification

Verification commands run after anonymization:

```bash
# No PII in any file
grep -ri "kenyon\|chunj\|jon chun" .  # returns nothing

# No annotator names in CSV headers
head -1 data/human-gold/*.csv  # shows only Annotator_N

# Paper .tex has no author names/emails
grep -i "chun\|sussman\|kocaman\|sidorko\|koirala\|mangine\|mccloud\|eisenbeis\|akanwe\|gassama\|gonzalez\|enright\|dunson\|tiffanie\|rosenstiel\|idowu" papers/dmlr2026/cei-tom_dataset.tex  # returns nothing

# Pipeline runs successfully with anonymized headers
python scripts/run_pipeline_dmlr2026.py --stage all_local
```
