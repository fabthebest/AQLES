---
license: mit
task_categories:
  - text-classification
language:
  - en
tags:
  - quality-assessment
  - probing
  - interpretability
  - negativity-bias
  - evaluative-language
size_categories:
  - 1K<n<10K
---

# AQLES Quality Lexicon

A controlled lexicon of 200 English quality-evaluative words, uniformly distributed across 5 quality tiers (40 words per tier), designed for probing transformer hidden states.

## What this is

This dataset was built for [AQLES](https://github.com/fabthebest/aqles), a probing framework that tests whether transformers encode evaluative quality judgments in their hidden states. Each word has a continuous quality score (0.01–1.00) calibrated against the NRC VAD lexicon, and a categorical tier assignment.

## Tiers

| Tier | Label | Score Range | Example Words |
|---|---|---|---|
| 4 | Exceptional | ≥ 0.90 | masterful, pristine, sublime |
| 3 | Excellent | 0.78–0.89 | stellar, outstanding, superb |
| 2 | Good | 0.45–0.77 | competent, adequate, reliable |
| 1 | Mediocre | 0.15–0.44 | lackluster, amateurish, flawed |
| 0 | Terrible | < 0.15 | abysmal, loathsome, vile |

## Structure

Each row is one probing sentence (200 words × 10 templates = 2,000 rows):

- `word`: the quality-evaluative word
- `quality_score`: continuous score (float, 0.01–1.00)
- `tier`: categorical tier (int, 0–4)
- `template_id`: which of the 10 sentence templates (int, 0–9)
- `sentence`: the complete probing sentence

## Key property

Template identity contributes negligibly to probe score variance (η² = 0.002%). Word identity explains 98.4% of variance. The sentential frame is a controlled vehicle, not a confound.

## Citation

```bibtex
@misc{filsaime2026aqles,
  author = {Fils-Aimé, Fabrice},
  title = {AQLES: Probing Transformer Hidden States to Decode Quality Ranking Geometry},
  year = {2026},
  url = {https://github.com/fabthebest/aqles}
}
```
