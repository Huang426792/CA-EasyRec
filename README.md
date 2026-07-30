# CA-EasyRec

Confidence-Aware EasyRec reduces the influence of likely false negatives in
contrastive language-model training for personalized recommendation.

[中文说明](README_zh.md) ·
[EasyRec integration](integration/EASYREC_INTEGRATION.md) ·
[GitHub upload guide](UPLOAD_TO_GITHUB_zh.md)

> **Result status:** this repository contains tested method code and a
> deterministic toy smoke experiment. It does **not** claim unmeasured Sports,
> Steam, or Yelp results.

## Method

EasyRec treats other batch items as negatives. In sparse implicit feedback,
some unobserved items may actually fit the user's interests. CA-EasyRec trains
one LightGCN teacher per source domain and converts each same-domain teacher
affinity into:

```text
confidence = sigmoid((q - row_mean) / (row_std + delta))
weight = epsilon + (1 - epsilon) * (1 - confidence) ** gamma
```

A negative denominator term becomes:

```text
weight * exp(text_similarity / temperature)
```

High-affinity negatives receive smaller weights. Positive labels,
cross-domain candidates, and rows with fewer than two eligible negatives keep
weight 1. Teacher scores are detached, and no teacher is used at target-domain
inference.

## What is implemented

- auditable NumPy and differentiable PyTorch confidence weighting;
- weighted InfoNCE with stable log-sum-exp;
- LightGCN propagation, BPR training, and one frozen teacher per domain;
- strict cross-domain score isolation;
- compact CSV/JSONL and official EasyRec data readers;
- EasyRec-style batch assembly;
- deterministic end-to-end toy experiment;
- Recall@K and NDCG@K evaluation;
- official EasyRec integration guide pinned to a known upstream commit;
- unit tests and GitHub Actions.

## Quick start

Python 3.10 or newer is required.

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
python -m pip install numpy scipy
python -m ca_easyrec.demo \
  --output artifacts/demo \
  --teacher-epochs 5 \
  --text-epochs 5
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
python -m pip install numpy scipy
python -m ca_easyrec.demo --output artifacts/demo --teacher-epochs 5 --text-epochs 5
```

The command writes:

```text
artifacts/demo/
├── teacher.pt
├── text_model.pt
└── metrics.json
```

`metrics.json` explicitly labels its numbers as `toy_smoke_test`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Compact data format

For independent experiments, a directory may contain:

```text
interactions.csv
user_profiles.jsonl
item_profiles.jsonl
```

`interactions.csv`:

```csv
domain,user_id,item_id,split
books,u1,i4,train
books,u1,i9,test
```

User profile JSONL:

```json
{"domain":"books","user_id":"u1","profile":"likes space fiction"}
```

Item profile JSONL:

```json
{"domain":"books","item_id":"i4","profile":"a space exploration novel"}
```

## Repository layout

```text
src/ca_easyrec/
  weighting.py       confidence calibration
  losses.py          weighted InfoNCE
  lightgcn.py        collaborative teacher
  teacher_bank.py    domain-isolated frozen embeddings
  data.py            compact and EasyRec data readers
  text_model.py      lightweight offline demo encoder
  training.py        EasyRec-style CA batch objective
  official.py        source-teacher training command
  demo.py            end-to-end smoke experiment
integration/         upstream EasyRec integration
tests/               unit and end-to-end tests
```

## Relationship to EasyRec

This is an independent course-project extension and is not an official
HKUDS/EasyRec release. Upstream source is not vendored. Cite the EasyRec paper
when using its model, data, profiles, or evaluation protocol.

## Citation

```bibtex
@software{liu2026caeasyrec,
  author  = {Di Liu},
  title   = {CA-EasyRec: Confidence-Aware EasyRec},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Huang426792/CA-EasyRec}
}
```

## License

Original code in this repository is released under the [MIT License](LICENSE).
