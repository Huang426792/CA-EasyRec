# CA-EasyRec Code Design

## Goal

Provide an original, upload-ready research repository for the method described
in the ACL manuscript: a frozen LightGCN teacher estimates collaborative
affinity, and a confidence-aware contrastive loss reduces the repulsive force
of likely false negatives during EasyRec-style text encoder training.

## Chosen approach

The repository will use a standalone core package plus an upstream integration
guide.

- A full EasyRec fork would be convenient, but it would duplicate unrelated
  upstream code and make it difficult to distinguish the assignment's
  contribution.
- A patch file alone would be small, but it would not be independently
  testable and would be fragile when upstream files change.
- The selected approach implements the new method as an isolated package,
  includes a deterministic toy experiment, and documents the small adapter
  required by upstream EasyRec commit
  `81e818b767689046c89edf5669dfbf64598db221`.

## Architecture

### Confidence weighting

`ca_easyrec.weighting` accepts a teacher-affinity matrix and a mask identifying
same-domain negatives. For each anchor with at least two eligible negatives, it
standardizes the eligible affinities, applies a sigmoid, and computes

`w = epsilon + (1 - epsilon) * (1 - confidence) ** gamma`.

Cross-domain candidates, positive labels, and rows with fewer than two
same-domain negatives retain unit weight. The PyTorch implementation detaches
teacher-derived weights from autograd.

### Contrastive objective

`ca_easyrec.losses` implements the EasyRec-style user-to-item objective. User
embeddings are compared with the batch's positive items and explicitly sampled
negative items. Multiplying a negative term in the denominator by `w` is
implemented as adding `log(w)` to that negative logit before cross entropy.
The positive logit is never reweighted.

The package exposes both NumPy reference functions and PyTorch training
functions. The reference functions make the equations directly auditable,
while the PyTorch path supports training.

### LightGCN teacher

`ca_easyrec.lightgcn` implements normalized bipartite propagation, BPR training,
negative sampling, and frozen embedding export. A `TeacherEmbeddingBank` stores
separate user and item embeddings for every domain and refuses to score
cross-domain pairs.

### Data and demo

`ca_easyrec.data` defines a compact CSV/JSONL research format and a reader for
the official EasyRec `trn_mat.pkl`, `user_profile.json`, and
`item_profile.json` layout. A deterministic toy generator creates two domains
with sparse interactions and profile text.

`python -m ca_easyrec.demo` trains the teacher and a lightweight text encoder,
prints Recall/NDCG metrics, and writes checkpoints. It is a smoke test of the
complete data-to-model flow, not a substitute for the paper's large-scale
experiments.

### Upstream EasyRec integration

`integration/EASYREC_INTEGRATION.md` pins the inspected upstream commit and
shows the exact metadata fields and loss call to add. It does not vendor
upstream source. The existing RoBERTa encoder, MLM loss, distributed gathering,
and zero-shot evaluator remain upstream responsibilities.

## Error handling

- Validate `epsilon`, `gamma`, temperature, tensor shapes, IDs, and domains.
- Reject non-contiguous LightGCN IDs with an actionable message.
- Refuse to load teacher files with missing metadata or incompatible shapes.
- Use deterministic random generators when a seed is supplied.
- Never silently assign a cross-domain teacher score.

## Testing

Tests cover:

- high teacher affinity produces a lower negative weight;
- cross-domain candidates and positives retain unit weight;
- fewer than two same-domain negatives fall back to unit weights;
- weighted InfoNCE matches a manually computed denominator;
- PyTorch and NumPy implementations agree when PyTorch is installed;
- LightGCN output shapes and frozen export/load round-trip;
- Recall/NDCG calculations;
- the toy command writes the expected artifacts.

The final verification includes the full unit test suite, Python bytecode
compilation, a clean-package build, and a fresh toy run.

## Scope boundary

The repository contains no fabricated Sports, Steam, or Yelp results. Full
paper numbers require downloading the official EasyRec data and running the
documented multi-seed experiment. The toy metrics are labeled as smoke-test
outputs only.
