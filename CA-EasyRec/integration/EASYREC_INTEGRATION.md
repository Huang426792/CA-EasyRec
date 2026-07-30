# Integrating CA-EasyRec with HKUDS/EasyRec

This guide targets the inspected upstream commit:

```text
81e818b767689046c89edf5669dfbf64598db221
```

Upstream repository: <https://github.com/HKUDS/EasyRec>

CA-EasyRec does not redistribute the upstream source. The change is deliberately
isolated: keep EasyRec's RoBERTa encoder, MLM branch, profiles, evaluation, and
checkpoint handling, but replace the unweighted contrastive cross entropy with
the confidence-aware loss in this package.

## 1. Prepare both repositories

```bash
git clone https://github.com/HKUDS/EasyRec.git
git -C EasyRec checkout 81e818b767689046c89edf5669dfbf64598db221
git clone https://github.com/Huang426792/CA-EasyRec.git
python -m pip install -e ./CA-EasyRec
```

Download and unzip the official EasyRec data exactly as described upstream.
The source domains are `arts`, `games`, `movies`, `home`, `electronics`, and
`tools`.

## 2. Train the frozen source-domain teachers

From any directory:

```bash
ca-easyrec-teachers \
  --data-root /path/to/EasyRec/data \
  --domains arts games movies home electronics tools \
  --output /path/to/EasyRec/artifacts/ca_teachers.pt \
  --embedding-dim 64 \
  --num-layers 2 \
  --epochs 100 \
  --device cuda:0
```

The command reads each `trn_mat.pkl`, trains one LightGCN with BPR, and writes a
single frozen `TeacherEmbeddingBank`. It also writes
`ca_teachers.pt.history.json`.

Teacher selection and the final paper runs should use source validation data.
The command above is the training path; add checkpoint selection according to
the experimental protocol before reporting measured results.

## 3. Preserve IDs and domains in EasyRec's dataset

At upstream commit `81e818b`, `LazyPretrainEmbedderDataset` returns only three
token sequences. Confidence lookup also needs the original domain, user ID,
positive item ID, and sampled-negative item ID.

In `utility/load_data.py`, extend `_get_sentence_input()`:

```python
return {
    "user_profile": u_profile,
    "positive_item_profile": i_pos_profile,
    "negative_item_profile": i_neg_profile,
    "domain": _dataset,
    "user_id": u,
    "positive_item_id": i_pos,
    "negative_item_id": i_neg,
}
```

Pass those fields through `_preprocess()` and `__getitem__()`. In
`DataCollatorForPretrainEmbedderDataset.__call__`, preserve them alongside the
padded tensors:

```python
batch.update(
    user_domains=[instance["domain"] for instance in instances],
    user_ids=torch.tensor([instance["user_id"] for instance in instances]),
    positive_item_domains=[instance["domain"] for instance in instances],
    positive_item_ids=torch.tensor(
        [instance["positive_item_id"] for instance in instances]
    ),
    negative_item_domains=[instance["domain"] for instance in instances],
    negative_item_ids=torch.tensor(
        [instance["negative_item_id"] for instance in instances]
    ),
)
```

## 4. Load the teacher once

Add these arguments to upstream `ModelArguments`:

```python
teacher_bank_path: str = field(default="artifacts/ca_teachers.pt")
ca_epsilon: float = field(default=0.3)
ca_gamma: float = field(default=1.0)
```

In EasyRec's model initialization:

```python
from ca_easyrec.teacher_bank import TeacherEmbeddingBank

self.teacher_bank = TeacherEmbeddingBank.load(
    self.model_args.teacher_bank_path
)
```

The bank stores detached CPU tensors. `score_matrix()` moves only the small
batch score block to the current training device.

## 5. Replace the contrastive loss

Add these metadata arguments to `Easyrec.forward()`:

```python
user_domains=None,
user_ids=None,
positive_item_domains=None,
positive_item_ids=None,
negative_item_domains=None,
negative_item_ids=None,
```

After EasyRec has constructed `user_pooler_output`,
`pos_item_pooler_output`, and `neg_item_pooler_output`, replace the direct
`CrossEntropyLoss(cos_sim, labels)` block with:

```python
from ca_easyrec.training import ca_easyrec_batch_loss

ca_output = ca_easyrec_batch_loss(
    user_embeddings=user_pooler_output,
    positive_item_embeddings=pos_item_pooler_output,
    negative_item_embeddings=neg_item_pooler_output,
    user_domains=user_domains,
    user_ids=user_ids,
    positive_item_domains=positive_item_domains,
    positive_item_ids=positive_item_ids,
    negative_item_domains=negative_item_domains,
    negative_item_ids=negative_item_ids,
    teacher_bank=self.teacher_bank,
    temperature=self.model_args.temp,
    epsilon=self.model_args.ca_epsilon,
    gamma=self.model_args.ca_gamma,
)
cos_sim = ca_output.logits
loss = ca_output.loss
```

Keep the upstream MLM block unchanged:

```python
loss = loss + self.model_args.mlm_weight * masked_lm_loss
```

Candidate columns are ordered as:

```text
[all positive-item embeddings, all explicitly sampled negative-item embeddings]
```

For user row `r`, positive label `r` is forced to weight 1. Same-domain
negative positions use LightGCN confidence. Cross-domain positions use weight
1. A row with fewer than two eligible same-domain negatives also falls back to
unit weights.

Setting `ca_epsilon=1.0` is an exact loss-level ablation: every weight is one.

## 6. Distributed training requirement

The upstream model gathers text embeddings across workers. When distributed
training is enabled, gather the three ID tensors and numeric domain IDs in the
same worker/rank order before invoking `ca_easyrec_batch_loss`; otherwise the
metadata will not align with the gathered embeddings.

For the first correctness run, use one GPU. After its loss and weight audit
passes, add distributed metadata gathering and compare one-GPU versus
multi-GPU candidate ordering on a fixed batch.

## 7. Audit before the full experiment

For a fixed batch, log:

```python
print(ca_output.weights.min().item())
print(ca_output.weights.max().item())
print(ca_output.eligible_mask.sum().item())
```

Verify:

- all positive diagonal weights are exactly 1;
- every cross-domain candidate weight is exactly 1;
- high-affinity eligible negatives receive smaller weights;
- `ca_output.weights.requires_grad` is `False`;
- `ca_epsilon=1.0` reproduces the unweighted loss within floating-point
  tolerance.

Only after this audit should the five-seed Sports/Steam/Yelp experiment be
started.
