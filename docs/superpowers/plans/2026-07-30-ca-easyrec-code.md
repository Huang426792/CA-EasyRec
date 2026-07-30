# CA-EasyRec Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an original, upload-ready CA-EasyRec research repository with a frozen LightGCN teacher, confidence-aware contrastive loss, deterministic toy experiment, EasyRec integration guide, tests, and packaging metadata.

**Architecture:** Keep the new method in a small `src/ca_easyrec` package instead of copying upstream EasyRec. Implement auditable NumPy equations and differentiable PyTorch equivalents, train one LightGCN teacher per source domain, and demonstrate the full flow with a lightweight profile encoder. Pin the inspected upstream EasyRec commit in the integration guide.

**Tech Stack:** Python 3.10+, NumPy, PyTorch 2.2+, SciPy, standard-library `unittest`, setuptools.

## Global Constraints

- Do not vendor or claim authorship of HKUDS/EasyRec source code.
- Match the manuscript weighting equation exactly.
- Cross-domain candidates, positive labels, and rows with fewer than two eligible negatives use weight 1.
- Teacher-derived PyTorch weights must be detached from autograd.
- Toy metrics are smoke-test outputs, not paper results.
- Use deterministic seeds in tests and examples.

---

### Task 1: Confidence weighting and contrastive loss

**Files:**
- Create: `src/ca_easyrec/__init__.py`
- Create: `src/ca_easyrec/weighting.py`
- Create: `src/ca_easyrec/losses.py`
- Create: `tests/test_weighting.py`
- Create: `tests/test_losses.py`

**Interfaces:**
- Produces: `confidence_weights_numpy(scores, eligible_mask, epsilon, gamma, delta) -> np.ndarray`
- Produces: `confidence_weights_torch(scores, eligible_mask, epsilon, gamma, delta) -> torch.Tensor`
- Produces: `weighted_info_nce_numpy(logits, labels, weights) -> float`
- Produces: `weighted_info_nce_torch(logits, labels, weights) -> torch.Tensor`

- [ ] **Step 1: Write failing confidence-weight tests**

```python
class ConfidenceWeightTests(unittest.TestCase):
    def test_high_affinity_receives_lower_weight(self):
        scores = np.array([[0.0, 1.0, 3.0]], dtype=np.float64)
        mask = np.array([[True, True, True]])
        got = confidence_weights_numpy(scores, mask, epsilon=0.2, gamma=1.0)
        self.assertGreater(got[0, 0], got[0, 2])

    def test_ineligible_and_small_rows_keep_unit_weight(self):
        scores = np.array([[1.0, 4.0], [2.0, 3.0]])
        mask = np.array([[True, False], [False, False]])
        np.testing.assert_allclose(
            confidence_weights_numpy(scores, mask, 0.2, 1.0),
            np.ones_like(scores),
        )
```

- [ ] **Step 2: Run tests and verify missing-module failure**

Run: `python -m unittest tests.test_weighting -v`

Expected: import failure because `ca_easyrec.weighting` does not exist.

- [ ] **Step 3: Implement validated row-wise calibration**

Implement shape checks, parameter checks, row-wise masked mean/standard
deviation, sigmoid confidence, the manuscript equation, and the unit-weight
fallback. Implement the PyTorch version with the same branches and return
`weights.detach()`.

- [ ] **Step 4: Run weighting tests**

Run: `python -m unittest tests.test_weighting -v`

Expected: all weighting tests pass.

- [ ] **Step 5: Write failing weighted-loss tests**

```python
class WeightedLossTests(unittest.TestCase):
    def test_matches_hand_computed_denominator(self):
        logits = np.array([[2.0, 1.0, 0.0]])
        labels = np.array([0])
        weights = np.array([[1.0, 0.5, 1.0]])
        expected = -math.log(
            math.exp(2.0) /
            (math.exp(2.0) + 0.5 * math.exp(1.0) + math.exp(0.0))
        )
        self.assertAlmostEqual(
            weighted_info_nce_numpy(logits, labels, weights),
            expected,
            places=12,
        )
```

- [ ] **Step 6: Run the loss test and verify failure**

Run: `python -m unittest tests.test_losses -v`

Expected: import failure because `ca_easyrec.losses` does not exist.

- [ ] **Step 7: Implement stable weighted InfoNCE**

Set positive weights to one, add `log(weight)` to negative logits, and use a
log-sum-exp calculation. The PyTorch implementation uses
`torch.nn.functional.cross_entropy`.

- [ ] **Step 8: Run core tests and commit**

Run: `python -m unittest tests.test_weighting tests.test_losses -v`

Expected: all tests pass.

Commit: `git commit -am "feat: add confidence-aware contrastive objective"`

---

### Task 2: LightGCN teacher and embedding bank

**Files:**
- Create: `src/ca_easyrec/lightgcn.py`
- Create: `src/ca_easyrec/teacher_bank.py`
- Create: `tests/test_lightgcn.py`
- Create: `tests/test_teacher_bank.py`

**Interfaces:**
- Produces: `LightGCN(num_users, num_items, embedding_dim, num_layers)`
- Produces: `train_lightgcn(edges, num_users, num_items, config) -> (LightGCN, list[float])`
- Produces: `TeacherEmbeddingBank.add_domain(name, user_embeddings, item_embeddings)`
- Produces: `TeacherEmbeddingBank.score(user_domains, user_ids, item_domains, item_ids) -> Tensor`
- Produces: `TeacherEmbeddingBank.save(path)` and `.load(path)`

- [ ] **Step 1: Write failing graph and bank tests**

Use a four-edge graph to assert final embedding shapes, finite BPR loss,
cross-domain scores equal zero with an invalidity mask, same-domain scores
equal hand-computed dot products, and save/load round-trip equality.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_lightgcn tests.test_teacher_bank -v`

Expected: missing-module failures.

- [ ] **Step 3: Implement normalized LightGCN propagation**

Build the symmetric user-item adjacency, calculate
`D^-1/2 A D^-1/2`, propagate embeddings for `num_layers`, and average the
initial and propagated embeddings. Implement BPR
`-log(sigmoid(s_pos - s_neg))` plus L2 regularization.

- [ ] **Step 4: Implement domain-separated frozen embedding storage**

Validate two-dimensional tensors and IDs. Score only pairs whose user and item
domains match an installed domain. Save CPU tensors plus a format version and
load with explicit validation.

- [ ] **Step 5: Run graph tests and commit**

Run: `python -m unittest tests.test_lightgcn tests.test_teacher_bank -v`

Expected: all tests pass.

Commit: `git commit -am "feat: add frozen LightGCN teachers"`

---

### Task 3: Data loading, text model, and metrics

**Files:**
- Create: `src/ca_easyrec/data.py`
- Create: `src/ca_easyrec/text_model.py`
- Create: `src/ca_easyrec/metrics.py`
- Create: `tests/test_data.py`
- Create: `tests/test_metrics.py`
- Create: `tests/fixtures/tiny/`

**Interfaces:**
- Consumes: `interactions.csv` columns `domain,user_id,item_id,split`
- Consumes: JSONL profiles with `domain`, entity ID, and `profile`
- Produces: `ResearchDataset`
- Produces: `load_easyrec_domain(path, domain) -> ResearchDomain`
- Produces: `HashingProfileEncoder(vocabulary_size, embedding_dim)`
- Produces: `recall_ndcg_at_k(scores, truth, seen, k) -> dict[str, float]`

- [ ] **Step 1: Write failing fixture-based data and metric tests**

Use literal fixtures to verify stable contiguous ID maps, rejection of unknown
profile IDs, masking of seen items, Recall@2, and NDCG@2 against hand-computed
values.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_data tests.test_metrics -v`

Expected: missing-module failures.

- [ ] **Step 3: Implement the compact format and official-data adapter**

Read CSV/JSONL with deterministic sorting. For the official adapter, load
`trn_mat.pkl` and newline-delimited profile JSON using the same contiguous IDs
as the matrices.

- [ ] **Step 4: Implement lightweight profile encoder and metrics**

Tokenize lowercase text with a stable SHA-256 bucket mapping, average trainable
token embeddings, normalize profile vectors, mask seen items with `-inf`, and
calculate macro Recall/NDCG.

- [ ] **Step 5: Run data/metric tests and commit**

Run: `python -m unittest tests.test_data tests.test_metrics -v`

Expected: all tests pass.

Commit: `git commit -am "feat: add profile data and ranking evaluation"`

---

### Task 4: End-to-end toy experiment

**Files:**
- Create: `src/ca_easyrec/demo.py`
- Create: `src/ca_easyrec/training.py`
- Create: `tests/test_demo.py`
- Create: `scripts/run_demo.sh`
- Create: `scripts/run_demo.ps1`

**Interfaces:**
- Produces: `python -m ca_easyrec.demo --output <dir> --seed 2026`
- Produces: `<dir>/teacher.pt`, `<dir>/text_model.pt`, and
  `<dir>/metrics.json`

- [ ] **Step 1: Write failing end-to-end artifact test**

Run the demo with minimal epochs in a temporary directory and assert that all
three artifacts exist, metrics JSON contains `recall@3` and `ndcg@3`, and all
values are finite in `[0, 1]`.

- [ ] **Step 2: Run the demo test and verify failure**

Run: `python -m unittest tests.test_demo -v`

Expected: missing-module failure.

- [ ] **Step 3: Implement deterministic teacher and text training**

Generate two toy domains, train separate LightGCN teachers, freeze their
embeddings, create EasyRec-style batches with explicit negatives, calculate
same-domain eligibility, train the profile encoder with CA loss, and evaluate
held-out interactions.

- [ ] **Step 4: Run demo tests and a fresh CLI smoke test**

Run:

```bash
python -m unittest tests.test_demo -v
python -m ca_easyrec.demo --output artifacts/demo --seed 2026 --teacher-epochs 5 --text-epochs 5
```

Expected: tests pass and three artifacts are written.

- [ ] **Step 5: Commit**

Commit: `git commit -am "feat: add reproducible CA-EasyRec demo"`

---

### Task 5: EasyRec integration and repository documentation

**Files:**
- Create: `integration/EASYREC_INTEGRATION.md`
- Create: `README.md`
- Create: `README_zh.md`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `LICENSE`
- Create: `CITATION.cff`
- Create: `.gitignore`
- Create: `paper/Di_Liu_NLP_Assignment.pdf`
- Create: `paper/source/`

**Interfaces:**
- Documents: official EasyRec commit
  `81e818b767689046c89edf5669dfbf64598db221`
- Documents: installation, demo, official-data experiment, interpretation,
  limitations, and GitHub upload commands.

- [ ] **Step 1: Add package and repository metadata**

Define a setuptools `src` package, Python `>=3.10`, runtime dependencies on
NumPy/SciPy/PyTorch, optional EasyRec dependencies matching Transformers
`4.40.0`, and MIT licensing for this repository's original code.

- [ ] **Step 2: Write upstream integration guide**

Explain the metadata additions to EasyRec's dataset/collator/model forward
call, the candidate ordering `[positive_batch, sampled_negative_batch]`, the
positive diagonal, same-domain mask, teacher score lookup, and the call to
`weighted_info_nce_torch`. Explicitly distinguish upstream and original code.

- [ ] **Step 3: Write English and Chinese READMEs**

Include quick start, repository tree, equations, data format, real experiment
steps, expected artifacts, citation, paper link, and the statement that no
large-scale result is included until measured.

- [ ] **Step 4: Add the paper artifacts**

Copy the validated assignment PDF and editable LaTeX sources into `paper/`.

- [ ] **Step 5: Build package and commit**

Run: `python -m build`

Expected: wheel and source distribution are created.

Commit: `git commit -am "docs: prepare reproducible open-source release"`

---

### Task 6: Final verification and archive

**Files:**
- Create: `CA-EasyRec-upload.zip` outside the repository

- [ ] **Step 1: Run all tests**

Run: `python -m unittest discover -s tests -v`

Expected: zero failures and zero errors.

- [ ] **Step 2: Run static and package checks**

Run:

```bash
python -m compileall -q src tests
python -m build
python -m pip install --force-reinstall --no-deps dist/*.whl
```

Expected: every command exits with status 0.

- [ ] **Step 3: Run a clean toy experiment**

Run:

```bash
python -m ca_easyrec.demo --output artifacts/final-smoke --seed 2026 --teacher-epochs 5 --text-epochs 5
```

Expected: `teacher.pt`, `text_model.pt`, and valid `metrics.json`.

- [ ] **Step 4: Inspect repository status and archive tracked files**

Run:

```bash
git status --short
git archive --format=zip --output=../CA-EasyRec-upload.zip HEAD
unzip -t ../CA-EasyRec-upload.zip
```

Expected: clean status and a valid ZIP.
