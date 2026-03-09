# KV Cache Compaction — Adjacent Concepts & Cross-Pollination Map

A discovery-driven exploration of techniques from adjacent fields that map
onto specific components of the KV cache compaction pipeline. Each concept
includes: what it is, the structural analogy, the concrete improvement it
suggests, and relevant papers.

---

## How to Read This Document

The compaction pipeline has four components. Each adjacent concept maps to
one or more:

```
[A] Key Selection    — choosing which t keys to keep from T
[B] Bias Fitting     — finding beta so attention mass is preserved
[C] Value Fitting    — finding C_v so attention output is preserved
[D] Budget Alloc     — distributing budget across heads/layers
```

---

## 1. CUR Decomposition & Interpolative Decomposition

**Maps to: [A] Key Selection + [C] Value Fitting**

### What it is

CUR decomposition factorizes a matrix A ≈ C U R where C is a subset of
columns of A, R is a subset of rows, and U is a small linking matrix. The
Interpolative Decomposition (ID) is the related factorization A ≈ C D where
D is an interpolation matrix containing the identity.

### The structural analogy

The attention output matrix is softmax(Q K^T / sqrt(d)) @ V. This is
*exactly* a matrix that CUR was designed to approximate:
- C = selected key columns (key selection)
- R = corresponding value rows (value selection)
- U = the linking/interpolation matrix (plays the role of beta + C_v fitting)

The compaction pipeline's three steps are literally a CUR decomposition of
the attention output matrix, with the added constraint that the "C" selection
must work through the softmax nonlinearity.

### What it suggests concretely

**Leverage score sampling** for key selection. Instead of max-attention
scoring, compute leverage scores of the attention matrix (proportional to
diagonal of the projection onto the top-k singular subspace). Leverage scores
capture *structural importance* — how much removing a column degrades the
low-rank approximation — rather than just *magnitude*.

### Direct application exists

**CurDKV** (Shi et al., NeurIPS 2025) already applies CUR decomposition to
KV cache compression. Key finding: *attention score approximation does not
guarantee output preservation*, but CUR-based selection minimizes end-to-end
reconstruction loss. Achieves up to 9.6% higher accuracy than SnapKV and
ChunkKV under aggressive compression.

### Papers
- [CurDKV: Value-Guided KV Compression via CUR Decomposition](https://arxiv.org/abs/2509.15038) (NeurIPS 2025)
- [Column and Row Subset Selection Using Nuclear Scores](https://arxiv.org/abs/2407.01698) (2024)
- Voronin & Martinsson, "Efficient algorithms for CUR and interpolative matrix decompositions" (2017)

---

## 2. Coresets

**Maps to: [A] Key Selection + [B] Bias Fitting**

### What it is

A coreset is a small weighted subset of a dataset that provably approximates
some function (loss, density, etc.) computed over the full dataset. The key
idea: each coreset point carries a *weight* representing how many original
points it "stands for."

### The structural analogy

The KV cache compaction problem IS coreset construction:
- "Dataset" = the T original KV pairs
- "Function to preserve" = the attention output for any query
- "Coreset" = the t selected KV pairs
- **"Coreset weights" = exp(beta)** — this is exactly the bias term!

The beta term in Attention Matching is a *coreset weight*. The connection is
not metaphorical — it's algebraically identical.

### What it suggests concretely

**Sensitivity-based sampling.** Coreset theory says the optimal sampling
probability for point j is proportional to its "sensitivity" — the maximum
influence it has on the objective across all possible queries:

```
s_j = sup_q  |contribution of key j to attention output for query q|
           / |total attention output for query q|
```

This is theoretically superior to max-attention scoring because it accounts
for the *relative* contribution, not just the absolute attention weight. The
total coreset size needed for (1+eps) approximation is O(s_total / eps^2)
where s_total = sum of all sensitivities.

**Importance: this gives a theoretical lower bound on t.** For a given error
tolerance eps, coreset theory tells you the minimum number of keys you need.
The current method has no such guarantee.

### Papers
- [Improved Coresets for Kernel Density Estimates](https://dl.acm.org/doi/abs/10.5555/3174304.3175477) (SODA 2018) — O(1/eps^2) coreset size, dimension-independent
- Feldman & Langberg, "A Unified Framework for Approximating and Clustering Data" (STOC 2011)
- Braverman et al., "New Frameworks for Offline and Streaming Coreset Constructions" (2016)

---

## 3. Nyström Approximation

**Maps to: [A] Key Selection + [B] Bias Fitting**

### What it is

The Nyström method approximates a large kernel matrix K ≈ K_{nm} K_{mm}^{-1}
K_{mn} by selecting m "landmark" points and using the kernel evaluations
between all points and the landmarks.

### The structural analogy

The attention weight matrix softmax(Q K^T / sqrt(d)) is a kernel matrix
(specifically, a softmax kernel). The Nyström approximation selects a subset
of "landmark keys" and reconstructs the full attention matrix from them. The
linking matrix K_{mm}^{-1} plays the same role as the NNLS-fitted weights —
it corrects for how the landmarks represent the full set.

### What it suggests concretely

**Nyström landmark selection methods** as alternatives to max-attention key
scoring:
- **K-means Nyström:** cluster keys, use centroids as landmarks. O(T*k*iters).
  This naturally produces *diverse* landmarks without redundancy.
- **Leverage score Nyström:** sample keys proportional to their statistical
  leverage in the kernel matrix. Gives provable (1+eps) multiplicative error
  bounds.
- **Greedy Nyström:** iteratively add the landmark that most reduces the
  approximation error. This is essentially what OMP does, confirming that OMP
  is the right greedy strategy.

**Key difference from current approach:** Nyström theory says the
approximation quality depends on the *spectral decay* of the kernel matrix.
If the attention matrix has fast spectral decay (few dominant eigenvalues),
aggressive compression is possible. This suggests a *diagnostic*: compute the
top-k eigenvalues of the attention matrix to predict achievable compression
before attempting it.

### Papers
- Williams & Seeger, "Using the Nyström Method to Speed Up Kernel Machines" (NeurIPS 2001)
- [Improving CUR and Nyström Using QR](https://www.jmlr.org/papers/volume14/wang13c/wang13c.pdf) (JMLR 2013)
- Kumar et al., "Sampling Methods for the Nyström Method" (JMLR 2012)

---

## 4. Determinantal Point Processes (DPPs)

**Maps to: [A] Key Selection**

### What it is

A DPP is a probability distribution over subsets that favors *diverse*
selections. The probability of selecting a subset S is proportional to
det(L_S), where L_S is the principal submatrix of a kernel matrix L. Items
that are similar (high kernel value) are unlikely to be co-selected.

### The structural analogy

The key selection problem needs both *importance* (keep high-attention keys)
and *diversity* (don't keep redundant keys). Max-attention scoring gets
importance but ignores diversity. OMP gets both but is slow.

DPPs naturally balance quality and diversity in a single framework. The DPP
kernel L can be constructed as:

```
L_ij = quality_i * similarity(key_i, key_j) * quality_j
```

where quality = attention importance score and similarity = cosine similarity
between key vectors.

### What it suggests concretely

**k-DPP sampling** to select exactly t keys. Standard complexity is O(T^3)
for eigendecomposition, but recent work brings this down:
- Coreset-based k-DPP (Li et al., 2016): linear-time construction
- Alpha-DPP (Calandriello et al., NeurIPS 2020): sublinear sampling

For typical T ~ 1000-60000, even O(T^2) may be acceptable given the rest of
the pipeline is O(T * t).

**Practical simplification:** Greedy DPP maximization (select item maximizing
det increase) is O(T * t^2) and often within a constant factor of optimal.
This is simpler than OMP while capturing the diversity benefit.

### Papers
- [Kulesza & Taskar, "Determinantal Point Processes for ML"](http://www.alexkulesza.com/pubs/dpps_fnt12.pdf) (FnT ML 2012) — comprehensive reference
- [k-DPPs: Fixed-Size DPPs](https://icml.cc/2011/papers/611_icmlpaper.pdf) (ICML 2011)
- [Efficient k-DPP Sampling](https://arxiv.org/abs/1509.01618) (2016)
- [Sampling from k-DPP Without Looking at All Items](https://proceedings.neurips.cc/paper/2020/file/4d410063822cd9be28f86701c0bc3a31-Paper.pdf) (NeurIPS 2020)

---

## 5. Column Subset Selection Problem (CSSP) & Leverage Scores

**Maps to: [A] Key Selection**

### What it is

Given a matrix A, select t columns that minimize ||A - A_S A_S^+ A|| (the
error of projecting A onto the selected column span). The gold standard is
leverage score sampling — sample columns proportional to their "leverage,"
which measures how much each column contributes to the row space.

### The structural analogy

Key selection IS column subset selection on the attention matrix. Each key
defines a column of the attention matrix; selecting t keys = selecting t
columns. The CSSP literature provides both algorithms and provable
approximation guarantees.

### What it suggests concretely

**Leverage scores can be computed in O(T * d_k) time** from the key matrix
alone (via a randomized SVD), without computing the full attention matrix.
This is cheaper than the current approach which requires the full Q_ref @ K^T
product.

The approximation guarantee: with O(t * log(t) / eps^2) columns sampled by
leverage scores, the reconstruction error is within (1 + eps) of the best
rank-t approximation. This is an *existential guarantee* — the current
pipeline has none.

**Practical suggestion:** Replace max-attention scoring with leverage score
sampling. If the result is worse (because leverage scores don't account for
the softmax nonlinearity), use a hybrid: pre-filter by leverage scores, then
re-rank by attention.

### Papers
- Drineas et al., "CUR Matrix Decompositions for Improved Data Analysis" (PNAS 2009)
- [Provably Correct Column Subset Selection](https://jmlr.org/papers/volume18/15-233/15-233.pdf) (JMLR 2017)
- Mahoney & Drineas, "CUR Factorization and Leverage Scores" ([lecture notes](https://www.cs.cornell.edu/courses/cs6220/2017fa/CS6220_Lecture14.pdf))

---

## 6. Optimal Transport & Sinkhorn Iterations

**Maps to: [B] Bias Fitting**

### What it is

Optimal transport (OT) finds the minimum-cost way to transform one
probability distribution into another. Sinkhorn iterations solve the
entropy-regularized OT problem via alternating row/column normalization of
a cost matrix.

### The structural analogy

The mass matching problem (Step 2) asks: find weights w such that the
compacted attention mass matches the original. This is a 1D transport problem:
redistribute the original mass (distributed across T keys) onto t keys via
non-negative weights.

More precisely, the NNLS problem M @ w ≈ m can be reinterpreted as: find a
transport plan from the "compressed" distribution to the "full" distribution
that preserves marginals.

### What it suggests concretely

**Replace projected gradient NNLS with Sinkhorn-like multiplicative updates.**
Recent work on Optimal Transport Linear Models (arXiv:2504.04609) shows that
Sinkhorn-like iterations can solve non-negative linear regression with an OT
loss. The updates are multiplicative, which *automatically ensures
non-negativity* — no projection step needed. This is potentially faster and
more numerically stable than the current projected gradient approach.

**Entropic regularization** naturally prevents degenerate solutions (all mass
on one key), acting as a smoother version of the 1e-12 floor hack.

### Papers
- Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NeurIPS 2013)
- [Scalable Approximate Algorithms for OT Linear Models](https://arxiv.org/html/2504.04609v1) (2025)
- [Near-Linear Time Approximation Algorithms for OT via Sinkhorn](https://arxiv.org/pdf/1705.09634) (NeurIPS 2017)

---

## 7. Kernel Herding & Maximum Mean Discrepancy (MMD)

**Maps to: [A] Key Selection + [B] Bias Fitting**

### What it is

Kernel herding is a deterministic algorithm that greedily selects points to
minimize the Maximum Mean Discrepancy (MMD) between the selected subset and
the full distribution. It is equivalent to the Frank-Wolfe algorithm applied
to MMD minimization. Convergence rate: O(1/n) for n selected points.

### The structural analogy

The key selection problem is: choose t keys such that the attention
distribution over selected keys approximates the attention distribution over
all keys. MMD is a natural metric for this — it measures the worst-case
difference in expectations over functions in a reproducing kernel Hilbert
space.

Kernel herding applied to key selection would:
1. Start with empty selection
2. At each step, add the key that most reduces MMD between the selected
   subset's attention distribution and the full distribution
3. Each selected key gets a weight (equivalent to beta)

### What it suggests concretely

**Kernel herding as a principled replacement for OMP.** OMP minimizes mass
residual; herding minimizes MMD. The convergence rate O(1/n) means the
approximation error halves every time you double the budget — a clean
theoretical guarantee.

**The connection to Frank-Wolfe is key:** kernel herding = Frank-Wolfe on
MMD = conditional gradient on a convex objective. This means it inherits
Frank-Wolfe's convergence guarantees while naturally producing sparse
(subset) solutions. The current OMP has no such convergence rate guarantee.

### Papers
- Chen, Welling, Smola, "Super-Samples from Kernel Herding" (UAI 2010)
- [Performance Analysis of Greedy MMD Minimization](https://link.springer.com/article/10.1007/s11222-022-10184-1) (Statistics & Computing 2022)
- [Improved Coresets for Kernel Density Estimates](https://dl.acm.org/doi/abs/10.5555/3174304.3175477) (SODA 2018)

---

## 8. Rate-Distortion Theory

**Maps to: [D] Budget Allocation + overall compression limits**

### What it is

Rate-distortion theory establishes the theoretical minimum "rate" (bits or,
here, number of retained keys) needed to represent a source at a given
distortion level. The rate-distortion function R(D) is the fundamental limit
of lossy compression.

### The structural analogy

For a given model, context, and quality tolerance D (measured as attention
output MSE), there exists a *minimum* number of KV entries t* below which no
compaction method can succeed. This is the rate-distortion limit for the
specific "source" (the attention function).

### What it suggests concretely

**Compute an empirical R(D) curve per head** by measuring reconstruction error
at various compression levels. This directly tells you:
- Which heads have fast-decaying R(D) → compress aggressively
- Which heads have slow-decaying R(D) → need more budget
- The overall achievable compression ratio for a given quality target

This is a more principled version of the paper's "sensitivity curves" for
per-head budget allocation. Instead of measuring sensitivity heuristically,
measure the actual rate-distortion function.

**Vector quantization (VQ):** Rate-distortion theory says VQ is always better
than scalar quantization. Applied to KV compaction: representing C_v as
codebook entries (k-means centroids of original V vectors) rather than
arbitrary vectors should approach the rate-distortion limit more efficiently.

### Papers
- Shannon, "Coding Theorems for a Discrete Source with a Fidelity Criterion" (1959)
- [Stanford — Lossy Compression Basics & Quantization](https://stanforddatacompressionclass.github.io/notes/lossy/quant.html)
- [Gray & Neuhoff — Quantization](https://www.math.ucdavis.edu/~saito/data/quantization/44it06-gray.pdf) (IEEE 1998)

---

## 9. Frank-Wolfe (Conditional Gradient) Method

**Maps to: [A] Key Selection + [C] Value Fitting jointly**

### What it is

Frank-Wolfe optimizes a smooth function over a convex set by iteratively
adding the element from the constraint set that most improves the objective.
It naturally produces *sparse* solutions — after t iterations, the solution
is a combination of at most t extreme points.

### The structural analogy

If we constrain C_v to be a weighted combination of at most t original value
vectors, Frank-Wolfe jointly solves key selection AND value fitting:
- Each iteration selects one key (the "extreme point" that most reduces
  the attention output error)
- The weights on selected keys are jointly optimized
- After t iterations: t selected keys with optimal combination weights

This is equivalent to OMP but with a cleaner convergence theory.

### What it suggests concretely

**Frank-Wolfe as a unified framework for Steps 1+3.** Instead of the current
pipeline (select keys, then fit values separately), Frank-Wolfe selects keys
and fits values in a single optimization loop. Each iteration:
1. Compute gradient of attention output error w.r.t. the current C_v
2. Find the original value vector most aligned with this gradient (= select
   next key)
3. Update the combination weights (line search or fixed step)

**Convergence:** O(1/t) rate for smooth objectives, with the sparsity of the
solution directly matching the budget constraint.

### Papers
- Frank & Wolfe, "An Algorithm for Quadratic Programming" (1956)
- Jaggi, "Revisiting Frank-Wolfe" (ICML 2013)
- Clarkson, "Coresets, Sparse Greedy Approximation, and the Frank-Wolfe Algorithm" (2010)

---

## 10. Alternating Minimization

**Maps to: [B] Bias Fitting + [C] Value Fitting jointly**

### What it is

Alternating minimization optimizes a function of two variables (x, y) by
alternately fixing one and optimizing the other. Despite overall
non-convexity, it converges to a global optimum for many structured problems
(matrix factorization, phase retrieval) given a good initialization.

### The structural analogy

The current pipeline fits beta (Step 2) and C_v (Step 3) independently. But
they interact: the optimal C_v depends on beta (through the softmax weights),
and the "best" beta depends on what C_v can achieve. Alternating minimization
would:
1. Fit beta with C_v fixed (NNLS — existing Step 2)
2. Fit C_v with beta fixed (LS — existing Step 3)
3. Repeat until convergence

### What it suggests concretely

**2-3 alternation rounds between Steps 2 and 3.** Theory says: if the
initialization is good (which it is — single-pass Steps 2+3), alternating
minimization converges *linearly* to the joint optimum. The per-iteration
cost is just repeating the existing NNLS + LS solves, which together take ~4s.

**This is the single cheapest improvement to implement** — it requires zero
new algorithms, just a loop around existing Steps 2 and 3.

**Convergence guarantee:** For bilinear problems (which the beta/C_v
interaction approximates), alternating minimization achieves linear
convergence to the global optimum from a constant-factor initialization
(Jain et al., 2013).

### Papers
- Jain, Netrapalli, Sanghavi, "Low-rank Matrix Completion using Alternating Minimization" (STOC 2013)
- [Nonconvex Optimization Meets Low-Rank Matrix Factorization: An Overview](https://yuxinchen2020.github.io/publications/NcxOverview_Arxiv.pdf)
- Bolte et al., "Proximal Alternating Linearized Minimization" (Math. Programming 2014)

---

## 11. Attention Head Pruning & the Lottery Ticket Hypothesis

**Maps to: [D] Budget Allocation**

### What it is

Research on which attention heads can be removed without hurting performance.
The lottery ticket hypothesis says sparse subnetworks exist that match full
network performance. Applied to heads: most heads are redundant; a small
"winning ticket" subset does the heavy lifting.

### The structural analogy

Per-head budget allocation in KV compaction asks: which heads are sensitive
to compression? Head pruning research asks: which heads matter at all? These
are the same question at different compression levels.

### What it suggests concretely

**Known head importance patterns transfer directly:**
- Voita et al. (ACL 2019): only a small subset of heads have interpretable
  functions (positional, syntactic, rare-word). These are the heads that need
  the most budget.
- Behnke & Heafield (EMNLP 2020): up to 75% of heads can be removed in
  transformer-big with negligible BLEU loss. This implies 75% of heads could
  be maximally compressed.
- Differentiable subset pruning (Li et al., 2021): learns per-head importance
  scores differentiably. These scores could directly inform budget allocation.

**Concrete application:** Use a pre-computed head importance ranking (from
pruning literature) as the prior for budget allocation. Heads ranked as
"prunable" get minimal budget; heads ranked as "essential" get maximum budget.

### Papers
- [Voita et al., "Analyzing Multi-Head Self-Attention"](https://lena-voita.github.io/posts/acl19_heads.html) (ACL 2019)
- [Behnke & Heafield, "Losing Heads in the Lottery"](https://aclanthology.org/2020.emnlp-main.211/) (EMNLP 2020)
- [Differentiable Subset Pruning of Transformer Heads](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00436/108868/) (TACL 2021)

---

## 12. Mixture of Experts Merging

**Maps to: [C] Value Fitting**

### What it is

MoE compression combines multiple expert networks into fewer ones while
preserving output quality. Recent methods (MergeMoE, Sub-MoE, PuzzleMoE)
merge expert weights via learned combinations, SVD-based subspace alignment,
or output-matching optimization.

### The structural analogy

Each original value vector V_j is an "expert" that contributes to the
attention output. Compaction merges T "experts" into t by finding C_v vectors
that preserve the combined output — exactly what MoE merging does.

### What it suggests concretely

**Output-matching merging (MergeMoE):** rather than fitting C_v to minimize
||X @ C_v - Y|| directly, use the MergeMoE insight that *merging should
target output matching, not parameter averaging*. For KV compaction, this
means optimizing C_v to match the final model output (after all subsequent
layers), not just the attention output of the current layer.

**Subspace merging (Sub-MoE):** joint SVD on concatenated value vectors to
find shared structure, then merge in the shared subspace. This could identify
which value vectors are naturally groupable, informing both key selection and
value fitting.

### Papers
- [MergeMoE: Efficient Compression via Expert Output Merging](https://arxiv.org/html/2510.14436v1) (2025)
- [Sub-MoE: Compression via Subspace Expert Merging](https://arxiv.org/abs/2506.23266) (2025)
- [PuzzleMoE: Sparse Expert Merging](https://arxiv.org/html/2511.04805v1) (2025)

---

## 13. Matrix Sketching (Frequent Directions)

**Maps to: [A]+[C] — an alternative to subset selection entirely**

### What it is

Matrix sketching maintains a low-rank approximation of a stream of vectors.
Frequent Directions (Liberty, 2013) maintains a t-row sketch B such that
||A - A B^+ B|| ≤ ||A - A_t|| + ||A||_F / sqrt(t), where A_t is the best
rank-t approximation.

### The structural analogy

Instead of selecting t keys from T, project *all* T keys into a
t-dimensional sketch. The sketch is a t × d matrix that approximates the
full key matrix. This is fundamentally different from subset selection — the
sketched keys don't correspond to any original token.

### What it suggests concretely

**Streaming compaction.** Frequent Directions processes keys one at a time,
maintaining a fixed-size sketch. This enables *online* compaction during
generation — every new token's KV pair is absorbed into the sketch — without
needing to batch-process the full cache.

**The trade-off:** sketching produces better low-rank approximations than
subset selection (provably optimal bounds), but the sketched keys lose
interpretability and correspondence to original tokens. This makes it
incompatible with the current beta framework (which assumes C_k ⊆ K).

**Hybrid approach:** Use sketching for the *value* side (where correspondence
doesn't matter) while keeping subset selection for keys (where the softmax
structure requires real key vectors).

### Papers
- Liberty, "Simple and Deterministic Matrix Sketching" (KDD 2013)
- Ghashami et al., "Frequent Directions: Simple and Deterministic Matrix Sketching" (SIAM 2016)

---

## 14. Knowledge Distillation (Feature-Level)

**Maps to: [C] Value Fitting — alternative loss functions**

### What it is

Feature-level distillation trains a student to match intermediate
representations of a teacher, not just final outputs. CKA (Centered Kernel
Alignment) measures representational similarity between layers, invariant
to rotation and scaling.

### The structural analogy

Step 3 (value fitting) minimizes ||X @ C_v - Y|| — MSE between compacted and
original attention outputs. But does matching attention output at layer L
guarantee matching the final model output? Feature distillation research
says: not necessarily. Intermediate representation errors can amplify or
cancel through subsequent layers.

### What it suggests concretely

**End-to-end distillation loss.** Instead of matching attention output at the
current layer, propagate the compacted output through subsequent layers and
minimize final output divergence. This is more expensive but directly
optimizes what we care about.

**CKA as a diagnostic:** compute CKA between the original and compacted
attention outputs. If CKA is high even when MSE is moderate, the compaction
preserves the *structure* of the representation (what matters), not just the
exact values.

**Layer-wise importance weighting:** distillation research shows some layers
are more important to match than others. Apply more aggressive compression
to layers where errors don't propagate, less to layers where they do.

### Papers
- Romero et al., "FitNets: Hints for Thin Deep Nets" (ICLR 2015)
- Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019)
- Palu (ICLR 2025): already applies layer-wise sensitivity for KV cache via SVD decomposition

---

## 15. Compressed Sensing & Sparse Recovery

**Maps to: [B] Bias Fitting**

### What it is

Compressed sensing recovers sparse signals from underdetermined linear
measurements. The mass matching equation M @ w = m is an underdetermined
system (n_q equations, t unknowns, n_q < t typically) with a non-negativity
constraint.

### The structural analogy

The NNLS problem in Step 2 is a non-negative sparse recovery problem. The
measurement matrix M has a specific structure — its entries are exponentials
of dot products — which may satisfy restricted isometry-like properties.

### What it suggests concretely

**Basis pursuit (L1 minimization) instead of NNLS.** If we want *sparse*
biases (most beta_j = 0, a few are large), L1 regularization naturally
produces this. Sparse beta means most keys contribute their "natural" mass
while a few keys are boosted to compensate for removed keys.

**The sparsity pattern is informative:** if the NNLS solution has many w_j ≈ 1
(beta_j ≈ 0), those keys are "self-sufficient" — they naturally carry the
right mass. Keys with large w_j are "load-bearing" — they compensate for
many removed keys. This pattern could guide key selection: prefer keys whose
natural mass is close to what's needed (w ≈ 1) and avoid keys that require
extreme beta corrections.

### Papers
- Candès & Tao, "Near-Optimal Signal Recovery from Random Projections" (IEEE IT 2006)
- Slawski & Hein, "Non-negative Least Squares for High-Dimensional Linear Models" (2013)

---

## 16. K-Means Clustering of Keys

**Maps to: [A] Key Selection + [C] Value Fitting**

### What it is

Lloyd's algorithm partitions T keys into t clusters and represents each
cluster by its centroid. This is the classic vector quantization approach.

### The structural analogy

Instead of selecting t keys (subset selection), cluster all T keys into t
groups and use centroids. Each centroid is a weighted average of the keys in
its cluster — a natural "merged" key that represents multiple original keys.

### What it suggests concretely

**Centroid keys lift the C_k ⊆ K constraint** (identified as a key limitation
in the paper) while remaining computationally cheap. K-means on T keys with
d_k dimensions runs in O(T * d_k * t * iters) — comparable to existing key
scoring.

**The cluster assignment naturally defines beta:** if cluster j contains n_j
keys, then beta_j ≈ log(n_j) (the cluster mass).

**The cluster centroid naturally defines C_v:** the centroid value is the
attention-weighted average of values in the cluster.

**Trade-off:** centroid keys are not real key vectors from the cache, so they
may not be compatible with inference engines that expect integer token
indices. But at the tensor level, you just overwrite the key/value data.

---

## 17. Expectation-Maximization (EM)

**Maps to: [A]+[B]+[C] — an alternative joint framework**

### What it is

EM iterates between assigning data points to clusters (E-step) and
recomputing cluster parameters (M-step). It maximizes a lower bound on the
log-likelihood.

### The structural analogy

View compaction as a latent variable model:
- Latent variable z_j ∈ {1,...,t}: which compacted key does original key j
  map to?
- E-step: compute soft assignment probabilities (attention-weighted)
- M-step: recompute C_k, beta, C_v given assignments

### What it suggests concretely

**Soft assignment instead of hard selection.** The current approach makes a
hard decision in Step 1 (select or discard each key). EM would maintain soft
assignments throughout, allowing keys to "partially contribute" to multiple
compacted entries. This is more expressive but harder to implement in the
existing attention framework.

**Practical version:** Run hard selection (Step 1) but then do 2-3 EM
iterations: reassign borderline keys, update beta and C_v. This captures
most of the benefit of soft assignment with minimal implementation cost.

---

## Summary: Concept-to-Component Map

| Concept | Key Select [A] | Bias [B] | Value [C] | Budget [D] | New idea? |
|---------|:-:|:-:|:-:|:-:|-----------|
| CUR / ID decomposition | x | | x | | Leverage scores for selection |
| Coresets | x | x | | | Sensitivity sampling + theoretical bounds |
| Nyström approximation | x | x | | | Spectral decay diagnostic |
| DPPs | x | | | | Diversity-aware selection |
| CSSP / Leverage scores | x | | | | Cheap O(Td) score computation |
| Optimal transport / Sinkhorn | | x | | | Multiplicative NNLS updates |
| Kernel herding / MMD | x | x | | | Frank-Wolfe with convergence rate |
| Rate-distortion theory | | | | x | Theoretical compression limits |
| Frank-Wolfe | x | | x | | Joint key+value optimization |
| Alternating minimization | | x | x | | Iterate Steps 2+3 (cheapest win) |
| Head pruning / lottery | | | | x | Pre-computed head importance |
| MoE merging | | | x | | Output-matching loss |
| Matrix sketching | x | | x | | Streaming online compaction |
| Feature distillation | | | x | | End-to-end loss |
| Compressed sensing | | x | | | Sparse beta via L1 |
| K-means clustering | x | | x | | Centroid keys (lift C_k ⊆ K) |
| EM algorithm | x | x | x | | Soft assignment framework |

### Top 5 Most Actionable (effort vs. impact)

1. **Alternating minimization** (Sec 10) — loop existing Steps 2+3. Zero new
   code, just a for loop. Likely the cheapest quality improvement.
2. **Leverage score key selection** (Sec 5) — O(Td) computation, no Q_ref
   needed. Could eliminate the repeat-prefill cost entirely.
3. **DPP / greedy determinantal selection** (Sec 4) — add diversity to key
   selection without OMP's full cost.
4. **Sinkhorn for mass matching** (Sec 6) — drop-in NNLS replacement with
   automatic non-negativity.
5. **K-means centroid keys** (Sec 16) — lifts the C_k ⊆ K restriction at
   low cost. The researchers identified this restriction as a key limitation.
