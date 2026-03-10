// KV Cache Compaction via Attention Matching - Math Utilities
//
// Pure CPU float32 linear algebra routines used by the compaction algorithm.
// Extracted for testability.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

// ============================================================================
// Linear algebra utilities (CPU-side, float32)
// ============================================================================

// Compute C = A * B^T  where A is (m x k), B is (n x k), result is (m x n)
static void mat_mul_ABt(const float * A, const float * B, float * C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[j * k + l];
            }
            C[i * n + j] = sum;
        }
    }
}

// Compute C = A^T * B  where A is (m x k), B is (m x n), result is (k x n)
static void mat_mul_AtB(const float * A, const float * B, float * C, int m, int k, int n) {
    // zero out C
    memset(C, 0, k * n * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < n; l++) {
                C[j * n + l] += A[i * k + j] * B[i * n + l];
            }
        }
    }
}

// Softmax over rows: input (m x n), output (m x n), in-place safe
static void softmax_rows(float * data, int m, int n) {
    for (int i = 0; i < m; i++) {
        float * row = data + i * n;
        float max_val = row[0];
        for (int j = 1; j < n; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        float inv_sum = 1.0f / (sum + 1e-12f);
        for (int j = 0; j < n; j++) {
            row[j] *= inv_sum;
        }
    }
}

// Row-wise exp with max-shift for numerical stability: input (m x n)
// Returns exp(data - max_per_row) and stores the sum per row in row_sums
static void exp_rows_stable(float * data, float * row_sums, int m, int n) {
    for (int i = 0; i < m; i++) {
        float * row = data + i * n;
        float max_val = row[0];
        for (int j = 1; j < n; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        row_sums[i] = sum;
    }
}

// Solve non-negative least squares via projected gradient descent:
//   min_{w >= 0} ||A*w - b||^2
// A is (m x n), b is (m), w is (n)
// Returns solution in w
static void nnls_solve(const float * A, const float * b, float * w, int m, int n, int max_iter = 200) {
    // Precompute A^T * A and A^T * b
    std::vector<float> AtA(n * n);
    std::vector<float> Atb(n);

    // AtA = A^T * A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            AtA[i * n + j] = sum;
        }
    }

    // Atb = A^T * b
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int k = 0; k < m; k++) {
            sum += A[k * n + i] * b[k];
        }
        Atb[i] = sum;
    }

    // Initialize w to unconstrained least squares, clamped to >= 0
    // Simple init: w = max(0, (A^T A)^{-1} A^T b) via gradient descent from w=1
    for (int i = 0; i < n; i++) {
        w[i] = 1.0f;
    }

    // Compute step size: 1 / (max eigenvalue of AtA) ≈ 1 / (trace(AtA))
    float trace = 0.0f;
    for (int i = 0; i < n; i++) {
        trace += AtA[i * n + i];
    }
    float step = 1.0f / (trace + 1e-8f);

    // Projected gradient descent
    std::vector<float> grad(n);
    for (int iter = 0; iter < max_iter; iter++) {
        // grad = AtA * w - Atb
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += AtA[i * n + j] * w[j];
            }
            grad[i] = sum - Atb[i];
        }

        // w = max(0, w - step * grad)
        for (int i = 0; i < n; i++) {
            w[i] = std::max(1e-12f, w[i] - step * grad[i]);
        }
    }
}

// Solve least squares: min ||A*x - b||^2 via normal equations
// A is (m x n), b is (m x p), x is (n x p)
// Uses Cholesky-like approach: x = (A^T A)^{-1} A^T b
// For simplicity, uses pseudo-inverse via regularized normal equations
static void least_squares_solve(const float * A, const float * b, float * x,
                                int m, int n, int p, float ridge = 1e-6f) {
    // Compute AtA = A^T * A  (n x n)
    std::vector<float> AtA(n * n, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            AtA[i * n + j] = sum;
        }
    }

    // Add ridge regularization
    for (int i = 0; i < n; i++) {
        AtA[i * n + i] += ridge;
    }

    // Compute Atb = A^T * b  (n x p)
    std::vector<float> Atb(n * p, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int l = 0; l < p; l++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * b[k * p + l];
            }
            Atb[i * p + l] = sum;
        }
    }

    // Solve AtA * x = Atb via Gaussian elimination with partial pivoting
    // Augmented matrix [AtA | Atb]
    std::vector<float> aug(n * (n + p));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * (n + p) + j] = AtA[i * n + j];
        }
        for (int j = 0; j < p; j++) {
            aug[i * (n + p) + n + j] = Atb[i * p + j];
        }
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < n; col++) {
        // Find pivot
        int max_row = col;
        float max_val = fabsf(aug[col * (n + p) + col]);
        for (int row = col + 1; row < n; row++) {
            float val = fabsf(aug[row * (n + p) + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows
        if (max_row != col) {
            for (int j = 0; j < n + p; j++) {
                std::swap(aug[col * (n + p) + j], aug[max_row * (n + p) + j]);
            }
        }

        float pivot = aug[col * (n + p) + col];
        if (fabsf(pivot) < 1e-12f) {
            continue; // skip singular column
        }

        // Eliminate below
        for (int row = col + 1; row < n; row++) {
            float factor = aug[row * (n + p) + col] / pivot;
            for (int j = col; j < n + p; j++) {
                aug[row * (n + p) + j] -= factor * aug[col * (n + p) + j];
            }
        }
    }

    // Back substitution
    for (int col = n - 1; col >= 0; col--) {
        float pivot = aug[col * (n + p) + col];
        if (fabsf(pivot) < 1e-12f) {
            for (int j = 0; j < p; j++) {
                x[col * p + j] = 0.0f;
            }
            continue;
        }
        for (int j = 0; j < p; j++) {
            float val = aug[col * (n + p) + n + j];
            for (int row = col + 1; row < n; row++) {
                val -= aug[col * (n + p) + row] * x[row * p + j];
            }
            x[col * p + j] = val / pivot;
        }
    }
}

// ============================================================================
// Compaction algorithm types and implementation
// ============================================================================

struct compacted_head {
    std::vector<int>   selected_indices;  // which original tokens were selected
    std::vector<float> beta;              // attention mass biases [t]
    std::vector<float> C_v;               // refit values [t * d_v]
};

// Compact a single KV head using the Highest Attention Keys method
//
//   K:       [T, d_k] original keys for this head
//   V:       [T, d_v] original values for this head
//   Q_ref:   [n_q, d_k] reference queries
//   t:       target compacted size
//   d_k:     key dimension
//   d_v:     value dimension
//
// Returns compacted_head with selected indices, beta, and C_v
static compacted_head compact_head_highest_attn(
        const float * K, const float * V, const float * Q_ref,
        int T, int n_q, int d_k, int d_v, int t) {

    compacted_head result;
    result.selected_indices.resize(t);
    result.beta.resize(t);
    result.C_v.resize(t * d_v);

    if (t >= T) {
        // No compaction needed
        for (int i = 0; i < T; i++) result.selected_indices[i] = i;
        std::fill(result.beta.begin(), result.beta.end(), 0.0f);
        memcpy(result.C_v.data(), V, T * d_v * sizeof(float));
        return result;
    }

    // Step 1: Compute attention scores Q_ref @ K^T / sqrt(d_k)
    //   scores: [n_q, T]
    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);
    std::vector<float> scores(n_q * T);
    mat_mul_ABt(Q_ref, K, scores.data(), n_q, T, d_k);
    for (int i = 0; i < n_q * T; i++) {
        scores[i] *= inv_sqrt_dk;
    }

    // Compute exp(scores) with max-shift for mass computation
    std::vector<float> exp_scores(scores); // copy
    std::vector<float> row_sums(n_q);
    exp_rows_stable(exp_scores.data(), row_sums.data(), n_q, T);

    // Compute softmax attention weights for key scoring
    std::vector<float> attn_weights(scores);
    softmax_rows(attn_weights.data(), n_q, T);

    // Score each key: max attention weight across queries
    std::vector<float> key_scores(T, 0.0f);
    for (int j = 0; j < T; j++) {
        float max_score = 0.0f;
        for (int i = 0; i < n_q; i++) {
            float w = attn_weights[i * T + j];
            if (w > max_score) max_score = w;
        }
        key_scores[j] = max_score;
    }

    // Select top-t keys by score
    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return key_scores[a] > key_scores[b]; });

    // Sort selected indices for cache locality
    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());
    result.selected_indices = selected;

    // Step 2: Solve NNLS for beta (mass matching)
    //   We want: sum_j exp(q_i * C_k_j / sqrt(d)) * w_j ≈ sum_j exp(q_i * K_j / sqrt(d))
    //   where C_k are the selected keys and w_j = exp(beta_j)
    //
    //   Design matrix M: M_ij = exp(q_i * K_{selected[j]} / sqrt(d))
    //   Target: m_i = sum_j exp(q_i * K_j / sqrt(d)) = row_sums[i] (already computed)

    std::vector<float> M(n_q * t);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < t; j++) {
            M[i * t + j] = exp_scores[i * T + selected[j]];
        }
    }

    // Target mass: already computed as row_sums
    std::vector<float> w(t);
    nnls_solve(M.data(), row_sums.data(), w.data(), n_q, t);

    // beta = log(w)
    for (int j = 0; j < t; j++) {
        result.beta[j] = logf(std::max(1e-12f, w[j]));
    }

    // Step 3: Solve least squares for C_v (value fitting)
    //   We want: softmax(q * C_k^T + beta) * C_v ≈ softmax(q * K^T) * V
    //
    //   X_ij = softmax(q_i * C_k_j + beta_j) (compacted attention weights)
    //   Y_i  = softmax(q_i * K^T) * V         (original attention output)
    //   Solve: X * C_v = Y

    // Compute X: attention weights with compacted keys + bias
    // scores[] are already scaled by inv_sqrt_dk (line 286), so just add beta
    std::vector<float> X(n_q * t);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < t; j++) {
            X[i * t + j] = scores[i * T + selected[j]] + result.beta[j];
        }
    }
    softmax_rows(X.data(), n_q, t);

    // Compute Y: original attention output = attn_weights @ V  [n_q, d_v]
    std::vector<float> Y(n_q * d_v, 0.0f);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < T; j++) {
            float w_ij = attn_weights[i * T + j];
            for (int d = 0; d < d_v; d++) {
                Y[i * d_v + d] += w_ij * V[j * d_v + d];
            }
        }
    }

    // Solve: X * C_v = Y  =>  C_v = (X^T X)^{-1} X^T Y
    least_squares_solve(X.data(), Y.data(), result.C_v.data(), n_q, t, d_v);

    return result;
}

// ============================================================================
// Per-head sensitivity measurement and non-uniform budget allocation
// ============================================================================

// Compute reconstruction error (MSE) for a single head at a given compression ratio.
// This measures how sensitive a head is to compression — higher error = more sensitive.
//
//   K:       [T, d_k] original keys for this head
//   V:       [T, d_v] original values for this head
//   Q_ref:   [n_q, d_k] reference queries
//   t_keep:  number of tokens to keep after compaction
//
// Returns the mean squared error between original and compacted attention outputs
// averaged across all reference queries.
static float compute_head_reconstruction_error(
        const float * K, const float * V, const float * Q_ref,
        int T, int n_q, int d_k, int d_v, int t_keep) {

    if (t_keep >= T) return 0.0f;
    if (T <= 0 || n_q <= 0 || d_k <= 0 || d_v <= 0 || t_keep <= 0) return 0.0f;

    // Run compaction
    compacted_head result = compact_head_highest_attn(K, V, Q_ref, T, n_q, d_k, d_v, t_keep);

    const float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    // Compute original attention output for all reference queries: Y_orig = softmax(Q @ K^T / sqrt(d)) @ V
    // Compute compacted attention output: Y_comp = softmax(Q @ C_k^T / sqrt(d) + beta) @ C_v
    // MSE = mean over queries and dimensions of (Y_orig - Y_comp)^2

    // Extract C_k (selected keys)
    std::vector<float> C_k(t_keep * d_k);
    for (int i = 0; i < t_keep; i++) {
        memcpy(C_k.data() + i * d_k,
               K + result.selected_indices[i] * d_k,
               d_k * sizeof(float));
    }

    float total_mse = 0.0f;

    for (int q = 0; q < n_q; q++) {
        const float * qi = Q_ref + q * d_k;

        // Original: scores = qi @ K^T / sqrt(d)
        std::vector<float> orig_scores(T);
        for (int j = 0; j < T; j++) {
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++) dot += qi[d] * K[j * d_k + d];
            orig_scores[j] = dot * inv_sqrt_dk;
        }
        softmax_rows(orig_scores.data(), 1, T);

        // Original output
        std::vector<float> orig_out(d_v, 0.0f);
        for (int j = 0; j < T; j++) {
            for (int d = 0; d < d_v; d++) {
                orig_out[d] += orig_scores[j] * V[j * d_v + d];
            }
        }

        // Compacted: scores = qi @ C_k^T / sqrt(d) + beta
        std::vector<float> comp_scores(t_keep);
        for (int j = 0; j < t_keep; j++) {
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++) dot += qi[d] * C_k[j * d_k + d];
            comp_scores[j] = dot * inv_sqrt_dk + result.beta[j];
        }
        softmax_rows(comp_scores.data(), 1, t_keep);

        // Compacted output with C_v
        std::vector<float> comp_out(d_v, 0.0f);
        for (int j = 0; j < t_keep; j++) {
            for (int d = 0; d < d_v; d++) {
                comp_out[d] += comp_scores[j] * result.C_v[j * d_v + d];
            }
        }

        // Accumulate MSE
        for (int d = 0; d < d_v; d++) {
            float diff = comp_out[d] - orig_out[d];
            total_mse += diff * diff;
        }
    }

    return total_mse / (float)(n_q * d_v);
}

// Sensitivity profile for a single head: error at multiple compression ratios
struct head_sensitivity_profile {
    int    layer;
    int    head;
    float  sensitivity;                           // scalar summary (area under error curve)
    std::vector<std::pair<float, float>> curve;   // (ratio, mse) pairs
};

// Compute sensitivity profile for a head at multiple ratios.
// ratios: array of keep-ratios to test (e.g., {0.1, 0.2, 0.5, 0.8})
// Returns the profile with per-ratio errors and a scalar sensitivity summary.
static head_sensitivity_profile compute_head_sensitivity(
        const float * K, const float * V, const float * Q_ref,
        int T, int n_q, int d_k, int d_v,
        int layer, int head,
        const float * ratios, int n_ratios) {

    head_sensitivity_profile prof;
    prof.layer = layer;
    prof.head  = head;
    prof.curve.resize(n_ratios);

    float area = 0.0f;
    for (int r = 0; r < n_ratios; r++) {
        int t_keep = std::max(1, (int)(T * ratios[r]));
        float mse = compute_head_reconstruction_error(K, V, Q_ref, T, n_q, d_k, d_v, t_keep);
        prof.curve[r] = {ratios[r], mse};
        area += mse;  // simple sum as sensitivity proxy
    }

    prof.sensitivity = area / (float)n_ratios;
    return prof;
}

// Budget allocation result
struct head_budget_allocation {
    std::vector<int> budgets;        // per-head token budgets [n_total_heads]
    std::vector<float> weights;      // per-head sensitivity weights [n_total_heads]
};

// Allocate per-head token budgets given sensitivity scores.
//
// Uses proportional allocation: heads with higher sensitivity get more tokens.
// Total tokens kept = n_tokens_total (the global budget).
//
//   sensitivities: [n_heads] sensitivity scores (higher = needs more budget)
//   n_heads:       total number of (layer, head) pairs
//   n_active:      total active tokens in the cache
//   target_ratio:  overall compression ratio (fraction to keep)
//   min_ratio:     minimum per-head keep ratio (floor, e.g., 0.05)
//   max_ratio:     maximum per-head keep ratio (ceiling, e.g., 0.95)
//
// Returns per-head budgets (number of tokens to keep) and normalized weights.
// The global cell budget (cells actually evicted) uses the maximum per-head budget
// since cells are shared. Individual heads may use fewer cells than the global set.
static head_budget_allocation allocate_head_budgets(
        const float * sensitivities, int n_heads,
        int n_active, float target_ratio,
        float min_ratio = 0.02f, float max_ratio = 0.95f) {

    head_budget_allocation result;
    result.budgets.resize(n_heads);
    result.weights.resize(n_heads);

    if (n_heads <= 0 || n_active <= 0) return result;

    // Compute total sensitivity
    float total_sens = 0.0f;
    for (int i = 0; i < n_heads; i++) {
        total_sens += std::max(1e-12f, sensitivities[i]);
    }

    // Proportional allocation:
    //   weight[i] = sensitivity[i] / total_sensitivity
    //   budget[i] = n_active * (target_ratio * weight[i] * n_heads)
    //   (scaled so that mean(budget) = n_active * target_ratio)
    //
    // Intuition: if all sensitivities are equal, every head gets target_ratio * n_active.
    // More sensitive heads get proportionally more.

    const float mean_budget = n_active * target_ratio;
    const int min_budget = std::max(1, (int)(n_active * min_ratio));
    const int max_budget = std::min(n_active, (int)(n_active * max_ratio));

    // First pass: proportional allocation
    std::vector<float> raw_budgets(n_heads);
    for (int i = 0; i < n_heads; i++) {
        float w = std::max(1e-12f, sensitivities[i]) / total_sens;
        result.weights[i] = w;
        raw_budgets[i] = mean_budget * w * n_heads;  // scale so mean = mean_budget
    }

    // Second pass: clamp and redistribute
    // Use iterative clamping to handle min/max constraints
    std::vector<bool> clamped(n_heads, false);
    for (int iter = 0; iter < 10; iter++) {
        float clamped_total = 0.0f;
        float unclamped_total_sens = 0.0f;
        int n_unclamped = 0;

        for (int i = 0; i < n_heads; i++) {
            if (clamped[i]) {
                clamped_total += result.budgets[i];
            } else {
                unclamped_total_sens += std::max(1e-12f, sensitivities[i]);
                n_unclamped++;
            }
        }

        if (n_unclamped == 0) break;

        float remaining = mean_budget * n_heads - clamped_total;
        bool any_clamped = false;

        for (int i = 0; i < n_heads; i++) {
            if (clamped[i]) continue;

            float w = std::max(1e-12f, sensitivities[i]) / unclamped_total_sens;
            int b = (int)(remaining * w + 0.5f);
            b = std::max(min_budget, std::min(max_budget, b));

            if (b == min_budget || b == max_budget) {
                result.budgets[i] = b;
                clamped[i] = true;
                any_clamped = true;
            } else {
                result.budgets[i] = b;
            }
        }

        if (!any_clamped) break;
    }

    return result;
}

// Compute per-head sensitivity weights for global key scoring.
// This is a lighter-weight alternative to full budget allocation:
// instead of per-head budgets, weight each head's contribution to
// the global importance score differently.
//
//   sensitivities: [n_heads] sensitivity scores
//   n_heads:       number of heads
//   weights_out:   [n_heads] output weights (sum to n_heads)
//
// More sensitive heads get higher weight in the global score,
// so their important keys are more likely to be kept.
static void compute_sensitivity_weights(
        const float * sensitivities, int n_heads,
        float * weights_out) {

    if (n_heads <= 0) return;

    // Use sqrt of sensitivity for softer weighting (prevents extreme ratios)
    float total = 0.0f;
    for (int i = 0; i < n_heads; i++) {
        weights_out[i] = sqrtf(std::max(1e-12f, sensitivities[i]));
        total += weights_out[i];
    }

    // Normalize so weights sum to n_heads (preserving the average)
    float scale = (float)n_heads / (total + 1e-12f);
    for (int i = 0; i < n_heads; i++) {
        weights_out[i] *= scale;
    }
}
