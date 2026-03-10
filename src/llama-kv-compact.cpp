// KV Cache Compaction via Attention Matching
//
// Implements the "Highest Attention Keys" variant from:
//   "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026)
//   https://arxiv.org/abs/2602.16284
//
// Integrates the math primitives from tools/kv-compact/kv-compact-math.h
// with the KV cache infrastructure to perform in-place compaction.

#include "llama-kv-compact.h"

#include "llama-impl.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-hparams.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

// Include math primitives
#include "../tools/kv-compact/kv-compact-math.h"

// ============================================================================
// K/V data extraction helpers (read from ggml tensors)
// ============================================================================

static void read_k_head(const ggml_tensor * k_tensor, int head_idx, int n_embd_head_k,
                        int n_embd_k_gqa, int n_tokens, std::vector<float> & out) {
    out.resize(n_tokens * n_embd_head_k);

    const int head_offset = head_idx * n_embd_head_k;
    const size_t row_size = ggml_row_size(k_tensor->type, n_embd_k_gqa);

    if (k_tensor->type == GGML_TYPE_F32) {
        for (int i = 0; i < n_tokens; i++) {
            ggml_backend_tensor_get(k_tensor, out.data() + i * n_embd_head_k,
                                     i * row_size + head_offset * sizeof(float),
                                     n_embd_head_k * sizeof(float));
        }
    } else if (k_tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n_embd_head_k);
        for (int i = 0; i < n_tokens; i++) {
            ggml_backend_tensor_get(k_tensor, tmp.data(),
                                     i * row_size + head_offset * sizeof(ggml_fp16_t),
                                     n_embd_head_k * sizeof(ggml_fp16_t));
            for (int j = 0; j < n_embd_head_k; j++) {
                out[i * n_embd_head_k + j] = ggml_fp16_to_fp32(tmp[j]);
            }
        }
    } else {
        std::vector<float> full_row(n_embd_k_gqa);
        for (int i = 0; i < n_tokens; i++) {
            std::vector<uint8_t> raw(row_size);
            ggml_backend_tensor_get(k_tensor, raw.data(), i * row_size, row_size);
            ggml_get_type_traits(k_tensor->type)->to_float(raw.data(), full_row.data(), n_embd_k_gqa);
            memcpy(out.data() + i * n_embd_head_k, full_row.data() + head_offset, n_embd_head_k * sizeof(float));
        }
    }
}

static void read_v_head(const ggml_tensor * v_tensor, int head_idx, int n_embd_head_v,
                        int n_embd_v_gqa, int n_tokens, int kv_size, bool v_trans,
                        std::vector<float> & out) {
    out.resize(n_tokens * n_embd_head_v);

    if (!v_trans) {
        read_k_head(v_tensor, head_idx, n_embd_head_v, n_embd_v_gqa, n_tokens, out);
        return;
    }

    // V is transposed
    if (v_tensor->type == GGML_TYPE_F32) {
        for (int d = 0; d < n_embd_head_v; d++) {
            for (int t = 0; t < n_tokens; t++) {
                float val;
                size_t offset = ((size_t)(head_idx * n_embd_head_v + d) * kv_size + t) * sizeof(float);
                ggml_backend_tensor_get(v_tensor, &val, offset, sizeof(float));
                out[t * n_embd_head_v + d] = val;
            }
        }
    } else if (v_tensor->type == GGML_TYPE_F16) {
        for (int d = 0; d < n_embd_head_v; d++) {
            for (int t = 0; t < n_tokens; t++) {
                ggml_fp16_t val;
                size_t offset = ((size_t)(head_idx * n_embd_head_v + d) * kv_size + t) * sizeof(ggml_fp16_t);
                ggml_backend_tensor_get(v_tensor, &val, offset, sizeof(ggml_fp16_t));
                out[t * n_embd_head_v + d] = ggml_fp16_to_fp32(val);
            }
        }
    } else {
        LLAMA_LOG_ERROR("%s: unsupported V tensor type for transposed read: %s\n",
                        __func__, ggml_type_name(v_tensor->type));
        std::fill(out.begin(), out.end(), 0.0f);
    }
}

// ============================================================================
// Main compaction implementation
// ============================================================================

int32_t llama_kv_compact_impl(
        struct llama_context * ctx,
        llama_seq_id           seq_id [[maybe_unused]],
        llama_compact_params   params) {

    if (!ctx) {
        LLAMA_LOG_ERROR("%s: null context\n", __func__);
        return -1;
    }

    const auto & model   = ctx->get_model();
    const auto & hparams = model.hparams;
    auto * memory = ctx->get_memory();

    // cast to llama_kv_cache
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        LLAMA_LOG_ERROR("%s: memory type is not llama_kv_cache — compaction not supported\n", __func__);
        return -1;
    }

    const float target_ratio = params.target_ratio;
    if (target_ratio <= 0.0f || target_ratio > 1.0f) {
        LLAMA_LOG_ERROR("%s: target_ratio must be in (0, 1], got %f\n", __func__, target_ratio);
        return -1;
    }

    // determine stream for this sequence
    // for now, assume stream 0 (unified cache)
    const uint32_t stream = 0;
    const uint32_t kv_size = kv->get_kv_size();

    // count active cells and collect their indices
    std::vector<uint32_t> active_cells;
    active_cells.reserve(kv_size);
    for (uint32_t i = 0; i < kv_size; ++i) {
        if (!kv->is_cell_empty(stream, i)) {
            active_cells.push_back(i);
        }
    }

    const uint32_t n_active = active_cells.size();
    if (n_active == 0) {
        LLAMA_LOG_WARN("%s: no active cells in cache\n", __func__);
        return 0;
    }

    const uint32_t n_keep = std::max(1u, (uint32_t)(n_active * target_ratio));

    LLAMA_LOG_INFO("%s: compacting KV cache: %u -> %u tokens (ratio %.2f, Q source: %s)\n",
                   __func__, n_active, n_keep, target_ratio,
                   (params.use_repeat_prefill && kv->has_captured_q(0)) ? "captured" : "K-proxy");

    if (n_keep >= n_active) {
        LLAMA_LOG_INFO("%s: n_keep >= n_active, nothing to compact\n", __func__);
        return n_active;
    }

    const uint32_t n_embd_head_k = hparams.n_embd_head_k;
    const uint32_t n_embd_head_v = hparams.n_embd_head_v;
    const uint32_t n_layer       = hparams.n_layer;
    const bool     v_trans       = kv->get_v_trans();

    // determine reference query source
    const bool use_captured_q = params.use_repeat_prefill && kv->has_captured_q(0);
    if (params.use_repeat_prefill && !use_captured_q) {
        LLAMA_LOG_WARN("%s: repeat-prefill requested but no captured Q vectors available, falling back to K-proxy\n", __func__);
    }

    // n_ref_q is per-head and may vary when using captured Q; set default for K-proxy mode
    const int n_ref_q_default = params.n_ref_queries > 0 ? params.n_ref_queries : std::min((int)n_active, 64);

    // clear any existing bias
    kv->clear_compaction_bias();

    // ========================================================================
    // Non-uniform budget support
    // ========================================================================
    //
    // When use_nonuniform_budgets is enabled, each (layer, head) gets a
    // sensitivity-weighted contribution to the global key score. This causes
    // keys important to sensitive heads to be preferentially kept.
    //
    // The approach:
    //   1. First pass: compute per-head key scores and sensitivity estimates
    //   2. Compute sensitivity weights from reconstruction error proxies
    //   3. Second pass (or reweight): weight global scores by sensitivity
    //   4. Select global top-n_keep using weighted scores
    //
    // When disabled, all heads contribute equally (original behavior).

    const bool use_nonuniform = params.use_nonuniform_budgets;

    // Count total heads across all layers for sensitivity indexing
    uint32_t total_heads = 0;
    std::vector<uint32_t> head_layer_map;   // layer index for each global head
    std::vector<uint32_t> head_index_map;   // head-within-layer for each global head
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!hparams.has_kv(il)) continue;
        uint32_t n_head_kv_l = hparams.n_head_kv(il);
        for (uint32_t h = 0; h < n_head_kv_l; ++h) {
            head_layer_map.push_back(il);
            head_index_map.push_back(h);
            total_heads++;
        }
    }

    // Per-head key scores: [total_heads][n_active]
    // We always compute these, then either weight or sum uniformly
    std::vector<std::vector<float>> per_head_scores(total_heads, std::vector<float>(n_active, 0.0f));

    // Per-head K data cache (reused for AM fitting later)
    struct head_kv_data {
        std::vector<float> K;       // [n_active * n_embd_head_k]
        std::vector<float> Q_ref;   // [n_ref_q * n_embd_head_k]
        int n_ref_q = 0;
    };
    std::vector<head_kv_data> head_data(total_heads);

    // First pass: compute per-head key importance scores
    uint32_t gh = 0; // global head index
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!hparams.has_kv(il)) continue;

        ggml_tensor * k_tensor = kv->get_k_raw(il);
        if (!k_tensor) { gh += hparams.n_head_kv(il); continue; }

        const uint32_t n_embd_k_gqa = k_tensor->ne[0];
        const uint32_t n_head_kv_l  = hparams.n_head_kv(il);

        for (uint32_t h = 0; h < n_head_kv_l; ++h, ++gh) {
            auto & hd = head_data[gh];

            // read K data for active cells
            hd.K.resize(n_active * n_embd_head_k);
            for (uint32_t i = 0; i < n_active; ++i) {
                const size_t row_size = ggml_row_size(k_tensor->type, n_embd_k_gqa);
                const uint32_t head_offset = h * n_embd_head_k;

                if (k_tensor->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(k_tensor, hd.K.data() + i * n_embd_head_k,
                                             active_cells[i] * row_size + head_offset * sizeof(float),
                                             n_embd_head_k * sizeof(float));
                } else if (k_tensor->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> tmp(n_embd_head_k);
                    ggml_backend_tensor_get(k_tensor, tmp.data(),
                                             active_cells[i] * row_size + head_offset * sizeof(ggml_fp16_t),
                                             n_embd_head_k * sizeof(ggml_fp16_t));
                    for (uint32_t j = 0; j < n_embd_head_k; ++j) {
                        hd.K[i * n_embd_head_k + j] = ggml_fp16_to_fp32(tmp[j]);
                    }
                } else {
                    std::vector<float> full_row(n_embd_k_gqa);
                    std::vector<uint8_t> raw(row_size);
                    ggml_backend_tensor_get(k_tensor, raw.data(), active_cells[i] * row_size, row_size);
                    ggml_get_type_traits(k_tensor->type)->to_float(raw.data(), full_row.data(), n_embd_k_gqa);
                    memcpy(hd.K.data() + i * n_embd_head_k, full_row.data() + head_offset,
                           n_embd_head_k * sizeof(float));
                }
            }

            // get reference queries: captured Q vectors or K-proxy fallback
            if (use_captured_q && kv->has_captured_q(il)) {
                uint32_t n_captured = 0;
                const float * captured = kv->get_captured_q(il, h, &n_captured);
                hd.n_ref_q = (int)n_captured;
                hd.Q_ref.assign(captured, captured + hd.n_ref_q * n_embd_head_k);
            } else {
                hd.n_ref_q = n_ref_q_default;
                hd.Q_ref.resize(hd.n_ref_q * n_embd_head_k);
                for (int q = 0; q < hd.n_ref_q; ++q) {
                    int src_idx = (int)(q * (n_active - 1) / (hd.n_ref_q - 1 + 1e-9f));
                    memcpy(hd.Q_ref.data() + q * n_embd_head_k,
                           hd.K.data() + src_idx * n_embd_head_k,
                           n_embd_head_k * sizeof(float));
                }
            }

            // compute attention scores and key importance
            const float inv_sqrt_dk = 1.0f / sqrtf((float)n_embd_head_k);
            std::vector<float> scores(hd.n_ref_q * n_active);
            mat_mul_ABt(hd.Q_ref.data(), hd.K.data(), scores.data(), hd.n_ref_q, n_active, n_embd_head_k);
            for (int i = 0; i < hd.n_ref_q * (int)n_active; i++) {
                scores[i] *= inv_sqrt_dk;
            }

            // softmax
            std::vector<float> attn(scores);
            softmax_rows(attn.data(), hd.n_ref_q, n_active);

            // max attention weight per key across queries
            for (uint32_t j = 0; j < n_active; ++j) {
                float max_w = 0.0f;
                for (int q = 0; q < hd.n_ref_q; ++q) {
                    float w = attn[q * n_active + j];
                    if (w > max_w) max_w = w;
                }
                per_head_scores[gh][j] = max_w;
            }
        }
    }

    // Compute sensitivity weights
    std::vector<float> sensitivity_weights(total_heads, 1.0f);

    if (use_nonuniform && total_heads > 1) {
        // Estimate per-head sensitivity using attention entropy as proxy
        // Heads with concentrated attention (low entropy) are more sensitive
        // to losing their important keys. Heads with diffuse attention are
        // more robust to compression.
        //
        // We use the variance of per-key importance scores as the sensitivity
        // metric: higher variance = some keys are much more important = sensitive.

        std::vector<float> sensitivities(total_heads);
        for (uint32_t i = 0; i < total_heads; ++i) {
            float mean = 0.0f;
            for (uint32_t j = 0; j < n_active; ++j) {
                mean += per_head_scores[i][j];
            }
            mean /= (float)n_active;

            float var = 0.0f;
            for (uint32_t j = 0; j < n_active; ++j) {
                float d = per_head_scores[i][j] - mean;
                var += d * d;
            }
            var /= (float)n_active;
            sensitivities[i] = var;
        }

        compute_sensitivity_weights(sensitivities.data(), total_heads, sensitivity_weights.data());

        LLAMA_LOG_INFO("%s: non-uniform budgets enabled, sensitivity weights computed for %u heads\n",
                       __func__, total_heads);

        // Log a summary of weight distribution
        float w_min = sensitivity_weights[0], w_max = sensitivity_weights[0];
        for (uint32_t i = 1; i < total_heads; ++i) {
            if (sensitivity_weights[i] < w_min) w_min = sensitivity_weights[i];
            if (sensitivity_weights[i] > w_max) w_max = sensitivity_weights[i];
        }
        LLAMA_LOG_INFO("%s: sensitivity weight range: [%.3f, %.3f] (1.0 = uniform)\n",
                       __func__, w_min, w_max);
    }

    // Aggregate weighted global key scores
    std::vector<float> global_key_scores(n_active, 0.0f);
    for (uint32_t i = 0; i < total_heads; ++i) {
        float w = sensitivity_weights[i];
        for (uint32_t j = 0; j < n_active; ++j) {
            global_key_scores[j] += per_head_scores[i][j] * w;
        }
    }

    // select top-n_keep keys by aggregated importance
    std::vector<uint32_t> rank_indices(n_active);
    std::iota(rank_indices.begin(), rank_indices.end(), 0);
    std::partial_sort(rank_indices.begin(), rank_indices.begin() + n_keep, rank_indices.end(),
                      [&](uint32_t a, uint32_t b) {
                          return global_key_scores[a] > global_key_scores[b];
                      });

    // sort kept indices for cache locality
    std::vector<uint32_t> kept_active(rank_indices.begin(), rank_indices.begin() + n_keep);
    std::sort(kept_active.begin(), kept_active.end());

    // map back to cell indices
    std::vector<uint32_t> kept_cells(n_keep);
    for (uint32_t i = 0; i < n_keep; ++i) {
        kept_cells[i] = active_cells[kept_active[i]];
    }

    // now for each layer/head, run full AM: compute beta and C_v for the kept positions
    // Reuse cached K and Q_ref data from the scoring pass above
    gh = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (!hparams.has_kv(il)) {
            continue;
        }

        ggml_tensor * k_tensor = kv->get_k_raw(il);
        ggml_tensor * v_tensor = kv->get_v_raw(il);
        if (!k_tensor) { gh += hparams.n_head_kv(il); continue; }

        const uint32_t n_embd_v_gqa = v_tensor ? v_tensor->ne[0] : 0;
        const uint32_t n_head_kv_l  = hparams.n_head_kv(il);

        for (uint32_t h = 0; h < n_head_kv_l; ++h, ++gh) {
            // K and Q_ref already cached from scoring pass
            const auto & hd = head_data[gh];
            const float * K_all = hd.K.data();
            const float * Q_ref_am = hd.Q_ref.data();
            int n_ref_q_am = hd.n_ref_q;

            // Read V data (not cached in scoring pass)
            std::vector<float> V_all;
            if (v_tensor) {
                read_v_head(v_tensor, h, n_embd_head_v, n_embd_v_gqa, n_active, kv_size, v_trans, V_all);
            }

            // run full AM compaction on this head
            compacted_head result = compact_head_highest_attn(
                K_all,
                V_all.empty() ? nullptr : V_all.data(),
                Q_ref_am,
                n_active, n_ref_q_am, n_embd_head_k,
                v_tensor ? n_embd_head_v : 0,
                n_keep);

            // set beta values for kept positions
            for (uint32_t i = 0; i < n_keep; ++i) {
                kv->set_compaction_bias(il, h, kept_cells[i], result.beta[i]);
            }

            // write K data for kept positions (just the selected rows)
            std::vector<float> K_kept(n_keep * n_embd_head_k);
            for (uint32_t i = 0; i < n_keep; ++i) {
                memcpy(K_kept.data() + i * n_embd_head_k,
                       K_all + kept_active[i] * n_embd_head_k,
                       n_embd_head_k * sizeof(float));
            }
            kv->write_k_compact(il, h, K_kept.data(), kept_cells.data(), n_keep);

            // write C_v for kept positions
            if (v_tensor && !result.C_v.empty()) {
                kv->write_v_compact(il, h, result.C_v.data(), kept_cells.data(), n_keep);
            }
        }
    }

    // update cell metadata: evict non-kept cells
    kv->compact_cells(kept_cells.data(), n_keep, stream);

    LLAMA_LOG_INFO("%s: compaction complete: %u -> %u tokens\n", __func__, n_active, n_keep);

    return (int32_t)n_keep;
}
