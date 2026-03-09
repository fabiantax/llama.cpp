// Server-side KV cache compaction via attention-based importance scoring
//
// Replaces the simple context shift (discard first half) with intelligent
// token selection based on attention importance across all layers and heads.
//
// Algorithm:
//   1. Save KV cache state for the sequence
//   2. Parse state buffer to extract K/V data per layer/head
//   3. Score each token by max attention importance across layers × heads × reference queries
//   4. Force-select first n_keep tokens, then select best remaining by importance
//   5. Copy original V values at selected positions (simplified, no NNLS/LSQ)
//   6. Build compacted state with contiguous position remapping
//   7. Clear cache and reload compacted state
//
// Reference: "Fast KV Compaction via Attention Matching" (Zweiger et al., arXiv:2602.16284)

#pragma once

#include "llama.h"
#include "kv-compact-state.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

struct kv_compact_result {
    std::vector<int> selected_indices;  // [t] original token indices that survived
    int t;                              // number of tokens kept
    bool success;
};

// Run KV cache compaction on a single sequence
//
// ctx:           llama context
// seq_id:        sequence to compact
// n_keep:        number of tokens at the start to always keep (system prompt, etc.)
// compact_ratio: fraction of total tokens to keep (0.0-1.0)
//
// Returns: selected token indices and success flag
//          On failure, falls back to standard context shift behavior
static kv_compact_result server_kv_compact(
        llama_context * ctx,
        llama_seq_id seq_id,
        int n_keep,
        int n_total,
        float compact_ratio) {

    kv_compact_result result;
    result.success = false;
    result.t = 0;

    // 1. Save state
    const size_t state_size = llama_state_seq_get_size(ctx, seq_id);
    if (state_size == 0) {
        return result;
    }

    std::vector<uint8_t> state_buf(state_size);
    const size_t saved = llama_state_seq_get_data(ctx, state_buf.data(), state_buf.size(), seq_id);
    if (saved == 0) {
        return result;
    }

    // 2. Parse state (detect IMROPE for ext data)
    const llama_model * model_ptr = llama_get_model(ctx);
    const enum llama_rope_type rope_type = llama_model_rope_type(model_ptr);
    const uint32_t n_pos_per_embd = (rope_type == LLAMA_ROPE_TYPE_MROPE ||
                                     rope_type == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;

    parsed_kv_state kv_state;
    if (!kv_state.parse(state_buf.data(), saved, n_pos_per_embd)) {
        return result;
    }

    if (kv_state.n_stream == 0 || kv_state.streams[0].cell_count == 0) {
        return result;
    }

    const auto & sd = kv_state.streams[0];
    const int T = (int)sd.cell_count;

    if (T <= n_keep) {
        return result; // nothing to compact
    }

    // Get dimensions from parsed state (handles GQA correctly)
    const auto & ld0 = sd.layers[0];
    const int n_embd_k_gqa = ld0.n_embd_k_gqa();
    const int n_embd_v_gqa = ld0.n_embd_v_gqa_computed();

    // Infer n_head_kv from model (model_ptr already obtained above)
    const int n_head_kv = llama_model_n_head_kv(model_ptr);

    if (n_embd_k_gqa == 0 || n_head_kv == 0 || n_embd_k_gqa % n_head_kv != 0) {
        return result;
    }

    const int d_k = n_embd_k_gqa / n_head_kv;
    const int d_v = n_embd_v_gqa / n_head_kv;

    // 3. Compute target size
    int t = std::max(n_keep + 1, (int)(T * compact_ratio));
    t = std::min(t, T); // can't keep more than we have

    // 4. Compute importance scores using last quarter as reference queries
    const int n_ref = std::max(8, T / 4);
    const int ref_start = T - n_ref;
    const float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    std::vector<float> global_importance(T, 0.0f);

    // Force-select first n_keep by giving them max importance
    for (int i = 0; i < n_keep && i < T; i++) {
        global_importance[i] = 1e30f;
    }

    // Score remaining tokens across all layers × heads × reference queries
    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];

        for (int h = 0; h < n_head_kv; h++) {
            for (int qi = 0; qi < n_ref; qi++) {
                const float * q_row = ld.K.data() + (ref_start + qi) * n_embd_k_gqa + h * d_k;

                for (int ki = n_keep; ki < T; ki++) {
                    const float * k_row = ld.K.data() + ki * n_embd_k_gqa + h * d_k;
                    float dot = 0.0f;
                    for (int d = 0; d < d_k; d++) {
                        dot += q_row[d] * k_row[d];
                    }
                    float score = dot * inv_sqrt_dk;
                    if (score > global_importance[ki]) {
                        global_importance[ki] = score;
                    }
                }
            }
        }
    }

    // 5. Select top-t tokens
    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return global_importance[a] > global_importance[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());

    // 6. Build C_v — copy original V at selected positions (simplified, no NNLS/LSQ)
    std::vector<std::vector<std::vector<float>>> cv_all(sd.n_layer);

    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        cv_all[l].resize(n_head_kv);

        for (int h = 0; h < n_head_kv; h++) {
            cv_all[l][h].resize(t * d_v);
            for (int j = 0; j < t; j++) {
                const float * v_row = ld.V.data() + selected[j] * n_embd_v_gqa + h * d_v;
                memcpy(cv_all[l][h].data() + j * d_v, v_row, d_v * sizeof(float));
            }
        }
    }

    // 7. Remap positions to be contiguous (0..t-1) for server compatibility
    //    This matches the behavior of standard context shift where positions
    //    are shifted to be contiguous after discarding tokens.
    for (int j = 0; j < t; j++) {
        kv_state.streams[0].cells[selected[j]].pos = (int32_t)j;
    }

    // 8. Build compacted state buffer
    auto compacted_buf = build_compacted_state(kv_state, selected, cv_all, n_head_kv, d_k, d_v, n_pos_per_embd);
    if (compacted_buf.empty()) {
        return result;
    }

    // 9. Clear cache and reload compacted state
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, seq_id, -1, -1);

    const size_t loaded = llama_state_seq_set_data(ctx, compacted_buf.data(), compacted_buf.size(), seq_id);
    if (loaded == 0) {
        return result;
    }

    // Success
    result.selected_indices = selected;
    result.t = t;
    result.success = true;
    return result;
}
