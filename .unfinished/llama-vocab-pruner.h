#pragma once

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_set>
#include <vector>

// Speculative vocabulary pruning for sublinear lm_head computation.
//
// Maintains a "hot set" of tokens likely to be sampled. Computes logits only
// for the hot set first, then uses Cauchy-Schwarz bounds to decide whether
// the full computation is needed.
//
// Mathematical guarantee:
//   For row i of lm_head: |logit_i| <= ||row_i|| * ||hidden||
//   Precompute max_norm_outside = max(||row_i|| for i NOT in hot_set).
//   If top_hot_logit > max_norm_outside * ||hidden||, the hot set contains
//   the true argmax.

struct llama_vocab_pruner {
    static constexpr int DEFAULT_HOT_SET_SIZE   = 4096;
    static constexpr int DEFAULT_RECENT_SIZE    = 8192;
    static constexpr int REBUILD_INTERVAL       = 64;

    // Sorted token IDs in the hot set
    std::vector<llama_token> hot_set;

    // Precomputed L2 norms of each row of the output weight matrix
    std::vector<float> output_row_norms;

    // Max norm among tokens NOT in the hot set
    float max_norm_outside_hot_set = 0.0f;

    int hot_set_capacity = DEFAULT_HOT_SET_SIZE;
    bool norms_initialized = false;
    bool enabled = false;  // must be enabled via LLAMA_VOCAB_PRUNE=1 env var

    // Ring buffer of recently-seen top-K tokens
    std::vector<llama_token> recent_tokens;
    int recent_idx = 0;
    int recent_capacity = DEFAULT_RECENT_SIZE;

    // Counter for rebuild scheduling
    int tokens_since_rebuild = 0;

    // Stats
    int64_t n_pruned_hits  = 0;  // times hot set was sufficient
    int64_t n_pruned_misses = 0; // times full compute was needed
    int64_t n_total_decodes = 0;

    void update(llama_token token) {
        if (!enabled) return;

        if ((int)recent_tokens.size() < recent_capacity) {
            recent_tokens.push_back(token);
        } else {
            recent_tokens[recent_idx] = token;
        }
        recent_idx = (recent_idx + 1) % recent_capacity;
        tokens_since_rebuild++;
    }

    bool needs_rebuild() const {
        return tokens_since_rebuild >= REBUILD_INTERVAL;
    }

    void rebuild_hot_set(int n_vocab) {
        if (!enabled || !norms_initialized) return;

        tokens_since_rebuild = 0;

        // Count frequency of each token in the recent buffer
        std::vector<int32_t> freq(n_vocab, 0);
        for (const auto & tok : recent_tokens) {
            if (tok >= 0 && tok < n_vocab) {
                freq[tok]++;
            }
        }

        // Build candidate list: (frequency, token_id) sorted descending
        std::vector<std::pair<int32_t, llama_token>> candidates;
        candidates.reserve(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            if (freq[i] > 0) {
                candidates.push_back({freq[i], (llama_token)i});
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });

        // Take top hot_set_capacity tokens, plus always include high-norm tokens
        std::unordered_set<llama_token> hot_tokens;

        // Phase 1: add frequent tokens
        for (int i = 0; i < (int)candidates.size() && (int)hot_tokens.size() < hot_set_capacity; i++) {
            hot_tokens.insert(candidates[i].second);
        }

        // Phase 2: if we still have room, add highest-norm tokens (they're most dangerous to miss)
        if ((int)hot_tokens.size() < hot_set_capacity) {
            std::vector<std::pair<float, llama_token>> norm_sorted;
            norm_sorted.reserve(n_vocab);
            for (int i = 0; i < n_vocab; i++) {
                norm_sorted.push_back({output_row_norms[i], (llama_token)i});
            }
            std::sort(norm_sorted.begin(), norm_sorted.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });

            for (const auto & [norm, tok] : norm_sorted) {
                if ((int)hot_tokens.size() >= hot_set_capacity) break;
                hot_tokens.insert(tok);
            }
        }

        // Phase 3: if still room, fill with sequential tokens (common for BPE vocabs)
        for (int i = 0; (int)hot_tokens.size() < hot_set_capacity && i < n_vocab; i++) {
            hot_tokens.insert((llama_token)i);
        }

        // Convert to sorted vector
        hot_set.assign(hot_tokens.begin(), hot_tokens.end());
        std::sort(hot_set.begin(), hot_set.end());

        // Recompute max norm outside hot set
        compute_max_norm_outside(n_vocab);
    }

    void compute_max_norm_outside(int n_vocab) {
        max_norm_outside_hot_set = 0.0f;

        // hot_set is sorted, so we can use binary search
        for (int i = 0; i < n_vocab; i++) {
            if (!std::binary_search(hot_set.begin(), hot_set.end(), (llama_token)i)) {
                max_norm_outside_hot_set = std::max(max_norm_outside_hot_set, output_row_norms[i]);
            }
        }
    }

    // Initialize with a default hot set (first N tokens) before any sampling data
    void init_default_hot_set(int n_vocab) {
        hot_set.resize(std::min(hot_set_capacity, n_vocab));
        for (int i = 0; i < (int)hot_set.size(); i++) {
            hot_set[i] = (llama_token)i;
        }

        if (norms_initialized) {
            compute_max_norm_outside(n_vocab);
        }
    }

    // Check if the hot set result is sufficient (Cauchy-Schwarz bound)
    // Returns true if we can skip the full computation
    bool check_confidence(float top_hot_logit, float hidden_norm) const {
        if (!enabled || hot_set.empty()) return false;
        return top_hot_logit > max_norm_outside_hot_set * hidden_norm;
    }

    float get_hit_rate() const {
        if (n_total_decodes == 0) return 0.0f;
        return (float)n_pruned_hits / (float)n_total_decodes;
    }
};
