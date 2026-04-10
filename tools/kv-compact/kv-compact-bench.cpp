// KV Cache Compaction Perplexity Benchmark
//
// Measures the impact of Attention Matching compaction on next-token prediction quality.
//
// Approach:
//   1. Fill a large context window with text (prefill)
//   2. Compact the KV cache at a specified ratio
//   3. Continue processing more text and measure per-token log-likelihood
//   4. Compare against baseline (no compaction) to quantify quality loss
//
// This differs from standard perplexity (which clears KV per chunk) because compaction
// is specifically about preserving quality over a long, continuous context.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>

static void print_usage(int argc, char ** argv) {
    (void) argc;
    LOG("\nKV Cache Compaction - Perplexity Benchmark\n\n");
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("  -m  MODEL         path to model file\n");
    LOG("  -f  FILE          text file for perplexity evaluation\n");
    LOG("  -c  N             context size (default: 2048)\n");
    LOG("  --compact-ratio R compaction ratio - fraction to KEEP (default: 0.5)\n");
    LOG("                    e.g. 0.5 = keep 50%% = 2x compression\n");
    LOG("                         0.2 = keep 20%% = 5x compression\n");
    LOG("                         0.1 = keep 10%% = 10x compression\n");
    LOG("                         0.02 = keep 2%% = 50x compression\n");
    LOG("  --compact-at N    compact after N tokens of prefill (default: n_ctx/2)\n");
    LOG("  --eval-tokens N   number of tokens to evaluate after compaction (default: 256)\n");
    LOG("  --no-compact      run baseline without compaction (for comparison)\n");
    LOG("  --use-repeat-prefill  use repeat-prefill Q vectors (better quality)\n");
    LOG("  --capture-q       enable Q capture during prefill\n");
    LOG("\n");
}

struct bench_params {
    float compact_ratio    = 0.5f;
    int   compact_at       = -1;   // -1 = auto (n_ctx/2)
    int   eval_tokens      = 256;
    bool  no_compact       = false;
    bool  use_repeat_prefill = false;
    bool  capture_q        = false;
    bool  evict_only       = false; // simple token eviction via seq_rm (baseline comparison)
};

// Process a batch of tokens through the model, collecting logits for specified positions
static bool process_batch(llama_context * ctx, llama_batch & batch) {
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("Failed to decode batch\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    bench_params bparams;

    // Pre-parse custom args and strip them before passing to common_params_parse
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--compact-ratio") == 0 && i + 1 < argc) {
            bparams.compact_ratio = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--compact-at") == 0 && i + 1 < argc) {
            bparams.compact_at = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--eval-tokens") == 0 && i + 1 < argc) {
            bparams.eval_tokens = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-compact") == 0) {
            bparams.no_compact = true;
        } else if (strcmp(argv[i], "--use-repeat-prefill") == 0) {
            bparams.use_repeat_prefill = true;
        } else if (strcmp(argv[i], "--capture-q") == 0) {
            bparams.capture_q = true;
        } else if (strcmp(argv[i], "--evict-only") == 0) {
            bparams.evict_only = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    int filtered_argc = (int) filtered_argv.size();

    // Parse standard params (with custom args stripped)
    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_PERPLEXITY, print_usage)) {
        return 1;
    }

    if (bparams.compact_ratio <= 0.0f || bparams.compact_ratio >= 1.0f) {
        LOG_ERR("compact-ratio must be between 0 and 1 (exclusive)\n");
        return 1;
    }

    common_init();

    // Initialize
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_context * ctx   = llama_init->context();
    llama_model   * model = llama_init->model();

    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_ctx   = llama_n_ctx(ctx);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = llama_n_batch(ctx);

    // Auto-set compact_at
    if (bparams.compact_at < 0) {
        bparams.compact_at = n_ctx / 2;
    }

    LOG_INF("=== KV Cache Compaction Perplexity Benchmark ===\n");
    LOG_INF("Context: %d, Batch: %d, Vocab: %d\n", n_ctx, n_batch, n_vocab);
    if (bparams.no_compact) {
        LOG_INF("Mode: BASELINE (no compaction)\n");
    } else {
        LOG_INF("Mode: COMPACT (ratio=%.2f, keep %.0f%%, %.1fx compression)\n",
                bparams.compact_ratio, bparams.compact_ratio * 100.0f,
                1.0f / bparams.compact_ratio);
        LOG_INF("Compact after: %d tokens\n", bparams.compact_at);
    }
    LOG_INF("Eval tokens: %d\n", bparams.eval_tokens);

    // Tokenize input
    std::string text;
    if (!params.prompt.empty()) {
        text = params.prompt;
    } else {
        LOG_ERR("No input text. Use -p or -f to specify input.\n");
        return 1;
    }

    std::vector<llama_token> tokens = common_tokenize(vocab, text, true, false);
    const int n_tokens = (int) tokens.size();

    const int total_needed = bparams.compact_at + bparams.eval_tokens;
    if (n_tokens < total_needed) {
        LOG_ERR("Input too short: %d tokens, need at least %d (compact_at=%d + eval=%d)\n",
                n_tokens, total_needed, bparams.compact_at, bparams.eval_tokens);
        LOG_ERR("Provide more text via -f or -p, or reduce --compact-at / --eval-tokens\n");
        return 1;
    }

    if (bparams.compact_at + bparams.eval_tokens > n_ctx) {
        LOG_ERR("compact_at (%d) + eval_tokens (%d) > n_ctx (%d)\n",
                bparams.compact_at, bparams.eval_tokens, n_ctx);
        return 1;
    }

    LOG_INF("Input: %d tokens total, using first %d\n", n_tokens, total_needed);

    // Enable Q capture if requested
    if (bparams.capture_q && !bparams.no_compact) {
        llama_kv_cache_capture_q(ctx, true);
        LOG_INF("Q capture enabled\n");
    }

    // ---- Phase 1: Prefill ----
    LOG_INF("\n--- Phase 1: Prefill (%d tokens) ---\n", bparams.compact_at);

    auto t_start = std::chrono::high_resolution_clock::now();

    llama_batch batch = llama_batch_init(std::max(n_batch, bparams.compact_at), 0, 1);

    // Process prefill in batches
    for (int i = 0; i < bparams.compact_at; i += n_batch) {
        const int batch_size = std::min(n_batch, bparams.compact_at - i);

        common_batch_clear(batch);
        for (int j = 0; j < batch_size; j++) {
            common_batch_add(batch, tokens[i + j], i + j, {0}, false);
        }
        // Request logits for the last token of the last prefill batch
        if (i + batch_size >= bparams.compact_at) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        if (!process_batch(ctx, batch)) {
            llama_batch_free(batch);
            return 1;
        }
    }

    auto t_prefill = std::chrono::high_resolution_clock::now();
    float prefill_time = std::chrono::duration<float>(t_prefill - t_start).count();
    LOG_INF("Prefill: %.2f s (%.0f t/s)\n", prefill_time, bparams.compact_at / prefill_time);

    // ---- Phase 2: Compact (optional) ----
    int n_after_compact = bparams.compact_at;

    if (!bparams.no_compact) {
        if (bparams.evict_only) {
            // Simple token eviction: keep first few (sink) + last half + uniform sample
            LOG_INF("\n--- Phase 2: Simple token eviction (seq_rm) ---\n");

            auto t_compact_start = std::chrono::high_resolution_clock::now();

            n_after_compact = std::max(1, (int)(bparams.compact_at * bparams.compact_ratio));
            const int n_sink = std::min(4, n_after_compact / 4);
            const int n_recent = n_after_compact / 2;
            const int n_middle = n_after_compact - n_sink - n_recent;

            std::vector<bool> keep(bparams.compact_at, false);

            // Keep sink tokens
            for (int i = 0; i < n_sink && i < bparams.compact_at; i++) keep[i] = true;
            // Keep recent tokens
            for (int i = std::max(0, bparams.compact_at - n_recent); i < bparams.compact_at; i++) keep[i] = true;
            // Keep uniformly sampled middle tokens
            if (n_middle > 0 && bparams.compact_at > n_sink + n_recent) {
                int mid_start = n_sink;
                int mid_end = bparams.compact_at - n_recent;
                float step = (float)(mid_end - mid_start) / (float)n_middle;
                for (int i = 0; i < n_middle; i++) {
                    int idx = mid_start + (int)(i * step);
                    if (idx < mid_end) keep[idx] = true;
                }
            }

            llama_memory_t mem = llama_get_memory(ctx);
            for (int i = bparams.compact_at - 1; i >= 0; i--) {
                if (!keep[i]) {
                    llama_memory_seq_rm(mem, 0, i, i + 1);
                }
            }

            // Count actually kept
            n_after_compact = 0;
            for (int i = 0; i < bparams.compact_at; i++) {
                if (keep[i]) n_after_compact++;
            }

            auto t_compact_end = std::chrono::high_resolution_clock::now();
            float compact_time = std::chrono::duration<float>(t_compact_end - t_compact_start).count();

            LOG_INF("Evicted: %d -> %d tokens (%.1fx) in %.2f s\n",
                    bparams.compact_at, n_after_compact,
                    (float) bparams.compact_at / n_after_compact, compact_time);
        } else {
            LOG_INF("\n--- Phase 2: Compacting KV cache (AM) ---\n");

            auto t_compact_start = std::chrono::high_resolution_clock::now();

            llama_compact_params cparams = llama_compact_params_default();
            cparams.target_ratio = bparams.compact_ratio;
            cparams.use_repeat_prefill = bparams.use_repeat_prefill;

            n_after_compact = llama_kv_cache_compact(ctx, 0, cparams);

            auto t_compact_end = std::chrono::high_resolution_clock::now();
            float compact_time = std::chrono::duration<float>(t_compact_end - t_compact_start).count();

            if (n_after_compact < 0) {
                LOG_ERR("Compaction failed!\n");
                llama_batch_free(batch);
                return 1;
            }

            LOG_INF("Compacted: %d -> %d tokens (%.1fx) in %.2f s\n",
                    bparams.compact_at, n_after_compact,
                    (float) bparams.compact_at / n_after_compact, compact_time);
        }
    } else {
        LOG_INF("\n--- Phase 2: Skipped (baseline mode) ---\n");
    }

    // ---- Phase 3: Evaluate perplexity on continuation tokens ----
    LOG_INF("\n--- Phase 3: Evaluating perplexity on %d continuation tokens ---\n", bparams.eval_tokens);

    auto t_eval_start = std::chrono::high_resolution_clock::now();

    double nll = 0.0;     // negative log-likelihood accumulator
    double nll2 = 0.0;    // for variance
    int count = 0;

    // Process evaluation tokens one batch at a time
    // We need to feed token[compact_at + i] at position compact_at + i
    // and check the logits against token[compact_at + i + 1]
    for (int i = 0; i < bparams.eval_tokens; i += n_batch) {
        const int batch_size = std::min(n_batch, bparams.eval_tokens - i);

        common_batch_clear(batch);
        for (int j = 0; j < batch_size; j++) {
            const int tok_idx = bparams.compact_at + i + j;
            // Position in the sequence: after compaction, positions are preserved
            // so we continue from where we left off
            common_batch_add(batch, tokens[tok_idx], tok_idx, {0}, true);
        }

        if (!process_batch(ctx, batch)) {
            LOG_ERR("Failed at eval token %d\n", i);
            break;
        }

        // Extract logits and compute log-likelihood
        for (int j = 0; j < batch_size; j++) {
            const int tok_idx = bparams.compact_at + i + j;
            if (tok_idx + 1 >= n_tokens) break;

            const float * logits = llama_get_logits_ith(ctx, j);
            if (!logits) continue;

            // Compute log-softmax for the target token
            const llama_token target = tokens[tok_idx + 1];

            // Find max for numerical stability
            float max_logit = -INFINITY;
            for (int v = 0; v < n_vocab; v++) {
                if (logits[v] > max_logit) max_logit = logits[v];
            }

            // Compute log-sum-exp
            double sum_exp = 0.0;
            for (int v = 0; v < n_vocab; v++) {
                sum_exp += exp((double)(logits[v] - max_logit));
            }
            double log_sum_exp = (double) max_logit + log(sum_exp);

            // Negative log-likelihood for this token
            double token_nll = log_sum_exp - (double) logits[target];
            nll += token_nll;
            nll2 += token_nll * token_nll;
            count++;
        }
    }

    auto t_eval_end = std::chrono::high_resolution_clock::now();
    float eval_time = std::chrono::duration<float>(t_eval_end - t_eval_start).count();

    // ---- Results ----
    LOG_INF("\n=== Results ===\n");

    if (count > 0) {
        double avg_nll = nll / count;
        double ppl = exp(avg_nll);
        double var = nll2 / count - avg_nll * avg_nll;
        double std_err = (var > 0 && count > 1) ? sqrt(var / (count - 1)) : 0.0;
        double ppl_err = std_err * ppl;  // delta method approximation

        LOG_INF("Tokens evaluated: %d\n", count);
        LOG_INF("Average NLL:      %.4f\n", avg_nll);
        LOG_INF("Perplexity:       %.4f +/- %.4f\n", ppl, ppl_err);
        LOG_INF("Eval time:        %.2f s (%.0f t/s)\n", eval_time, count / eval_time);

        if (!bparams.no_compact) {
            LOG_INF("Compression:      %.1fx (%d -> %d tokens)\n",
                    (float) bparams.compact_at / n_after_compact,
                    bparams.compact_at, n_after_compact);
        }

        // Machine-readable output line for scripting
        LOG("\n[RESULT] compact_ratio=%.4f compression=%.1fx ppl=%.4f ppl_err=%.4f nll=%.4f n_eval=%d n_prefill=%d n_after=%d\n",
            bparams.no_compact ? 1.0f : bparams.compact_ratio,
            bparams.no_compact ? 1.0f : (float) bparams.compact_at / n_after_compact,
            ppl, ppl_err, avg_nll, count, bparams.compact_at, n_after_compact);
    } else {
        LOG_ERR("No tokens evaluated!\n");
    }

    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}
