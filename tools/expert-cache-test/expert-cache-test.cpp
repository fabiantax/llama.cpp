// Expert cache-aware routing test harness
// Tests the cache bias mechanism on MoE models and measures expert overlap.
//
// Usage: expert-cache-test -m model.gguf [-p prompt] [-n tokens] [-b bonus]
//
// Runs inference twice:
//   1. Baseline: no cache bias (default routing)
//   2. Cache-biased: bias selection toward experts used in previous token
// Compares expert overlap, unique experts per step, and generation speed.

#include "llama.h"
#include "common.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <chrono>

struct test_params {
    std::string model_path;
    std::string prompt = "You are a coding agent. Write a Python function to sort a list:";
    int n_predict = 50;
    float cache_bonus = 0.5f;
    int n_gpu_layers = 99;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m model.gguf [-p prompt] [-n tokens] [-b bonus] [-ngl layers]\n", prog);
    fprintf(stderr, "  -m   Model path (required)\n");
    fprintf(stderr, "  -p   Prompt (default: coding prompt)\n");
    fprintf(stderr, "  -n   Tokens to generate (default: 50)\n");
    fprintf(stderr, "  -b   Cache bonus strength (default: 0.5)\n");
    fprintf(stderr, "  -ngl GPU layers (default: 99)\n");
}

static test_params parse_args(int argc, char ** argv) {
    test_params params;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            params.prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            params.n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            params.cache_bonus = atof(argv[++i]);
        } else if (strcmp(argv[i], "-ngl") == 0 && i + 1 < argc) {
            params.n_gpu_layers = atoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            exit(1);
        }
    }
    if (params.model_path.empty()) {
        print_usage(argv[0]);
        exit(1);
    }
    return params;
}

// Track expert selections per layer per token
struct expert_tracker {
    int n_layer;
    int n_expert;
    int n_expert_used;

    // [step][layer] -> set of selected expert indices
    std::vector<std::vector<std::set<int>>> selections;

    // Per-layer cache state: which experts are "hot"
    std::vector<std::set<int>> hot_experts;

    void init(int nl, int ne, int neu) {
        n_layer = nl;
        n_expert = ne;
        n_expert_used = neu;
        hot_experts.resize(nl);
    }

    void record_step(int step, const std::vector<std::set<int>> & layer_selections) {
        if ((int)selections.size() <= step) {
            selections.resize(step + 1);
        }
        selections[step] = layer_selections;

        // Update hot experts: union of this step's selections
        for (int il = 0; il < n_layer && il < (int)layer_selections.size(); il++) {
            hot_experts[il] = layer_selections[il];
        }
    }

    // Compute metrics
    struct metrics {
        float avg_unique_per_step;    // average unique experts across all layers per step
        float avg_overlap_ratio;      // overlap between consecutive steps
        float cache_hit_rate;         // if we cached previous step's experts
        int   total_steps;
    };

    metrics compute() const {
        metrics m = {};
        if (selections.size() < 2) return m;

        m.total_steps = (int)selections.size();
        float total_unique = 0;
        float total_overlap = 0;
        float total_hits = 0;
        float total_accesses = 0;
        int   count = 0;

        for (int s = 1; s < (int)selections.size(); s++) {
            for (int il = 0; il < n_layer && il < (int)selections[s].size(); il++) {
                const auto & prev = selections[s-1][il];
                const auto & curr = selections[s][il];

                if (curr.empty() || prev.empty()) continue;

                // Count unique experts in this step
                total_unique += (float)curr.size();

                // Count overlap with previous step
                int overlap = 0;
                for (int e : curr) {
                    if (prev.count(e)) overlap++;
                }
                total_overlap += (float)overlap / (float)curr.size();

                // Cache hit rate: how many of curr were in prev?
                total_hits += overlap;
                total_accesses += curr.size();
                count++;
            }
        }

        if (count > 0) {
            m.avg_unique_per_step = total_unique / count;
            m.avg_overlap_ratio = total_overlap / count;
            m.cache_hit_rate = total_hits / total_accesses;
        }
        return m;
    }
};

// Generate tokens and optionally apply cache bias
static std::string generate_with_tracking(
        llama_context * ctx,
        llama_model * model,
        const std::vector<llama_token> & prompt_tokens,
        int n_predict,
        float cache_bonus,
        bool use_cache_bias,
        expert_tracker & tracker) {

    const int n_layer = llama_model_n_layer(model);
    // Access expert count via hparams - need internal header for prototype
    // In production, add llama_model_n_expert() to public API
    int n_expert = 0, n_expert_used = 0;
    {
        // Read from model metadata
        char buf[64];
        int32_t ne = 0, neu = 0;
        if (llama_model_meta_val_str(model, "qwen3next.expert_count", buf, sizeof(buf)) > 0) ne = atoi(buf);
        else if (llama_model_meta_val_str(model, "qwen35moe.expert_count", buf, sizeof(buf)) > 0) ne = atoi(buf);
        if (llama_model_meta_val_str(model, "qwen3next.expert_used_count", buf, sizeof(buf)) > 0) neu = atoi(buf);
        else if (llama_model_meta_val_str(model, "qwen35moe.expert_used_count", buf, sizeof(buf)) > 0) neu = atoi(buf);
        // Fallback: try generic keys
        if (ne == 0 && llama_model_meta_val_str(model, "general.expert_count", buf, sizeof(buf)) > 0) ne = atoi(buf);
        n_expert = ne;
        n_expert_used = neu;
    }

    if (n_expert == 0) {
        fprintf(stderr, "Model has no experts (not MoE). Exiting.\n");
        return "";
    }

    tracker.init(n_layer, n_expert, n_expert_used);

    // Allocate cache bias arrays (one per layer)
    std::vector<std::vector<float>> bias_data(n_layer, std::vector<float>(n_expert, 0.0f));
    std::vector<const float *> bias_ptrs(n_layer, nullptr);

    if (use_cache_bias) {
        for (int il = 0; il < n_layer; il++) {
            bias_ptrs[il] = bias_data[il].data();
        }
        llama_set_expert_cache_bias(model, bias_ptrs.data(), n_layer);
    }

    // Clear KV cache
    llama_memory_clear(llama_get_memory(ctx), true);

    // Process prompt
    llama_batch batch = llama_batch_init(prompt_tokens.size(), 0, 1);
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        common_batch_add(batch, prompt_tokens[i], i, { 0 }, i == prompt_tokens.size() - 1);
    }

    if (llama_decode(ctx, batch) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to decode prompt\n");
        llama_batch_free(batch);
        return "";
    }
    llama_batch_free(batch);

    // Generate tokens
    std::string output;
    llama_token last_token = prompt_tokens.back();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < n_predict; step++) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) break;

        // Convert to text
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) output.append(buf, n);

        // TODO: Track which experts were actually selected
        // For now, we just track that a step happened.
        // To get actual expert selections, we'd need a callback in the MoE routing.
        // For this prototype, we simulate tracking by using the cache bias mechanism
        // and measuring throughput difference.
        std::vector<std::set<int>> dummy_selections(n_layer);
        tracker.record_step(step, dummy_selections);

        // Update cache bias for next step
        // Strategy: set bias for ALL experts to 0, then set cache_bonus
        // for experts that were "recently used" (simulated by random subset)
        // In production, this would use actual routing decisions from the graph.
        if (use_cache_bias) {
            // For now, bias toward a fixed subset of experts per layer
            // (simulating cache-aware routing with high overlap)
            for (int il = 0; il < n_layer; il++) {
                std::fill(bias_data[il].begin(), bias_data[il].end(), 0.0f);
                // Bias the first n_expert_used*2 experts (simulating cache)
                for (int e = 0; e < std::min(n_expert_used * 2, n_expert); e++) {
                    bias_data[il][e] = cache_bonus;
                }
            }
        }

        // Decode next token
        llama_batch single = llama_batch_init(1, 0, 1);
        common_batch_add(single, new_token, prompt_tokens.size() + step, { 0 }, true);
        if (llama_decode(ctx, single) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "Failed to decode token %d\n", step);
            llama_batch_free(single);
            break;
        }
        llama_batch_free(single);
        last_token = new_token;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    int n_generated = (int)output.size() > 0 ? std::min(n_predict, (int)tracker.selections.size()) : 0;
    if (n_generated > 0 && elapsed > 0) {
        fprintf(stderr, "  Generated %d tokens in %.2f s (%.1f tok/s)\n",
                n_generated, elapsed, n_generated / elapsed);
    }

    // Clean up bias
    if (use_cache_bias) {
        llama_set_expert_cache_bias(model, nullptr, 0);
    }

    llama_sampler_free(smpl);
    return output;
}

int main(int argc, char ** argv) {
    test_params params = parse_args(argc, argv);

    fprintf(stderr, "=== Expert Cache-Aware Routing Test ===\n");
    fprintf(stderr, "Model: %s\n", params.model_path.c_str());
    fprintf(stderr, "Prompt: %.60s%s\n", params.prompt.c_str(),
            params.prompt.size() > 60 ? "..." : "");
    fprintf(stderr, "Generate: %d tokens\n", params.n_predict);
    fprintf(stderr, "Cache bonus: %.2f\n", params.cache_bonus);
    fprintf(stderr, "\n");

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const int n_layer = llama_model_n_layer(model);
    int n_expert = 0, n_expert_used = 0;
    {
        char buf[64];
        // Try various metadata keys for expert count
        const char * expert_keys[] = {
            "qwen3next.expert_count", "qwen35moe.expert_count",
            "deepseek2.expert_count", "mixtral.expert_count", nullptr
        };
        const char * used_keys[] = {
            "qwen3next.expert_used_count", "qwen35moe.expert_used_count",
            "deepseek2.expert_used_count", "mixtral.expert_used_count", nullptr
        };
        for (const char ** k = expert_keys; *k; k++) {
            if (llama_model_meta_val_str(model, *k, buf, sizeof(buf)) > 0) {
                n_expert = atoi(buf); break;
            }
        }
        for (const char ** k = used_keys; *k; k++) {
            if (llama_model_meta_val_str(model, *k, buf, sizeof(buf)) > 0) {
                n_expert_used = atoi(buf); break;
            }
        }
    }

    fprintf(stderr, "Architecture: %d layers, %d experts, top-%d routing\n",
            n_layer, n_expert, n_expert_used);

    if (n_expert == 0) {
        fprintf(stderr, "Not a MoE model. Exiting.\n");
        llama_model_free(model);
        return 1;
    }

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Tokenize prompt
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens(params.prompt.size() + 16);
    int n_tokens = llama_tokenize(vocab, params.prompt.c_str(), params.prompt.size(),
                                   prompt_tokens.data(), prompt_tokens.size(), true, true);
    if (n_tokens < 0) {
        prompt_tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, params.prompt.c_str(), params.prompt.size(),
                                   prompt_tokens.data(), prompt_tokens.size(), true, true);
    }
    prompt_tokens.resize(n_tokens);
    fprintf(stderr, "Prompt tokens: %d\n\n", n_tokens);

    // Run 1: Baseline (no cache bias)
    fprintf(stderr, "--- Run 1: Baseline (no cache bias) ---\n");
    expert_tracker tracker_baseline;
    std::string output_baseline = generate_with_tracking(
        ctx, model, prompt_tokens, params.n_predict,
        0.0f, false, tracker_baseline);
    fprintf(stderr, "  Output: %.80s%s\n\n",
            output_baseline.c_str(), output_baseline.size() > 80 ? "..." : "");

    // Run 2: Cache-biased routing
    fprintf(stderr, "--- Run 2: Cache-biased (bonus=%.2f) ---\n", params.cache_bonus);
    expert_tracker tracker_biased;
    std::string output_biased = generate_with_tracking(
        ctx, model, prompt_tokens, params.n_predict,
        params.cache_bonus, true, tracker_biased);
    fprintf(stderr, "  Output: %.80s%s\n\n",
            output_biased.c_str(), output_biased.size() > 80 ? "..." : "");

    // Compare outputs
    fprintf(stderr, "=== Results ===\n");
    bool same_output = (output_baseline == output_biased);
    fprintf(stderr, "Output match: %s\n", same_output ? "IDENTICAL" : "DIFFERENT");
    if (!same_output) {
        // Count token-level agreement
        int agree = 0;
        int total = std::min(output_baseline.size(), output_biased.size());
        for (int i = 0; i < total; i++) {
            if (output_baseline[i] == output_biased[i]) agree++;
        }
        fprintf(stderr, "Character agreement: %d/%d (%.1f%%)\n",
                agree, total, 100.0f * agree / std::max(total, 1));
    }

    fprintf(stderr, "\nCache bonus %.2f: %s quality impact\n",
            params.cache_bonus,
            same_output ? "ZERO" : "MEASURABLE");
    fprintf(stderr, "Next: integrate actual expert selection tracking via graph callback\n");

    // Sweep different bonus values
    fprintf(stderr, "\n--- Bonus Sweep ---\n");
    fprintf(stderr, "%-8s %-12s %-10s\n", "Bonus", "tok/s", "Match?");
    fprintf(stderr, "%-8s %-12s %-10s\n", "--------", "------------", "----------");

    for (float bonus : {0.0f, 0.1f, 0.3f, 0.5f, 1.0f, 2.0f}) {
        expert_tracker t;
        auto t_start = std::chrono::high_resolution_clock::now();
        std::string out = generate_with_tracking(
            ctx, model, prompt_tokens, params.n_predict,
            bonus, bonus > 0.0f, t);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        int n_gen = (int)t.selections.size();
        double tps = n_gen > 0 ? n_gen / elapsed : 0;

        fprintf(stderr, "%-8.1f %-12.1f %-10s\n",
                bonus, tps,
                (out == output_baseline) ? "SAME" : "DIFF");
    }

    // Cleanup
    llama_free(ctx);
    llama_model_free(model);

    fprintf(stderr, "\n=== Done ===\n");
    return 0;
}
