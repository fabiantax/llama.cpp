// KV Cache Compaction — Per-Head Sensitivity Profiler
//
// Measures per-(layer, head) reconstruction error at multiple compression
// ratios.  Outputs a JSON sensitivity profile that can be loaded by
// llama_kv_cache_compact() for non-uniform budget allocation.
//
// Usage:
//   llama-kv-compact-profile -m model.gguf -p "calibration text"
//       --profile-output profile.json --ratios "0.1,0.2,0.5,0.8"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kv-compact-math.h"

// ============================================================================
// KV data helpers (same as kv-compact.cpp — shared via header in production)
// ============================================================================

static void read_k_head_profile(const ggml_tensor * k_tensor, int head_idx,
                                int n_embd_head_k, int n_embd_k_gqa,
                                int n_tokens, std::vector<float> & out) {
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
            memcpy(out.data() + i * n_embd_head_k, full_row.data() + head_offset,
                   n_embd_head_k * sizeof(float));
        }
    }
}

static void read_v_head_profile(const ggml_tensor * v_tensor, int head_idx,
                                int n_embd_head_v, int n_embd_v_gqa,
                                int n_tokens, int kv_size, bool v_trans,
                                std::vector<float> & out) {
    out.resize(n_tokens * n_embd_head_v);
    if (!v_trans) {
        read_k_head_profile(v_tensor, head_idx, n_embd_head_v, n_embd_v_gqa, n_tokens, out);
        return;
    }
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
        LOG_ERR("Unsupported V tensor type for transposed read: %s\n", ggml_type_name(v_tensor->type));
        std::fill(out.begin(), out.end(), 0.0f);
    }
}

// ============================================================================
// JSON output
// ============================================================================

static std::string escape_json_string(const std::string & s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            default:   out += c;
        }
    }
    return out;
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(int argc, char ** argv) {
    (void) argc;
    LOG("\nKV Cache Compaction — Per-Head Sensitivity Profiler\n\n");
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("  -m  MODEL             path to model file\n");
    LOG("  -p  PROMPT            calibration text\n");
    LOG("  -f  FILE              read calibration text from file\n");
    LOG("  -c  N                 context size (default: 2048)\n");
    LOG("  --profile-output FILE output JSON profile (default: stdout)\n");
    LOG("  --ratios CSV          compression ratios to test (default: 0.1,0.2,0.3,0.5,0.8)\n");
    LOG("  --n-ref-queries N     number of reference queries (default: 32)\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;
    std::string profile_output;
    std::string ratios_str = "0.1,0.2,0.3,0.5,0.8";
    int n_ref_queries = 32;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--profile-output") == 0 && i + 1 < argc) {
            profile_output = argv[++i];
        } else if (strcmp(argv[i], "--ratios") == 0 && i + 1 < argc) {
            ratios_str = argv[++i];
        } else if (strcmp(argv[i], "--n-ref-queries") == 0 && i + 1 < argc) {
            n_ref_queries = std::stoi(argv[++i]);
        }
    }

    // Parse ratios
    std::vector<float> ratios;
    {
        std::istringstream ss(ratios_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            float r = std::stof(tok);
            if (r > 0.0f && r < 1.0f) ratios.push_back(r);
        }
    }
    if (ratios.empty()) {
        LOG_ERR("No valid ratios provided\n");
        return 1;
    }

    common_init();

    LOG_INF("=== KV Cache Compaction — Sensitivity Profiler ===\n");
    LOG_INF("Ratios to test: ");
    for (size_t i = 0; i < ratios.size(); i++) {
        LOG_INF("%.2f%s", ratios[i], i + 1 < ratios.size() ? ", " : "\n");
    }

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
    const int n_layer   = llama_model_n_layer(model);
    const int n_head_kv = llama_model_n_head_kv(model);
    const int n_embd    = llama_model_n_embd(model);
    const int n_head    = llama_model_n_head(model);
    const int n_embd_head_k = n_embd / n_head;

    LOG_INF("Model: %d layers, %d KV heads, n_embd_head=%d\n", n_layer, n_head_kv, n_embd_head_k);

    // Tokenize and prefill
    std::string prompt = params.prompt;
    if (prompt.empty()) {
        LOG_ERR("No calibration text provided. Use -p or -f.\n");
        return 1;
    }

    std::vector<llama_token> tokens = common_tokenize(vocab, prompt, true, false);
    const int n_tokens = (int)tokens.size();

    if (n_tokens < 32) {
        LOG_ERR("Calibration text too short (%d tokens). Need at least 32.\n", n_tokens);
        return 1;
    }

    LOG_INF("Calibration: %d tokens\n", n_tokens);

    // Prefill
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, {0}, (i == n_tokens - 1));
    }

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("Failed to decode calibration batch\n");
        llama_batch_free(batch);
        return 1;
    }

    LOG_INF("Prefill complete.\n");

    // Access KV cache
    llama_memory_t mem = llama_get_memory(ctx);
    // Save state for raw tensor access
    const size_t state_size = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> state_buf(state_size);
    llama_state_seq_get_data(ctx, state_buf.data(), state_buf.size(), 0);

    // Parse state to get K/V tensors per layer
    // (Same parsing as kv-compact.cpp)
    const uint8_t * ptr = state_buf.data();
    const uint8_t * end_ptr = ptr + state_size;

    uint32_t n_stream_state;
    memcpy(&n_stream_state, ptr, sizeof(n_stream_state));
    ptr += sizeof(n_stream_state);

    // We only process stream 0
    uint32_t cell_count;
    memcpy(&cell_count, ptr, sizeof(cell_count));
    ptr += sizeof(cell_count);

    if (cell_count == 0) {
        LOG_ERR("No cells in KV cache\n");
        llama_batch_free(batch);
        return 1;
    }

    // Skip cell metadata
    for (uint32_t c = 0; c < cell_count && ptr < end_ptr; c++) {
        ptr += sizeof(llama_pos);
        uint32_t n_seq_id;
        memcpy(&n_seq_id, ptr, sizeof(n_seq_id));
        ptr += sizeof(n_seq_id);
        ptr += n_seq_id * sizeof(llama_seq_id);
    }

    uint32_t v_trans_state, n_layer_state;
    memcpy(&v_trans_state, ptr, sizeof(v_trans_state));
    ptr += sizeof(v_trans_state);
    memcpy(&n_layer_state, ptr, sizeof(n_layer_state));
    ptr += sizeof(n_layer_state);

    // Read K data per layer
    std::vector<std::vector<float>> all_K(n_layer_state);
    std::vector<int> k_gqa_sizes(n_layer_state);

    for (uint32_t l = 0; l < n_layer_state && ptr < end_ptr; l++) {
        int32_t k_type_i;
        uint64_t k_size_row;
        memcpy(&k_type_i, ptr, sizeof(k_type_i));
        ptr += sizeof(k_type_i);
        memcpy(&k_size_row, ptr, sizeof(k_size_row));
        ptr += sizeof(k_size_row);

        const size_t k_data_size = cell_count * k_size_row;

        if (k_type_i == GGML_TYPE_F32) {
            int n_floats = (int)(k_size_row / sizeof(float));
            k_gqa_sizes[l] = n_floats;
            all_K[l].resize(cell_count * n_floats);
            memcpy(all_K[l].data(), ptr, k_data_size);
        } else if (k_type_i == GGML_TYPE_F16) {
            int n_floats = (int)(k_size_row / sizeof(ggml_fp16_t));
            k_gqa_sizes[l] = n_floats;
            all_K[l].resize(cell_count * n_floats);
            const ggml_fp16_t * src = (const ggml_fp16_t *)ptr;
            for (size_t i = 0; i < (size_t)(cell_count * n_floats); i++) {
                all_K[l][i] = ggml_fp16_to_fp32(src[i]);
            }
        } else {
            k_gqa_sizes[l] = 0;
        }
        ptr += k_data_size;
    }

    // Read V data per layer
    std::vector<std::vector<float>> all_V(n_layer_state);
    std::vector<int> v_gqa_sizes(n_layer_state);

    if (!v_trans_state) {
        for (uint32_t l = 0; l < n_layer_state && ptr < end_ptr; l++) {
            int32_t v_type_i;
            uint64_t v_size_row;
            memcpy(&v_type_i, ptr, sizeof(v_type_i));
            ptr += sizeof(v_type_i);
            memcpy(&v_size_row, ptr, sizeof(v_size_row));
            ptr += sizeof(v_size_row);

            const size_t v_data_size = cell_count * v_size_row;

            if (v_type_i == GGML_TYPE_F32) {
                int n_floats = (int)(v_size_row / sizeof(float));
                v_gqa_sizes[l] = n_floats;
                all_V[l].resize(cell_count * n_floats);
                memcpy(all_V[l].data(), ptr, v_data_size);
            } else if (v_type_i == GGML_TYPE_F16) {
                int n_floats = (int)(v_size_row / sizeof(ggml_fp16_t));
                v_gqa_sizes[l] = n_floats;
                all_V[l].resize(cell_count * n_floats);
                const ggml_fp16_t * src = (const ggml_fp16_t *)ptr;
                for (size_t i = 0; i < (size_t)(cell_count * n_floats); i++) {
                    all_V[l][i] = ggml_fp16_to_fp32(src[i]);
                }
            } else {
                v_gqa_sizes[l] = 0;
            }
            ptr += v_data_size;
        }
    } else {
        for (uint32_t l = 0; l < n_layer_state && ptr < end_ptr; l++) {
            int32_t v_type_i;
            uint32_t v_size_el;
            uint32_t n_embd_v_gqa_l;
            memcpy(&v_type_i, ptr, sizeof(v_type_i));
            ptr += sizeof(v_type_i);
            memcpy(&v_size_el, ptr, sizeof(v_size_el));
            ptr += sizeof(v_size_el);
            memcpy(&n_embd_v_gqa_l, ptr, sizeof(n_embd_v_gqa_l));
            ptr += sizeof(n_embd_v_gqa_l);

            const size_t v_data_size = (size_t)n_embd_v_gqa_l * cell_count * v_size_el;
            v_gqa_sizes[l] = n_embd_v_gqa_l;

            if (v_type_i == GGML_TYPE_F32) {
                all_V[l].resize(cell_count * n_embd_v_gqa_l);
                const float * src = (const float *)ptr;
                for (uint32_t d = 0; d < n_embd_v_gqa_l; d++) {
                    for (uint32_t c = 0; c < cell_count; c++) {
                        all_V[l][c * n_embd_v_gqa_l + d] = src[d * cell_count + c];
                    }
                }
            } else if (v_type_i == GGML_TYPE_F16) {
                all_V[l].resize(cell_count * n_embd_v_gqa_l);
                const ggml_fp16_t * src = (const ggml_fp16_t *)ptr;
                for (uint32_t d = 0; d < n_embd_v_gqa_l; d++) {
                    for (uint32_t c = 0; c < cell_count; c++) {
                        all_V[l][c * n_embd_v_gqa_l + d] = ggml_fp16_to_fp32(src[d * cell_count + c]);
                    }
                }
            } else {
                v_gqa_sizes[l] = 0;
            }
            ptr += v_data_size;
        }
    }

    (void)mem;

    // Profile each (layer, head) at each ratio
    LOG_INF("\nProfiling %d layers x %d heads x %d ratios...\n",
            n_layer, n_head_kv, (int)ratios.size());

    struct profile_entry {
        int layer;
        int head;
        float sensitivity;
        std::vector<std::pair<float, float>> curve;
    };
    std::vector<profile_entry> profiles;

    for (int il = 0; il < (int)n_layer_state; il++) {
        if (all_K[il].empty() || all_V[il].empty()) continue;

        const int n_embd_k_gqa_l = k_gqa_sizes[il];
        const int n_embd_v_gqa_l = v_gqa_sizes[il];
        if (n_embd_k_gqa_l == 0 || n_embd_v_gqa_l == 0) continue;

        const int d_k = n_embd_k_gqa_l / n_head_kv;
        const int d_v = n_embd_v_gqa_l / n_head_kv;

        for (int h = 0; h < n_head_kv; h++) {
            // Extract per-head K and V
            std::vector<float> K_head(cell_count * d_k);
            std::vector<float> V_head(cell_count * d_v);

            for (uint32_t tok = 0; tok < cell_count; tok++) {
                memcpy(K_head.data() + tok * d_k,
                       all_K[il].data() + tok * n_embd_k_gqa_l + h * d_k,
                       d_k * sizeof(float));
                memcpy(V_head.data() + tok * d_v,
                       all_V[il].data() + tok * n_embd_v_gqa_l + h * d_v,
                       d_v * sizeof(float));
            }

            // Use K-proxy for reference queries
            int actual_n_ref = std::min(n_ref_queries, (int)cell_count);
            std::vector<float> Q_ref(actual_n_ref * d_k);
            for (int q = 0; q < actual_n_ref; q++) {
                int src_idx = (int)(q * (cell_count - 1) / (actual_n_ref - 1 + 1e-9f));
                memcpy(Q_ref.data() + q * d_k,
                       K_head.data() + src_idx * d_k,
                       d_k * sizeof(float));
            }

            // Compute sensitivity profile
            head_sensitivity_profile prof = compute_head_sensitivity(
                K_head.data(), V_head.data(), Q_ref.data(),
                (int)cell_count, actual_n_ref, d_k, d_v,
                il, h, ratios.data(), (int)ratios.size());

            profile_entry entry;
            entry.layer = il;
            entry.head = h;
            entry.sensitivity = prof.sensitivity;
            entry.curve = prof.curve;
            profiles.push_back(entry);

            LOG_INF("  layer %2d head %d: sensitivity=%.8f\n", il, h, prof.sensitivity);
        }
    }

    // Output JSON
    std::ostringstream json;
    json << "{\n";
    json << "  \"model\": \"" << escape_json_string(params.model.path) << "\",\n";
    json << "  \"n_tokens\": " << n_tokens << ",\n";
    json << "  \"n_layer\": " << n_layer << ",\n";
    json << "  \"n_head_kv\": " << n_head_kv << ",\n";
    json << "  \"ratios\": [";
    for (size_t i = 0; i < ratios.size(); i++) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.4f", ratios[i]);
        json << buf << (i + 1 < ratios.size() ? ", " : "");
    }
    json << "],\n";
    json << "  \"heads\": [\n";
    for (size_t i = 0; i < profiles.size(); i++) {
        const auto & p = profiles[i];
        json << "    {\"layer\": " << p.layer << ", \"head\": " << p.head;
        char buf[32];
        snprintf(buf, sizeof(buf), "%.8f", p.sensitivity);
        json << ", \"sensitivity\": " << buf << ", \"curve\": [";
        for (size_t j = 0; j < p.curve.size(); j++) {
            char rbuf[32], ebuf[32];
            snprintf(rbuf, sizeof(rbuf), "%.4f", p.curve[j].first);
            snprintf(ebuf, sizeof(ebuf), "%.8f", p.curve[j].second);
            json << "[" << rbuf << ", " << ebuf << "]";
            if (j + 1 < p.curve.size()) json << ", ";
        }
        json << "]}";
        if (i + 1 < profiles.size()) json << ",";
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";

    if (profile_output.empty()) {
        printf("%s", json.str().c_str());
    } else {
        std::ofstream f(profile_output);
        if (!f.is_open()) {
            LOG_ERR("Failed to open %s for writing\n", profile_output.c_str());
            llama_batch_free(batch);
            llama_backend_free();
            return 1;
        }
        f << json.str();
        f.close();
        LOG_INF("\nProfile written to %s\n", profile_output.c_str());
    }

    // Summary statistics
    if (!profiles.empty()) {
        float min_s = profiles[0].sensitivity, max_s = profiles[0].sensitivity;
        float sum_s = 0.0f;
        for (const auto & p : profiles) {
            if (p.sensitivity < min_s) min_s = p.sensitivity;
            if (p.sensitivity > max_s) max_s = p.sensitivity;
            sum_s += p.sensitivity;
        }
        float mean_s = sum_s / profiles.size();

        LOG_INF("\n=== Sensitivity Summary ===\n");
        LOG_INF("  Heads profiled: %zu\n", profiles.size());
        LOG_INF("  Sensitivity range: [%.8f, %.8f]\n", min_s, max_s);
        LOG_INF("  Sensitivity mean:  %.8f\n", mean_s);
        LOG_INF("  Ratio max/min:     %.2fx\n", max_s / (min_s + 1e-12f));
        LOG_INF("\n  Higher sensitivity = head needs more budget (more sensitive to compression)\n");
    }

    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}
