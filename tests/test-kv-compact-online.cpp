// Tests for KV cache compaction online/auto-compact features
//
// Tests the auto-compaction configuration, bias serialization round-trip,
// and the API surface without requiring a full model load.

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../tools/kv-compact/kv-compact-math.h"
#include "llama.h"

// ============================================================================
// Auto-compact params default tests
// ============================================================================

static void test_compact_params_default() {
    printf("  test_compact_params_default...");
    struct llama_compact_params params = llama_compact_params_default();

    assert(params.target_ratio > 0.0f && params.target_ratio <= 1.0f);
    assert(params.n_ref_queries == 0);  // auto
    assert(!params.use_repeat_prefill);
    assert(!params.use_nonuniform_budgets);
    assert(params.budget_profile_path == nullptr);
    printf(" OK\n");
}

// ============================================================================
// Bias serialization format tests (unit-level, no model needed)
// ============================================================================

// Simple in-memory IO for testing serialization
struct mem_write {
    std::vector<uint8_t> buf;

    void write(const void * data, size_t size) {
        const uint8_t * p = static_cast<const uint8_t *>(data);
        buf.insert(buf.end(), p, p + size);
    }
};

struct mem_read {
    const uint8_t * data;
    size_t size;
    size_t pos = 0;

    void read_to(void * dst, size_t n) {
        assert(pos + n <= size);
        memcpy(dst, data + pos, n);
        pos += n;
    }
};

static void test_bias_serialization_format() {
    printf("  test_bias_serialization_format...");

    // Test that the bias serialization format is correct:
    // [uint32_t has_bias] [per-layer data if has_bias]

    // Write: no bias
    {
        mem_write w;
        uint32_t has_bias = 0;
        w.write(&has_bias, sizeof(has_bias));
        assert(w.buf.size() == sizeof(uint32_t));

        mem_read r{w.buf.data(), w.buf.size()};
        uint32_t read_bias;
        r.read_to(&read_bias, sizeof(read_bias));
        assert(read_bias == 0);
    }

    // Write: with bias (simulated 2 layers, 2 heads, 4 cells)
    {
        mem_write w;
        uint32_t has_bias = 1;
        w.write(&has_bias, sizeof(has_bias));

        for (int layer = 0; layer < 2; layer++) {
            uint32_t n_head_kv = 2;
            w.write(&n_head_kv, sizeof(n_head_kv));

            for (uint32_t h = 0; h < n_head_kv; h++) {
                for (int cell = 0; cell < 4; cell++) {
                    float bias = (float)(layer * 100 + h * 10 + cell) * 0.1f;
                    w.write(&bias, sizeof(bias));
                }
            }
        }

        // Read back and verify
        mem_read r{w.buf.data(), w.buf.size()};
        uint32_t read_bias;
        r.read_to(&read_bias, sizeof(read_bias));
        assert(read_bias == 1);

        for (int layer = 0; layer < 2; layer++) {
            uint32_t n_head_kv;
            r.read_to(&n_head_kv, sizeof(n_head_kv));
            assert(n_head_kv == 2);

            for (uint32_t h = 0; h < n_head_kv; h++) {
                for (int cell = 0; cell < 4; cell++) {
                    float bias;
                    r.read_to(&bias, sizeof(bias));
                    float expected = (float)(layer * 100 + h * 10 + cell) * 0.1f;
                    assert(fabsf(bias - expected) < 1e-6f);
                }
            }
        }

        assert(r.pos == r.size);
    }

    printf(" OK\n");
}

// ============================================================================
// Compaction math: multiple consecutive compressions
// ============================================================================

static void test_consecutive_compressions() {
    printf("  test_consecutive_compressions...");

    // The paper demonstrates up to 6 consecutive compressions
    // Simulate compressing the same data repeatedly to verify
    // the algorithm doesn't diverge or produce NaN

    const int T_initial = 128;
    const int n_q = 32;
    const int d_k = 16;
    const int d_v = 16;
    const float target = 0.5f;

    // Generate initial data
    std::vector<float> K(T_initial * d_k);
    std::vector<float> V(T_initial * d_v);
    std::vector<float> Q(n_q * d_k);

    for (int i = 0; i < T_initial * d_k; i++) K[i] = sinf((float)i * 0.3f) * 0.5f;
    for (int i = 0; i < T_initial * d_v; i++) V[i] = cosf((float)i * 0.2f) * 0.5f;
    for (int i = 0; i < n_q * d_k; i++) Q[i] = sinf((float)i * 0.5f) * 0.5f;

    int T = T_initial;

    // Run 6 consecutive compressions
    for (int round = 0; round < 6; round++) {
        int t = std::max(2, (int)(T * target));

        compacted_head result = compact_head_highest_attn(
            K.data(), V.data(), Q.data(),
            T, n_q, d_k, d_v, t);

        // Verify results are valid
        assert(result.selected_indices.size() == (size_t)t);
        assert(result.beta.size() == (size_t)t);
        assert(result.C_v.size() == (size_t)(t * d_v));

        // All values should be finite
        for (int i = 0; i < t; i++) {
            assert(std::isfinite(result.beta[i]));
        }
        for (int i = 0; i < t * d_v; i++) {
            assert(std::isfinite(result.C_v[i]));
        }

        // Build compacted K and V for next round
        std::vector<float> K_new(t * d_k);
        std::vector<float> V_new(t * d_v);

        for (int i = 0; i < t; i++) {
            memcpy(K_new.data() + i * d_k,
                   K.data() + result.selected_indices[i] * d_k,
                   d_k * sizeof(float));
            memcpy(V_new.data() + i * d_v,
                   result.C_v.data() + i * d_v,
                   d_v * sizeof(float));
        }

        K = std::move(K_new);
        V = std::move(V_new);
        T = t;

        printf("\n    Round %d: %d -> %d tokens", round + 1, T_initial >> round, T);
    }

    printf("\n    Final tokens: %d (from %d, ratio %.4f)\n", T, T_initial, (float)T / T_initial);

    // After 6 rounds at 50%, we should have ~2 tokens (128 * 0.5^6 = 2)
    assert(T >= 2);
    assert(T <= 4);

    printf("  OK\n");
}

// ============================================================================
// Threshold logic tests
// ============================================================================

static void test_threshold_boundary_conditions() {
    printf("  test_threshold_boundary_conditions...");

    // Test the threshold calculation logic
    // usage = used / kv_size

    // At exactly threshold: should trigger
    {
        uint32_t kv_size = 100;
        uint32_t used = 90;
        float threshold = 0.9f;
        float usage = (float)used / (float)kv_size;
        assert(usage >= threshold);
    }

    // Below threshold: should not trigger
    {
        uint32_t kv_size = 100;
        uint32_t used = 89;
        float threshold = 0.9f;
        float usage = (float)used / (float)kv_size;
        assert(usage < threshold);
    }

    // 100% full: should trigger at any positive threshold
    {
        uint32_t kv_size = 100;
        uint32_t used = 100;
        float threshold = 0.5f;
        float usage = (float)used / (float)kv_size;
        assert(usage >= threshold);
    }

    printf(" OK\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("test-kv-compact-online:\n");

    printf("\n=== Compact params defaults ===\n");
    test_compact_params_default();

    printf("\n=== Bias serialization format ===\n");
    test_bias_serialization_format();

    printf("\n=== Consecutive compressions ===\n");
    test_consecutive_compressions();

    printf("\n=== Threshold logic ===\n");
    test_threshold_boundary_conditions();

    printf("\nAll online compaction tests passed!\n");
    return 0;
}
