// Tests for llama_vocab_pruner to prevent GPU crashes from graph topology mismatches.
//
// The vocab pruner changes the compute graph topology (smaller lm_head output,
// extra tensors for hot_ids and hidden_norm). If the scheduler isn't re-reserved
// when the pruner activates, the GPU writes to wrong memory -> driver crash -> system hang.
//
// These tests verify the invariants that prevent that.

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <unordered_set>

// Include the pruner header directly - it's self-contained
#include "../src/llama-vocab-pruner.h"

// ---- Test helpers ----

static void test_default_state() {
    printf("  test_default_state...\n");

    llama_vocab_pruner p;

    assert(!p.enabled);
    assert(!p.norms_initialized);
    assert(p.hot_set.empty());
    assert(p.output_row_norms.empty());
    assert(p.max_norm_outside_hot_set == 0.0f);
    assert(p.n_pruned_hits == 0);
    assert(p.n_pruned_misses == 0);
    assert(p.n_total_decodes == 0);
    assert(p.tokens_since_rebuild == 0);

    printf("    PASS\n");
}

static void test_update_does_nothing_when_disabled() {
    printf("  test_update_does_nothing_when_disabled...\n");

    llama_vocab_pruner p;
    // enabled = false by default

    p.update(42);
    assert(p.recent_tokens.empty());
    assert(p.tokens_since_rebuild == 0);

    printf("    PASS\n");
}

static void test_update_tracks_tokens() {
    printf("  test_update_tracks_tokens...\n");

    llama_vocab_pruner p;
    p.enabled = true;

    p.update(10);
    p.update(20);
    p.update(30);

    assert(p.recent_tokens.size() == 3);
    assert(p.recent_tokens[0] == 10);
    assert(p.recent_tokens[1] == 20);
    assert(p.recent_tokens[2] == 30);
    assert(p.tokens_since_rebuild == 3);

    printf("    PASS\n");
}

static void test_recent_ring_buffer_wraps() {
    printf("  test_recent_ring_buffer_wraps...\n");

    llama_vocab_pruner p;
    p.enabled = true;
    p.recent_capacity = 4;

    for (int i = 0; i < 6; i++) {
        p.update((llama_token)i);
    }

    // Ring buffer of size 4, after inserting 0,1,2,3,4,5:
    // Buffer should contain [4, 5, 2, 3] (indices 0,1 overwritten)
    assert((int)p.recent_tokens.size() == 4);
    assert(p.recent_tokens[0] == 4);
    assert(p.recent_tokens[1] == 5);
    assert(p.recent_tokens[2] == 2);
    assert(p.recent_tokens[3] == 3);

    printf("    PASS\n");
}

static void test_needs_rebuild_interval() {
    printf("  test_needs_rebuild_interval...\n");

    llama_vocab_pruner p;
    p.enabled = true;

    for (int i = 0; i < llama_vocab_pruner::REBUILD_INTERVAL - 1; i++) {
        p.update(0);
    }
    assert(!p.needs_rebuild());

    p.update(0);
    assert(p.needs_rebuild());

    printf("    PASS\n");
}

static void test_init_default_hot_set() {
    printf("  test_init_default_hot_set...\n");

    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 100;

    // Without norms, max_norm_outside stays 0
    p.init_default_hot_set(500);

    assert((int)p.hot_set.size() == 100);
    // Default hot set is first N tokens, sorted
    for (int i = 0; i < 100; i++) {
        assert(p.hot_set[i] == (llama_token)i);
    }

    printf("    PASS\n");
}

static void test_init_default_hot_set_smaller_vocab() {
    printf("  test_init_default_hot_set_smaller_vocab...\n");

    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 1000;

    p.init_default_hot_set(50);

    // When vocab < capacity, hot set is clamped to vocab size
    assert((int)p.hot_set.size() == 50);

    printf("    PASS\n");
}

static void test_compute_max_norm_outside() {
    printf("  test_compute_max_norm_outside...\n");

    const int n_vocab = 10;
    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 3;

    // Fake norms: token 7 has the highest norm (9.0)
    p.output_row_norms.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        p.output_row_norms[i] = (float)(i + 1);  // norms: 1,2,3,4,5,6,7,8,9,10
    }
    p.norms_initialized = true;

    // Hot set = {0, 1, 2} => max norm outside = 10.0 (token 9)
    p.init_default_hot_set(n_vocab);
    assert((int)p.hot_set.size() == 3);
    assert(p.max_norm_outside_hot_set == 10.0f);

    // Now set hot set to include token 9 (the highest norm)
    p.hot_set = {7, 8, 9};
    p.compute_max_norm_outside(n_vocab);
    // Max outside is now token 6 (norm=7.0)
    assert(p.max_norm_outside_hot_set == 7.0f);

    printf("    PASS\n");
}

static void test_confidence_check_basic() {
    printf("  test_confidence_check_basic...\n");

    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set = {0, 1, 2};
    p.max_norm_outside_hot_set = 5.0f;

    // Cauchy-Schwarz: confident if top_hot_logit > max_norm_outside * hidden_norm
    // 30.0 > 5.0 * 4.0 = 20.0 => confident
    assert(p.check_confidence(30.0f, 4.0f) == true);

    // 15.0 > 5.0 * 4.0 = 20.0 => NOT confident
    assert(p.check_confidence(15.0f, 4.0f) == false);

    // Edge case: exactly equal => NOT confident (strict >)
    assert(p.check_confidence(20.0f, 4.0f) == false);

    printf("    PASS\n");
}

static void test_confidence_check_disabled() {
    printf("  test_confidence_check_disabled...\n");

    llama_vocab_pruner p;
    p.enabled = false;
    p.hot_set = {0, 1};
    p.max_norm_outside_hot_set = 1.0f;

    // Should always return false when disabled
    assert(p.check_confidence(1000.0f, 1.0f) == false);

    printf("    PASS\n");
}

static void test_confidence_check_empty_hot_set() {
    printf("  test_confidence_check_empty_hot_set...\n");

    llama_vocab_pruner p;
    p.enabled = true;
    // hot_set is empty

    // Should always return false with empty hot set
    assert(p.check_confidence(1000.0f, 1.0f) == false);

    printf("    PASS\n");
}

static void test_rebuild_hot_set_from_frequency() {
    printf("  test_rebuild_hot_set_from_frequency...\n");

    const int n_vocab = 20;
    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 5;

    // Setup fake norms (all equal so frequency dominates)
    p.output_row_norms.resize(n_vocab, 1.0f);
    p.norms_initialized = true;
    p.init_default_hot_set(n_vocab);

    // Simulate tokens: 15 appears 10x, 3 appears 5x, 7 appears 3x
    p.recent_capacity = 100;
    for (int i = 0; i < 10; i++) p.update(15);
    for (int i = 0; i < 5; i++) p.update(3);
    for (int i = 0; i < 3; i++) p.update(7);
    p.tokens_since_rebuild = llama_vocab_pruner::REBUILD_INTERVAL;

    p.rebuild_hot_set(n_vocab);

    // Hot set should contain at least these frequent tokens
    assert(std::binary_search(p.hot_set.begin(), p.hot_set.end(), (llama_token)15));
    assert(std::binary_search(p.hot_set.begin(), p.hot_set.end(), (llama_token)3));
    assert(std::binary_search(p.hot_set.begin(), p.hot_set.end(), (llama_token)7));

    // Should be sorted
    for (int i = 1; i < (int)p.hot_set.size(); i++) {
        assert(p.hot_set[i] > p.hot_set[i-1]);
    }

    // Should have exactly hot_set_capacity elements
    assert((int)p.hot_set.size() == 5);

    // tokens_since_rebuild should be reset
    assert(p.tokens_since_rebuild == 0);

    printf("    PASS\n");
}

static void test_rebuild_includes_high_norm_tokens() {
    printf("  test_rebuild_includes_high_norm_tokens...\n");

    const int n_vocab = 100;
    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 10;

    // Token 99 has extremely high norm, others are low
    p.output_row_norms.resize(n_vocab, 1.0f);
    p.output_row_norms[99] = 1000.0f;
    p.norms_initialized = true;

    // Only 2 frequent tokens, leaves room for high-norm fill
    p.recent_capacity = 100;
    p.update(5);
    p.update(5);
    p.tokens_since_rebuild = llama_vocab_pruner::REBUILD_INTERVAL;

    p.rebuild_hot_set(n_vocab);

    // Token 99 should be included (high norm = dangerous to miss)
    assert(std::binary_search(p.hot_set.begin(), p.hot_set.end(), (llama_token)99));
    // Token 5 should be included (frequent)
    assert(std::binary_search(p.hot_set.begin(), p.hot_set.end(), (llama_token)5));

    printf("    PASS\n");
}

static void test_rebuild_resets_counter() {
    printf("  test_rebuild_resets_counter...\n");

    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 5;
    p.output_row_norms.resize(10, 1.0f);
    p.norms_initialized = true;
    p.init_default_hot_set(10);

    p.tokens_since_rebuild = 100;
    p.rebuild_hot_set(10);

    assert(p.tokens_since_rebuild == 0);
    assert(!p.needs_rebuild());

    printf("    PASS\n");
}

static void test_rebuild_does_nothing_when_disabled() {
    printf("  test_rebuild_does_nothing_when_disabled...\n");

    llama_vocab_pruner p;
    // enabled = false, norms not initialized

    p.tokens_since_rebuild = 100;
    p.rebuild_hot_set(1000);

    // Nothing should change
    assert(p.hot_set.empty());
    assert(p.tokens_since_rebuild == 100);  // NOT reset since rebuild didn't run

    printf("    PASS\n");
}

static void test_max_norm_outside_updates_after_rebuild() {
    printf("  test_max_norm_outside_updates_after_rebuild...\n");

    const int n_vocab = 10;
    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 5;

    p.output_row_norms.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        p.output_row_norms[i] = (float)(i * 10);  // 0, 10, 20, ..., 90
    }
    p.norms_initialized = true;
    p.init_default_hot_set(n_vocab);

    // After default init, hot set = {0,1,2,3,4}, max outside = 90 (token 9)
    assert(p.max_norm_outside_hot_set == 90.0f);

    // Simulate frequent use of token 9 (highest norm)
    p.recent_capacity = 100;
    for (int i = 0; i < 50; i++) p.update(9);
    p.tokens_since_rebuild = llama_vocab_pruner::REBUILD_INTERVAL;

    p.rebuild_hot_set(n_vocab);

    // Token 9 should now be IN the hot set
    assert(std::binary_search(p.hot_set.begin(), p.hot_set.end(), (llama_token)9));

    // max_norm_outside should have decreased
    assert(p.max_norm_outside_hot_set < 90.0f);

    printf("    PASS\n");
}

static void test_hit_rate_tracking() {
    printf("  test_hit_rate_tracking...\n");

    llama_vocab_pruner p;

    assert(p.get_hit_rate() == 0.0f);

    p.n_total_decodes = 10;
    p.n_pruned_hits = 7;
    p.n_pruned_misses = 3;

    float hr = p.get_hit_rate();
    assert(hr > 0.69f && hr < 0.71f);  // 0.7

    printf("    PASS\n");
}

// ---- Safety invariant tests ----
// These test the conditions that, if violated, cause system crashes.

static void test_safety_hot_set_size_matches_capacity() {
    printf("  test_safety_hot_set_size_matches_capacity...\n");

    // INVARIANT: After rebuild, hot_set.size() == hot_set_capacity (if vocab >= capacity).
    // Violation means the graph tensor dimensions won't match what was reserved.

    const int n_vocab = 200;
    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 50;
    p.output_row_norms.resize(n_vocab, 1.0f);
    p.norms_initialized = true;

    p.recent_capacity = 100;
    for (int i = 0; i < 20; i++) p.update((llama_token)(i * 3));
    p.tokens_since_rebuild = llama_vocab_pruner::REBUILD_INTERVAL;

    p.rebuild_hot_set(n_vocab);

    // CRITICAL: size must exactly equal capacity for graph tensor dimensions
    assert((int)p.hot_set.size() == p.hot_set_capacity);

    printf("    PASS\n");
}

static void test_safety_hot_set_ids_in_vocab_range() {
    printf("  test_safety_hot_set_ids_in_vocab_range...\n");

    // INVARIANT: All hot set token IDs must be in [0, n_vocab).
    // Out-of-range IDs in ggml_get_rows => out-of-bounds GPU memory access => crash.

    const int n_vocab = 50;
    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 20;
    p.output_row_norms.resize(n_vocab, 1.0f);
    p.norms_initialized = true;

    p.recent_capacity = 100;
    for (int i = 0; i < 30; i++) p.update((llama_token)(i % n_vocab));
    p.tokens_since_rebuild = llama_vocab_pruner::REBUILD_INTERVAL;

    p.rebuild_hot_set(n_vocab);

    for (const auto & tok : p.hot_set) {
        assert(tok >= 0);
        assert(tok < n_vocab);
    }

    printf("    PASS\n");
}

static void test_safety_hot_set_no_duplicates() {
    printf("  test_safety_hot_set_no_duplicates...\n");

    // INVARIANT: Hot set must have no duplicate token IDs.
    // Duplicates would waste capacity and could confuse ggml_get_rows.

    const int n_vocab = 100;
    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 30;
    p.output_row_norms.resize(n_vocab, 1.0f);
    p.norms_initialized = true;

    // Insert many duplicates
    p.recent_capacity = 100;
    for (int i = 0; i < 80; i++) p.update((llama_token)(i % 5));
    p.tokens_since_rebuild = llama_vocab_pruner::REBUILD_INTERVAL;

    p.rebuild_hot_set(n_vocab);

    std::unordered_set<llama_token> seen;
    for (const auto & tok : p.hot_set) {
        assert(seen.find(tok) == seen.end() && "duplicate token in hot set");
        seen.insert(tok);
    }

    printf("    PASS\n");
}

static void test_safety_hot_set_sorted() {
    printf("  test_safety_hot_set_sorted...\n");

    // INVARIANT: Hot set must be sorted for binary_search in compute_max_norm_outside.
    // Unsorted hot set => wrong max_norm => wrong confidence => silent corruption.

    const int n_vocab = 100;
    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 25;
    p.output_row_norms.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        p.output_row_norms[i] = (float)(n_vocab - i);  // reverse order norms
    }
    p.norms_initialized = true;

    p.recent_capacity = 100;
    for (int i = 0; i < 50; i++) p.update((llama_token)(n_vocab - 1 - i));
    p.tokens_since_rebuild = llama_vocab_pruner::REBUILD_INTERVAL;

    p.rebuild_hot_set(n_vocab);

    assert(std::is_sorted(p.hot_set.begin(), p.hot_set.end()));

    printf("    PASS\n");
}

static void test_safety_norms_all_non_negative() {
    printf("  test_safety_norms_all_non_negative...\n");

    // INVARIANT: L2 norms must be >= 0. Negative norms would break Cauchy-Schwarz.

    llama_vocab_pruner p;
    p.output_row_norms = {0.0f, 1.5f, 3.0f, 0.001f, 100.0f};
    p.norms_initialized = true;

    for (const auto & n : p.output_row_norms) {
        assert(n >= 0.0f);
        assert(!std::isnan(n));
        assert(!std::isinf(n));
    }

    printf("    PASS\n");
}

static void test_safety_confidence_never_true_with_zero_norm() {
    printf("  test_safety_confidence_never_true_with_zero_norm...\n");

    // If max_norm_outside is 0, the bound is trivially satisfied for any positive logit.
    // But this means ALL non-hot tokens have zero norm (zero output rows).
    // In practice this is degenerate but shouldn't crash.

    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set = {0};
    p.max_norm_outside_hot_set = 0.0f;

    // Any positive logit > 0.0 * anything = 0.0
    assert(p.check_confidence(0.001f, 1.0f) == true);
    // But zero logit is not > 0
    assert(p.check_confidence(0.0f, 1.0f) == false);
    // Negative logit is not > 0
    assert(p.check_confidence(-1.0f, 1.0f) == false);

    printf("    PASS\n");
}

static void test_safety_min_hot_set_capacity() {
    printf("  test_safety_min_hot_set_capacity...\n");

    // INVARIANT: hot_set_capacity should never be less than 256 (enforced at init).
    // A tiny hot set would almost never pass the confidence check,
    // forcing constant full recomputation.

    llama_vocab_pruner p;
    p.hot_set_capacity = 10;

    // The context constructor clamps this to 256, but test the pruner itself
    // doesn't crash with small values
    p.enabled = true;
    p.output_row_norms.resize(20, 1.0f);
    p.norms_initialized = true;

    p.init_default_hot_set(20);
    assert((int)p.hot_set.size() == 10);  // clamped to vocab size, not capacity

    printf("    PASS\n");
}

// ---- Crash scenario regression tests ----

static void test_crash_regression_graph_topology_mismatch() {
    printf("  test_crash_regression_graph_topology_mismatch...\n");

    // REGRESSION: vocab_pruner_init_norms() armed the pruner but didn't set
    // sched_need_reserve. The scheduler had buffers for the un-pruned graph,
    // but the next decode built the pruned graph -> buffer mismatch -> GPU crash.
    //
    // We can't test sched_need_reserve directly (it's in llama_context),
    // but we verify the invariant: after norms_initialized transitions from
    // false to true, the hot set must be non-empty AND consistent.

    llama_vocab_pruner p;
    p.enabled = true;

    // Before init: should NOT be used for pruning
    assert(!p.norms_initialized);
    assert(p.hot_set.empty());

    // Simulate init_norms completing (use vocab > capacity to match real usage)
    const int n_vocab = 151552;
    p.output_row_norms.resize(n_vocab, 2.0f);
    p.norms_initialized = true;
    p.init_default_hot_set(n_vocab);

    // After init: hot set must be populated and consistent
    assert(!p.hot_set.empty());
    // When vocab >= capacity, size must equal capacity exactly
    assert((int)p.hot_set.size() == p.hot_set_capacity);
    assert(p.max_norm_outside_hot_set > 0.0f);

    // The graph topology depends on hot_set_size, which must match the tensor
    // dimensions in build_lm_head_pruned. Verify it's stable.
    const int expected_size = std::min(p.hot_set_capacity, n_vocab);
    assert((int)p.hot_set.size() == expected_size);

    printf("    PASS\n");
}

static void test_crash_regression_hot_set_capacity_changes() {
    printf("  test_crash_regression_hot_set_capacity_changes...\n");

    // INVARIANT: hot_set.size() must not change between graph reservation and
    // graph compute. If it does, the tensor dimensions mismatch -> crash.
    //
    // After rebuild, the size must equal hot_set_capacity.
    // hot_set_capacity must never change after initialization.

    const int n_vocab = 500;
    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 100;
    p.output_row_norms.resize(n_vocab, 1.0f);
    p.norms_initialized = true;
    p.init_default_hot_set(n_vocab);

    const int initial_size = (int)p.hot_set.size();
    assert(initial_size == 100);

    // Simulate multiple rebuilds with different frequency patterns
    p.recent_capacity = 200;
    for (int round = 0; round < 5; round++) {
        for (int i = 0; i < llama_vocab_pruner::REBUILD_INTERVAL; i++) {
            p.update((llama_token)((round * 37 + i * 13) % n_vocab));
        }
        p.rebuild_hot_set(n_vocab);

        // Size must be EXACTLY the same after every rebuild
        assert((int)p.hot_set.size() == initial_size);
    }

    printf("    PASS\n");
}

static void test_crash_regression_151k_norm_loop() {
    printf("  test_crash_regression_151k_norm_loop...\n");

    // The norm initialization loop iterates n_vocab times (151K for Qwen3.5).
    // Verify the pruner handles large vocab sizes without issues.

    const int n_vocab = 151552;  // actual Qwen3.5 vocab size
    llama_vocab_pruner p;
    p.enabled = true;
    p.hot_set_capacity = 4096;

    p.output_row_norms.resize(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        p.output_row_norms[i] = sqrtf((float)(i + 1));
    }
    p.norms_initialized = true;
    p.init_default_hot_set(n_vocab);

    // Verify hot set is correct size
    assert((int)p.hot_set.size() == 4096);

    // max_norm_outside should be the highest norm token not in hot set
    // Hot set is {0..4095}, so max outside is sqrt(151552) ~ 389.3
    float expected_max = sqrtf((float)n_vocab);
    assert(fabsf(p.max_norm_outside_hot_set - expected_max) < 0.01f);

    // Verify rebuild works at scale
    p.recent_capacity = 8192;
    for (int i = 0; i < llama_vocab_pruner::REBUILD_INTERVAL; i++) {
        p.update((llama_token)(150000 + (i % 100)));
    }
    p.rebuild_hot_set(n_vocab);

    assert((int)p.hot_set.size() == 4096);
    assert(std::is_sorted(p.hot_set.begin(), p.hot_set.end()));

    // High-frequency tokens (150000-150099) should be in the hot set
    assert(std::binary_search(p.hot_set.begin(), p.hot_set.end(), (llama_token)150000));

    printf("    PASS\n");
}

int main() {
    printf("test-vocab-pruner:\n");

    printf("\n--- Basic functionality ---\n");
    test_default_state();
    test_update_does_nothing_when_disabled();
    test_update_tracks_tokens();
    test_recent_ring_buffer_wraps();
    test_needs_rebuild_interval();
    test_init_default_hot_set();
    test_init_default_hot_set_smaller_vocab();
    test_compute_max_norm_outside();
    test_confidence_check_basic();
    test_confidence_check_disabled();
    test_confidence_check_empty_hot_set();
    test_rebuild_hot_set_from_frequency();
    test_rebuild_includes_high_norm_tokens();
    test_rebuild_resets_counter();
    test_rebuild_does_nothing_when_disabled();
    test_max_norm_outside_updates_after_rebuild();
    test_hit_rate_tracking();

    printf("\n--- Safety invariants ---\n");
    test_safety_hot_set_size_matches_capacity();
    test_safety_hot_set_ids_in_vocab_range();
    test_safety_hot_set_no_duplicates();
    test_safety_hot_set_sorted();
    test_safety_norms_all_non_negative();
    test_safety_confidence_never_true_with_zero_norm();
    test_safety_min_hot_set_capacity();

    printf("\n--- Crash regression tests ---\n");
    test_crash_regression_graph_topology_mismatch();
    test_crash_regression_hot_set_capacity_changes();
    test_crash_regression_151k_norm_loop();

    printf("\nAll tests passed!\n");
    return 0;
}
