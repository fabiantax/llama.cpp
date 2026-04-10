# Strix Halo Optimization Report — llama.cpp Fork

**Date:** 2026-03-09
**Branch:** `backup/strix-halo-optimizations` (also `hip-optimization`, `rocm-optimization`)
**Hardware:** AMD Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S (gfx1151, RDNA 3.5, 40 CUs), 68 GB LPDDR5X UMA
**Target Model:** Qwen3.5-35B-A3B Q4_K_M (hybrid SSM+MoE, 128 experts, top-8, ~3B active/token)

---

## Performance Summary

| Backend | Baseline | Optimized | Gain | Metric |
|---------|----------|-----------|------|--------|
| Vulkan (tg128) | 58.07 tok/s | **67.09 tok/s** | +15.5% | Token generation |
| HIP (tg128) | 43.84 tok/s | **56.46 tok/s** | +29% | Token generation |
| HIP (pp512) | 240.84 tok/s | **869.82 tok/s** | +261% | Prompt processing |
| Bandwidth efficiency | — | 60% (127/212 GB/s) | — | Of theoretical 212 GB/s |

Theoretical ceiling: 78 tok/s (70% bandwidth utilization). Remaining headroom: ~11 tok/s.

---

## Architecture: Why This Model Is Special

Qwen3.5-35B-A3B is **not a pure transformer** — it's a hybrid SSM + MoE model:
- 24 Delta-Net SSM layers per token (constant state, no KV cache scaling)
- MoE routing: 128 experts, top-8 selected per token
- ~3B parameters active per token out of 30B total
- Flat tok/s from tg32 to tg512 (SSM state is constant-size)

This means:
1. **KV cache optimizations don't help** (model is SSM-dominated)
2. **Flash attention tuning has minimal effect** (few attention layers)
3. **Expert weight prefetch matters** (only 8/128 experts loaded per token)
4. **SSM kernel fusion is critical** (24 layers × 11 dispatches = 264 dispatches/token)

---

## Optimizations: What Worked

### 1. SSM Shared Memory Tiling (+14.8%, 58.44 → 67.09 tok/s)

**The #1 win.** DELTA_NET_RECURRENCE was the #1 non-matmul cost: 13,758 us/5-token-iteration (18.3% of total).

**Problem:** Each thread reads state row i (128 floats = 512 bytes), but threads are 512 bytes apart → non-coalesced access, wasting 31/32 cache line bytes.

**Solution:** Tiled shared memory staging with transposed layout in LDS:
- TILE_K=64 (fills 32 KB shared memory exactly on RDNA 3.5)
- Coalesced global loads: thread j loads column j
- 2-pass processing: decay+SK in pass 1, state update+output in pass 2
- Only 2-way LDS bank conflicts (acceptable)

**File:** `ggml/src/ggml-vulkan/vulkan-shaders/ssm_recurrence.comp`

**Math per v-head per layer:**
```
Phase 1: s_decayed = s * exp(gate)
Phase 2: sk_j = Σ_i s_decayed[i][j] * k[i],  d_j = (v_j - sk_j) * beta
Phase 3: s_new[i][j] = s_decayed[i][j] + k[i] * d_j
Phase 4: o_j = Σ_i s_new[i][j] * q[i]
```

Dispatch: 32 workgroups × 128 threads = one workgroup per v-head, one thread per state row.

### 2. Fused SSM Recurrence Op (reduces 264 → 24 dispatches/tok)

Replaced 11 separate Vulkan dispatches per SSM layer (MUL, EXP, SUB, etc.) with a single fused compute shader.

**Files modified:**
- `ggml/include/ggml.h` — New `GGML_OP_DELTA_NET_RECURRENCE` opcode
- `ggml/src/ggml.c` — Op registration
- `ggml/src/ggml-cpu/ops.cpp` — CPU fallback
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — Pipeline creation + dispatch
- `src/models/delta-net-base.cpp` — `ggml_cont()` guards for non-contiguous tensors

### 3. Batched Elementwise Mega-Kernel (+7%)

Fuses ~260 tiny elementwise dispatches/token into a single batch dispatch using Buffer Device Addresses (BDA).

**Batched ops:** SILU, EXP, SOFTPLUS, SCALE, MUL, SUB, SIGMOID, RELU, NEG, TANH
**Excluded:** ADD (barrier overhead > savings, +1500 us regression verified)

**File:** `ggml/src/ggml-vulkan/vulkan-shaders/batched_elementwise.comp`

Implementation: SSBO holds op descriptors (2048 capacity), single dispatch processes all. 256 threads/workgroup, 4 elements/thread via specialization constant.

### 4. RDNA3 Wave64 Config (+8.7%)

Set wave64 subgroup size for soft_max, im2col, flash_attn shaders. RDNA 3.5 performs better with wave64 than wave32.

**File:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

### 5. Flash Attention Row-Split for UMA iGPUs

Removed `!device->uma` exclusion — row splitting works on RDNA 3.5 iGPU with shared L2 caching. Target 3 subgroups/SIMD, 20-22 KB shared memory.

### 6. HIP Fused Delta-Net SSM Kernel (+29% on HIP)

Same mathematical fusion as Vulkan but using HIP/CUDA shared memory (64 KB LDS → TILE_K=128).

**File:** `ggml/src/ggml-cuda/delta-net-recurrence.cu` (178 lines)

Spin scheduling for prompt processing dramatically improved pp512 from 240 → 869 tok/s.

---

## Optimizations: Infrastructure & Tools

### 7. MoE Expert Selection Analyzer

Tool to instrument inference and capture per-layer per-token expert routing decisions.

**File:** `tools/moe-analyzer/moe-analyzer.cpp` (981 lines)

Two modes:
- **Collect:** Intercepts `ffn_moe_topk` tensors during inference, streams expert IDs to binary file
- **Analyze:** Computes co-selection matrices, cross-layer transition accuracy, periodicity spectrum (FFT), staleness vs cache simulations (Belady's optimal, EMA, LRU)

Binary format: `MOELOG\x01\x00` header + metadata + per-token/per-layer expert selections.

### 8. APEX Runtime Scheduling (Hybrid CPU-GPU)

Based on APEX paper (arXiv:2506.03296). Framework for dynamically routing ops between CPU and GPU on UMA.

**Files:**
- `common/apex-scheduler.h/cpp` — Scheduler core (77+207 lines)
- `common/uma-profiler.h/cpp` — Roofline-based profiler (85+300 lines)
- `src/llama-context.cpp` — Graph callback integration

**4-Phase Architecture:**
1. **Profiler Hardening** — Per-op timing, batch-size tracking, decode/prefill separation
2. **Critical Inequality Gate** — APEX decision: `ratio < 2*(Tg_lin/Tg_att) + 3 + (Tg_att/Tg_lin)`
3. **Runtime Backend Routing** — Route attention to CPU, FFN to GPU (FFN is bandwidth-bound, attention is compute-bound)
4. **Async Split Overlap** — Event-based wait replaces full backend sync

Three strategies: GPU_ONLY, ASYNC_OVERLAP, ASYMMETRIC_PIPE.

### 9. UMA Auto-Configuration

Auto-detected settings for iGPU systems:
- **Disable mmap** (HIP page locking = 10-100x slower model loading)
- **Reduce CPU threads** to 25% (CPU/GPU share bandwidth on UMA → +10-30% tg)
- **Prefer full GPU offload** (GPU has ~2x effective bandwidth)

**File:** `common/common.cpp`

### 10. Bandwidth-Aware Layer Splitting

**Insight:** On UMA, GPU has ~2x bandwidth advantage. During decode:
- FFN (bandwidth-bound) → keep on GPU
- Attention (compute-bound) → overflow to CPU

New `LAYER_FRACTION_FFN` strategy routes FFN weights to GPU and attention weights to CPU.

### 11. MoE Expert Weight Prefetching

**CPU (ggml-cpu.c):**
- Inter-expert: When starting expert N, prefetch first rows of expert N+1
- Intra-expert: Within 16×16 blocks, prefetch next row before current completes

**GPU UMA (ggml-cuda.cu):**
- Skip bulk MUL_MAT_ID prefetch (wastes 75% bandwidth on top-8/128)
- Per-expert selective prefetch during computation loop
- Each slice fits better in 32 MB Infinity Cache

---

## What Failed (Regressions)

| Attempt | Impact | Root Cause |
|---------|--------|-----------|
| ADD in mega-kernel | +1500 us regression | Barrier/flush overhead > dispatch savings |
| UMA HostVisible allocation | 58→56 tok/s (-3.4%) | HostVisible+HostCoherent = uncached write-combined memory. Use DeviceLocal only. |
| SSM double-read removal | 58.71→54.13 (-7.8%) | Removing shared memory → non-coalesced reads 512B apart |
| MUL_MAT for SSM dots | Garbage output | ggml_mul_mat computes A^T×B, not A×B |
| Fusion plan caching (goto) | 43 tok/s | Skipped state setup, broke dependency tracking |

## Dead Ends

| Attempt | Why It Won't Work |
|---------|-------------------|
| Speculative decoding | Qwen3.5 uses M-RoPE, 4% acceptance rate |
| KV cache quantization | Not supported on Vulkan backend |
| Flash attention tuning | SSM-dominated model, few attention layers |
| Zero-copy mmap | GGUF tensor data not 4096-byte aligned |
| REPEAT op batching | Sequential RAW hazards prevent parallelism |

---

## Vulkan Profiling Breakdown (per 5-token iteration)

| Operation | Time (us) | Count | Per-op | % Total |
|-----------|-----------|-------|--------|---------|
| DELTA_NET_RECURRENCE | 13,758 | 120 | 115 us | 18.3% |
| CPY | 1,571 | 240 | 6.5 | 2.1% |
| MULTI_ADD | 1,447 | 320 | 4.5 | 1.9% |
| GET_ROWS | 1,165 | 248 | 4.7 | 1.6% |
| MUL (broadcast) | 1,108 | 440 | 2.5 | 1.5% |
| TOPK_MOE | 1,060 | 160 | 6.6 | 1.4% |
| RMS_NORM_MUL | 988 | 524 | 1.9 | 1.3% |
| SIGMOID | 625 | 320 | 2.0 | 0.8% |
| GLU | 543 | 320 | 1.7 | 0.7% |

Total: ~75,000 us (5 tokens) = ~15,000 us/tok GPU dispatch time.

---

## Remaining Optimization Headroom

| Opportunity | Est. Impact | Effort |
|-------------|-------------|--------|
| Broadcast MUL batching | +2-3 tok/s | Medium |
| CPY reduction | +1 tok/s | Low |
| CONT elimination | +1 tok/s | Low |
| Expert batch coalescing (server) | +10-15% | High |
| Fused expert-gating + top-K | +5-8% | Medium |
| Matmul shader tuning | Unknown | Very high |
| APEX Phase 4-5 completion | Unknown | High |

---

## Unfinished Work (.unfinished/)

**Vocabulary Pruner** (`llama-vocab-pruner.h`, 166 lines + 707 lines tests):
- Maintains "hot set" of ~4096 likely tokens
- Computes logits only for hot set, uses Cauchy-Schwarz bounds for full-vocab decision
- `|logit_i| ≤ ||row_i|| * ||hidden||` — mathematical guarantee
- For Qwen3.5 (151,936 vocab), could reduce final-layer compute dramatically
- Status: Unintegrated prototype with unit tests

---

## Key Files Reference

### Vulkan Shaders
- `ggml/src/ggml-vulkan/vulkan-shaders/ssm_recurrence.comp` — Fused SSM kernel
- `ggml/src/ggml-vulkan/vulkan-shaders/batched_elementwise.comp` — Mega-kernel

### Core Implementation
- `common/apex-scheduler.h/cpp` — APEX hybrid scheduling
- `common/uma-profiler.h/cpp` — UMA roofline profiler
- `tools/moe-analyzer/moe-analyzer.cpp` — Expert selection analyzer (981 lines)

### Backend Modifications
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — RDNA3 config, fused ops, UMA
- `ggml/src/ggml-cuda/delta-net-recurrence.cu` — HIP SSM kernel
- `ggml/src/ggml-cuda/ggml-cuda.cu` — UMA prefetch
- `ggml/src/ggml-cpu/ggml-cpu.c` — Expert weight prefetch

### Documentation
- `docs/development/adr-moe-token-generation-optimization.md`
- `docs/development/apex-architecture.md`
- `docs/development/fused-ssm-recurrence-spec.md`
- `docs/development/strix-halo-optimization-log.md`
- `docs/development/uma-auto-configuration.md`
- `docs/development/uma-bandwidth-aware-splitting.md`

### Build & Benchmark
- `scripts/bench-strix-halo-moe.sh` — MoE benchmark suite
- Build: `cmd.exe /c "C:\Users\fabia\build_llama.bat"` (Vulkan)
- Shader rebuild: `cmd.exe /c "C:\Users\fabia\build_shader.bat"`

---

## Lessons Learned

1. **Measure first** — Assumptions about bottlenecks are wrong more often than right
2. **Coalesced access is king** — 31/32 cache line waste on non-coalesced reads
3. **32 KB shared memory** on RDNA 3.5 (not 64 KB) — tile sizes must be precise
4. **Wave64 > wave32** on RDNA 3.5 by 8.7%
5. **DeviceLocal only on UMA** — HostCoherent = uncached write-combined = regression
6. **MoE is bandwidth-efficient** — 35B MoE (3B active) beats 4B dense model
7. **Selective prefetch** — Prefetching all 128 experts pollutes 32 MB Infinity Cache
8. **GPU warmup required** — Cold-start is ~24% slower on RDNA 3.5 iGPU
9. **`ggml_mul_mat` = A^T × B** — Not A × B. Verify semantics before fusing.
10. **Sequential SSM kills batching** — Fused kernels > batch dispatches for chain dependencies

---

## How to Resume

```bash
# Verify current performance
cmd.exe /c "C:\Users\fabia\build_llama.bat"
./build-win/bin/llama-bench.exe -m /c/Users/fabia/models/Qwen3.5-35B-A3B-Q4_K_M.gguf -t 4 -ngl 99 -p 0 -n 128

# Run MoE analyzer
./build-win/bin/llama-moe-analyzer.exe collect -m /path/to/model.gguf -p "test prompt"
./build-win/bin/llama-moe-analyzer.exe analyze -i moe_selections.bin

# HIP build
cmd.exe /c "C:\Users\fabia\build_hip_opt.bat"
```

**Branch structure:**
- `backup/strix-halo-optimizations` — All Vulkan + infrastructure work
- `hip-optimization` — HIP-specific fused SSM + spin scheduling
- `rocm-optimization` — ROCm tuning (shares with hip-optimization)
- `claude/kv-cache-compaction-*` — KV cache research (separate workstream)
