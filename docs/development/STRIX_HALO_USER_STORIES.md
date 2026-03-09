# User Stories: Strix Halo Optimization (llama.cpp Fork)

## Completed

### US-OPT-1: Fused Delta-Net SSM Recurrence
**As a** GPU kernel developer,
**I want** a single fused Vulkan compute shader for the Delta-Net SSM recurrence,
**So that** 264 per-token dispatches are reduced to 24, eliminating dispatch overhead.

**Acceptance Criteria:**
- [x] New `GGML_OP_DELTA_NET_RECURRENCE` opcode registered
- [x] Vulkan compute shader fusing 11 elementwise ops per SSM layer
- [x] CPU fallback implementation
- [x] Model integration with `ggml_cont()` guards for non-contiguous tensors
- [x] Correctness verified against unfused baseline

**Result:** 264 → 24 dispatches/token. Foundation for shared memory tiling.

---

### US-OPT-2: SSM Shared Memory Tiling (+14.8%)
**As a** GPU kernel developer,
**I want** coalesced memory access in the SSM recurrence shader via shared memory staging,
**So that** cache line utilization goes from 1/32 to full.

**Acceptance Criteria:**
- [x] Transposed state layout in LDS (32 KB, TILE_K=64)
- [x] Coalesced global loads: thread j loads column j
- [x] 2-pass processing: decay+SK in pass 1, state update+output in pass 2
- [x] 128 threads/workgroup, 32 workgroups (one per v-head)
- [x] Performance validated: 58.44 → 67.09 tok/s

**Result:** +14.8% token generation speed. Biggest single win.

---

### US-OPT-3: Batched Elementwise Mega-Kernel (+7%)
**As a** GPU kernel developer,
**I want** to batch hundreds of tiny elementwise dispatches into a single Vulkan dispatch,
**So that** dispatch overhead for SILU/MUL/SIGMOID/etc. is amortized.

**Acceptance Criteria:**
- [x] BDA-based single shader handling 10 op types
- [x] SSBO op descriptors (2048 capacity) with graceful fallback
- [x] Same-shape operands only (broadcast rejected after testing)
- [x] ADD excluded (verified +1500 us regression)
- [x] SIGMOID lookahead for TOPK_MOE fusion compatibility
- [x] 214/214 backend-ops tests pass

**Result:** +7% token generation speed.

---

### US-OPT-4: RDNA3 Wave64 Pipeline Configuration (+8.7%)
**As a** GPU kernel developer,
**I want** wave64 subgroup configuration for RDNA 3.5 shaders,
**So that** the GPU uses optimal occupancy for this architecture.

**Acceptance Criteria:**
- [x] wave64 applied to soft_max, im2col, flash_attn shaders
- [x] Verified wave64 > wave32 on Radeon 8060S
- [x] No regressions on other operations

**Result:** +8.7% vs wave32.

---

### US-OPT-5: Flash Attention UMA iGPU Support
**As a** UMA system user,
**I want** flash attention row-split enabled on iGPU systems,
**So that** attention computation benefits from shared L2 caching.

**Acceptance Criteria:**
- [x] Removed `!device->uma` exclusion for row_split
- [x] RDNA3 iGPU occupancy targeting (3 subgroups/SIMD, 20-22 KB shared mem)
- [x] Validated on Radeon 8060S

---

### US-OPT-6: HIP/ROCm Fused SSM Kernel (+29%)
**As a** HIP developer,
**I want** the fused Delta-Net SSM kernel ported to CUDA/HIP,
**So that** ROCm builds benefit from the same fusion as Vulkan.

**Acceptance Criteria:**
- [x] `delta-net-recurrence.cu` with TILE_K=128 (64 KB LDS on HIP)
- [x] Spin scheduling for prompt processing
- [x] tg128: 43.84 → 56.46 tok/s (+29%)
- [x] pp512: 240.84 → 869.82 tok/s (+261%)

---

### US-OPT-7: MoE Expert Selection Analyzer
**As a** ML engineer,
**I want** a tool to capture and analyze MoE expert routing patterns,
**So that** I can optimize expert weight prefetching and caching strategies.

**Acceptance Criteria:**
- [x] Collect mode: intercept expert selections during inference
- [x] Binary format with streaming writes
- [x] Analyze mode: co-selection matrices, cross-layer transitions, periodicity FFT
- [x] Staleness vs cache simulations (Belady's optimal, EMA, LRU)

**File:** `tools/moe-analyzer/moe-analyzer.cpp` (981 lines)

---

### US-OPT-8: MoE Expert Weight Prefetching
**As a** systems programmer,
**I want** selective expert weight prefetching for CPU and GPU UMA,
**So that** only the top-K active experts are prefetched, not all 128.

**Acceptance Criteria:**
- [x] CPU: Inter-expert and intra-expert `__builtin_prefetch` in MUL_MAT_ID
- [x] GPU UMA: Skip bulk tensor prefetch, add per-expert slice prefetch
- [x] Reduces Infinity Cache pollution (32 MB shared)

---

### US-OPT-9: APEX Runtime Scheduling Framework
**As a** systems architect,
**I want** bandwidth-aware hybrid CPU-GPU scheduling based on the APEX paper,
**So that** attention and FFN layers are routed to the optimal backend on UMA.

**Acceptance Criteria:**
- [x] Phase 1-3 implemented: Profiler, Critical Inequality Gate, Backend Routing
- [x] Three strategies: GPU_ONLY, ASYNC_OVERLAP, ASYMMETRIC_PIPE
- [x] UMA roofline profiler with arithmetic intensity classification
- [x] Graph callback integration in llama-context
- [ ] Phase 4-5: Async overlap and server mode (planned)

**Files:** `common/apex-scheduler.h/cpp`, `common/uma-profiler.h/cpp`

---

### US-OPT-10: UMA Auto-Configuration
**As a** iGPU user,
**I want** automatic detection and configuration for UMA systems,
**So that** performance is optimized without manual tuning.

**Acceptance Criteria:**
- [x] Auto-detect iGPU via GGML_BACKEND_DEVICE_TYPE_IGPU
- [x] Disable mmap (HIP page locking penalty)
- [x] Reduce CPU threads to 25% (bandwidth sharing)
- [x] Prefer full GPU offload

---

### US-OPT-11: Bandwidth-Aware Layer Splitting
**As a** systems architect,
**I want** FFN layers kept on GPU and attention overflowed to CPU,
**So that** bandwidth-bound ops use the faster GPU path on UMA.

**Acceptance Criteria:**
- [x] `LAYER_FRACTION_FFN` strategy implemented
- [x] FFN weights stay on GPU, attention weights overflow to CPU
- [x] Automatic based on UMA profiler roofline classification

---

## Documented Regressions (Do Not Repeat)

### US-OPT-R1: ADD in Mega-Kernel → +1500 us REGRESSION
### US-OPT-R2: UMA HostVisible Allocation → -2 tok/s REGRESSION
### US-OPT-R3: SSM Double-Read Removal → -7.8% REGRESSION
### US-OPT-R4: Speculative Decoding → 4% acceptance (DEAD END)
### US-OPT-R5: KV Cache Quantization → Not supported on Vulkan (DEAD END)

---

## Planned

### US-OPT-12: Broadcast MUL Batching (+2-3 tok/s)
### US-OPT-13: CPY Reduction (+1 tok/s)
### US-OPT-14: CONT Elimination (+1 tok/s)
### US-OPT-15: Sparse Expert Batch Coalescing (+10-15% server)
### US-OPT-16: Fused Expert-Gating + Top-K Kernel (+5-8%)
### US-OPT-17: Vocabulary Pruner Integration (prototype in .unfinished/)
