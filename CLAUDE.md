# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Task Management & Coordination

Organize todo items for optimized coordination and parallelization. It's better to have many small to do items than a few large ones. Apply best practises:
- 1 todo item can only be scoped to 1 file.
- 1 todo item can only be scoped to 1 specialized agent. If the todo also has tasks for a different specialized agent, the tasks must be split.
- If there are many tasks for 1 agent type, spawn and assign multiple agents of the same type.
- Update the todo list frequently, at least after finishing a todo item (so on every todo item).
- The item must be delivered in production-ready quality.

## Work Process

- Analyze current state, design desired state and how to get there. Back with Mermaid diagrams for state and flow or other diagrams relevant to the task.
- Implement and wire the features.
- Don't simplify or use stubs or todos etc.
- Make sure not to overwrite or delete existing functionality, especially when it's using SOTA models or algorithms.
- When done, verify your work. It must be delivered in production-ready quality.

## Build Commands

```bash
# Standard CMake build (Vulkan, Windows)
cmake -B build -DGGML_VULKAN=ON -G Ninja && cmake --build build --config Release

# Using presets (see CMakePresets.json for full list)
cmake --preset x64-windows-msvc-release-vulkan && cmake --build build-x64-windows-msvc-release-vulkan

# Fork-specific build scripts (Strix Halo)
cmd.exe /c "C:\Users\fabia\build_llama.bat"        # Full Vulkan build
cmd.exe /c "C:\Users\fabia\build_shader.bat"        # Shader rebuild only
cmd.exe /c "C:\Users\fabia\build_hip_opt.bat"       # HIP/ROCm optimized build

# Rebuild Vulkan shaders after modifying .comp files
# Shader gen is an ExternalProject — rebuild separately when vulkan-shaders-gen.cpp changes
# Build dir: build-win/ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build/
```

Key CMake options: `GGML_VULKAN`, `GGML_CUDA`, `GGML_METAL`, `GGML_SYCL`, `GGML_HIP`, `GGML_BLAS`, `GGML_RPC`.

## Testing

```bash
# Run all tests
ctest --test-dir build

# Run a single test by name
ctest --test-dir build -R test-chat

# Run backend-ops tests (essential after modifying ggml operators)
./build/bin/test-backend-ops

# Run with label filter
ctest --test-dir build -L main

# Benchmark (Vulkan, tg128)
./build/bin/llama-bench -m model.gguf -t 4 -ngl 99 -p 0 -n 128

# Perplexity check (verify no quality regression)
./build/bin/llama-perplexity -m model.gguf -f wikitext-2.txt
```

Test files live in `tests/`. CMake helpers: `llama_build()`, `llama_test()`, `llama_build_and_test()`.

## Architecture

### Layer Diagram

```
┌─────────────────────────────────────────────────┐
│  Tools & Examples (tools/, examples/)            │
│  llama-cli, llama-server, llama-bench, etc.      │
├─────────────────────────────────────────────────┤
│  Common Library (common/)                        │
│  arg parsing, sampling, chat templates, logging  │
├─────────────────────────────────────────────────┤
│  libllama (src/)                                 │
│  Model loading, graph building, KV cache,        │
│  inference, vocab, adapters, memory management   │
├─────────────────────────────────────────────────┤
│  ggml (ggml/)                                    │
│  Tensor ops, quantization, backend abstraction   │
├──────────┬──────────┬──────────┬────────────────┤
│ CPU      │ Vulkan   │ CUDA/HIP │ Metal/SYCL/... │
│ (ggml-   │ (ggml-   │ (ggml-   │ (17 backends   │
│  cpu/)   │  vulkan/)│  cuda/)  │  total)        │
└──────────┴──────────┴──────────┴────────────────┘
```

### Key Directories

- **`src/`** — Core `libllama` library. `llama-context.cpp` (inference engine), `llama-graph.cpp` (computation graph), `llama-arch.cpp` (architecture defs)
- **`src/models/`** — 70+ model architecture implementations (one file per arch: `llama.cpp`, `qwen2.cpp`, `phi3.cpp`, etc.)
- **`ggml/`** — Tensor library. `ggml.c` (ops), `ggml-backend.cpp` (backend abstraction), `ggml-quants.c` (quantization)
- **`ggml/src/ggml-vulkan/`** — Vulkan backend. Shaders in `vulkan-shaders/*.comp`, generated via `vulkan-shaders-gen.cpp`
- **`ggml/src/ggml-cuda/`** — CUDA/HIP backend. Per-op kernel files
- **`common/`** — Shared utilities: `arg.cpp` (CLI parsing), `sampling.cpp`, `chat.cpp` (149KB, template handling)
- **`tools/`** — CLI tools: `llama-cli`, `llama-server`, `llama-bench`, `llama-quantize`, etc.
- **`include/`** — Public API: `llama.h` (1560 lines), `ggml.h`

### Critical Conventions

- **Matrix multiply is transposed**: `C = ggml_mul_mat(ctx, A, B)` computes C = B × A^T (NOT A × B)
- **Dimension naming**: dim 0 = columns, dim 1 = rows, dim 2 = matrices (row-major storage)
- **Op registration**: New ops need changes in `ggml.h` (enum), `ggml.c` (registration), backend `.cpp` (implementation), and CPU fallback in `ggml-cpu/ops.cpp`
- **Vulkan shader pipeline**: `.comp` source → `vulkan-shaders-gen` → SPIR-V → embedded in `ggml-vulkan.cpp` via `string_to_spv()`

## Coding Standards

- **Style**: `snake_case` everywhere, 4 spaces indent, LF endings, `void * ptr`, `int & a`
- **Naming**: `<class>_<action>_<noun>` pattern (e.g., `llama_model_init()`)
- **Enums**: UPPER_CASE prefixed with enum name (e.g., `LLAMA_VOCAB_TYPE_BPE`)
- **Types**: `int32_t`, `size_t` in public APIs. Avoid `typedef struct`; use `struct foo {}` directly
- **Dependencies**: Minimize third-party deps. Basic `for` loops over fancy STL. No unnecessary templates
- **Formatting**: `clang-format` v15+ (`.clang-format` in root). Vertical alignment preferred
- **Files**: C/C++ names are lowercase with dashes, `.h`/`.c`/`.cpp` extensions

## Fork-Specific Work (fabiantax/llama.cpp)

This fork targets **Strix Halo** (AMD Ryzen AI Max+ 395, Radeon 8060S RDNA 3.5, 68GB UMA) optimizations for **Qwen3.5-35B-A3B** (hybrid SSM+MoE model).

### Optimization branches
- `backup/strix-halo-optimizations` — Vulkan + infrastructure (primary)
- `hip-optimization` / `rocm-optimization` — HIP-specific fused SSM + spin scheduling
- `claude/kv-cache-compaction-*` — KV cache research + GraphRAG pipeline

### Key fork files
- `ggml/src/ggml-vulkan/vulkan-shaders/ssm_recurrence.comp` — Fused SSM kernel with shared memory tiling
- `ggml/src/ggml-vulkan/vulkan-shaders/batched_elementwise.comp` — Mega-kernel batching ~260 dispatches
- `ggml/src/ggml-cuda/delta-net-recurrence.cu` — HIP fused SSM (TILE_K=128, 64KB LDS)
- `common/apex-scheduler.h/cpp` — APEX hybrid CPU-GPU scheduling
- `tools/kv-compact/` — KV cache compaction POC
- `graphrag-pipeline/` — Rust NER+RE extraction + Python ModernBERT training

### Documented regressions (do not repeat)
- ADD in mega-kernel → +1500 us regression (barrier overhead)
- UMA HostVisible allocation → -2 tok/s (use eDeviceLocal only)
- SSM double-read removal → -7.8% (non-coalesced access 512B apart)
- GPU shared memory is 32 KB on RDNA 3.5 (not 64 KB) — TILE_K=64 for Vulkan

### Reports
- `docs/development/STRIX_HALO_OPTIMIZATION_REPORT.md` — Full optimization report with profiling data
- `docs/development/STRIX_HALO_USER_STORIES.md` — User stories tracking
- `graphrag-pipeline/training/PIPELINE_STATUS.md` — ModernBERT fine-tuning status

PRs go to **fabiantax/llama.cpp** only — never upstream ggml-org/llama.cpp.
