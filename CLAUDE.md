IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Git Commit Rules

- **NEVER use "Claude" as the git author.** All commits must use the repository’s configured git user identity (`Fabian Tax <fabiantax@hotmail.com>`). Do not set GIT_AUTHOR_NAME or GIT_AUTHOR_EMAIL to Claude/Anthropic values.
- When resolving rebase conflicts, ensure the original author is preserved or reset to the repo owner — never to Claude.
- The `Co-Authored-By: Claude` trailer in the commit message body is fine, but the Author field must always be the repo owner.

## Task Management & Coordination

Organize todo items for optimized coordination and parallelization. It’s better to have many small to do items than a few large ones. Apply best practises:
- 1 todo item can only be scoped to 1 file.
- 1 todo item can only be scoped to 1 specialized agent. If the todo also has tasks for a different specialized agent, the tasks must be split.
- If there are many tasks for 1 agent type, spawn and assign multiple agents of the same type.
- Update the todo list frequently, at least after finishing a todo item (so on every todo item).
- The item must be delivered in production-ready quality.

## Task Confidence Tracking

- For each task or sub-task, output a **confidence %** estimate before starting work.
- If confidence falls below **70%**, search for more information (code, docs, arXiv papers, web) to increase confidence before proceeding.
- Use strategies like mermaid diagrams, hex dumps, or structured analysis to raise confidence when dealing with complex formats or protocols.
- Re-evaluate confidence after each significant discovery or blocker.

## Work Process

- Analyze current state, design desired state and how to get there. Back with Mermaid diagrams for state and flow or other diagrams relevant to the task.
- Implement and wire the features.
- Don’t simplify or use stubs or todos etc.
- Make sure not to overwrite or delete existing functionality, especially when it’s using SOTA models or algorithms.
- When done, verify your work. It must be delivered in production-ready quality.

## Developer Machine

- **APU**: AMD Ryzen AI MAX+ 395 "Strix Halo"
  - CPU: 16x Zen 5 cores, 32 threads, up to 5.1 GHz
  - GPU: Radeon 8060S — 40 RDNA 3.5 CUs, ~59.4 FP16 TFLOPS peak
  - NPU: XDNA 2, 50 TOPS
- **RAM**: 128 GB unified LPDDR5X-8000 (up to 96 GB allocatable as VRAM)
- **Memory Bandwidth**: ~215 GB/s measured, 256 GB/s theoretical (256-bit bus)
- **Backends**: ROCm/HIP (gfx1151), Vulkan (RADV), CPU
- **Target Model**: Unsloth Qwen3.5-35B-A3B GGUF (MoE, 2 KV heads)

## Goal

Developer building with AI agents. Target: **500 t/s token generation** (10x over current ~50 t/s baseline).

Key lever: **Attention Matching** — KV cache compression up to 50x with minimal accuracy loss.
With a 10-50x smaller KV cache, the memory bandwidth bottleneck (~215 GB/s) is spent reading weights
rather than a bloated cache, unlocking dramatically higher throughput for multi-agent workloads.

Path to 500 t/s:
- Qwen3.5-35B-A3B only activates ~3B params per token (~6 GB weights at Q4)
- At 215 GB/s bandwidth, theoretical peak for 6 GB active weights ≈ 35 t/s per stream
- With 50x KV compaction, KV reads become negligible → nearly all bandwidth goes to weights
- Parallel batching across agents: 8-16 concurrent streams sharing prefill amortizes overhead
- Combined batched throughput target: **500+ t/s aggregate across concurrent agent sessions**
