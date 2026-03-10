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

## Developer Machine

- **APU**: AMD Ryzen AI MAX+ 395 "Strix Halo"
  - CPU: 16x Zen 5 cores, 32 threads, up to 5.1 GHz
  - GPU: Radeon 8060S — 40 RDNA 3.5 CUs, ~59.4 FP16 TFLOPS peak
  - NPU: XDNA 2, 50 TOPS
- **RAM**: 128 GB unified LPDDR5X-8000 (up to 96 GB allocatable as VRAM)
- **Memory Bandwidth**: ~215 GB/s measured, 256 GB/s theoretical (256-bit bus)
- **Backends**: ROCm/HIP (gfx1151), Vulkan (RADV), CPU
- **Target Model**: Unsloth Qwen3.5-35B-A3B GGUF (MoE, 2 KV heads)
