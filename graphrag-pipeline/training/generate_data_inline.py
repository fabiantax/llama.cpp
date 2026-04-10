"""Generate NER + RE training data from expert annotations.

This generates training data directly without API calls, using
domain knowledge of GPU optimization research.
"""
import json
import os
import random
from itertools import combinations
from pathlib import Path

from schema import ENTITY_TYPES, RELATION_TYPES, REL_TYPE2ID

random.seed(42)


def find_all(text: str, sub: str) -> list[int]:
    """Find all occurrences of substring in text."""
    starts = []
    start = 0
    while True:
        idx = text.find(sub, start)
        if idx == -1:
            break
        starts.append(idx)
        start = idx + 1
    return starts


def annotate(text: str, entity_defs: list[tuple[str, str]],
             relation_defs: list[tuple[int, int, str]] | None = None) -> dict:
    """Create annotation from entity definitions.

    entity_defs: list of (text_to_find, entity_type) - finds first occurrence
    relation_defs: list of (head_entity_idx, tail_entity_idx, relation_type)
    """
    entities = []
    for ent_text, ent_type in entity_defs:
        idx = text.find(ent_text)
        if idx >= 0:
            entities.append({
                "start": idx,
                "end": idx + len(ent_text),
                "label": ent_type,
                "text": ent_text,
            })
        else:
            # Try case-insensitive
            idx = text.lower().find(ent_text.lower())
            if idx >= 0:
                entities.append({
                    "start": idx,
                    "end": idx + len(ent_text),
                    "label": ent_type,
                    "text": text[idx:idx + len(ent_text)],
                })

    relations = []
    if relation_defs:
        for h_idx, t_idx, rel_type in relation_defs:
            if h_idx < len(entities) and t_idx < len(entities):
                relations.append({
                    "head_idx": h_idx,
                    "tail_idx": t_idx,
                    "relation": rel_type,
                })

    return {"entities": entities, "relations": relations}


# ── Annotated examples ─────────────────────────────────────────────────
# Each example: (text, entity_defs, relation_defs)

EXAMPLES = [
    # FlashRNN paper examples
    (
        "While Transformers and other sequence-parallelizable neural network architectures seem like the current state of the art in sequence modeling, they specifically lack state-tracking capabilities. These are important for time-series tasks and logical reasoning. Traditional RNNs like LSTMs and GRUs, as well as modern variants like sLSTM do have these capabilities at the cost of strictly sequential processing.",
        [
            ("Transformers", "model_architecture"),
            ("LSTMs", "model_architecture"),
            ("GRUs", "model_architecture"),
            ("sLSTM", "model_architecture"),
            ("state-tracking capabilities", "gpu_feature"),
            ("sequential processing", "constraint"),
            ("sequence modeling", "algorithm"),
        ],
        [
            (0, 4, "LIMITS"),  # Transformers lack state-tracking
            (1, 4, "ENABLES"),  # LSTMs enable state-tracking
            (2, 4, "ENABLES"),  # GRUs enable state-tracking
            (3, 4, "ENABLES"),  # sLSTM enable state-tracking
            (5, 1, "LIMITS"),  # sequential processing limits LSTMs
        ],
    ),
    (
        "We show how fast these networks can get with our hardware-optimization FlashRNN in Triton and CUDA, optimizing kernels to the register level on modern GPUs. We extend traditional RNNs with a parallelization variant that processes multiple RNNs of smaller hidden state in parallel, similar to the head-wise processing in Transformers.",
        [
            ("FlashRNN", "software_framework"),
            ("Triton", "software_framework"),
            ("CUDA", "software_framework"),
            ("register level", "memory_pattern"),
            ("parallelization variant", "optimization_technique"),
            ("head-wise processing", "optimization_technique"),
            ("Transformers", "model_architecture"),
        ],
        [
            (0, 1, "IMPLEMENTS"),
            (0, 2, "IMPLEMENTS"),
            (0, 3, "OPTIMIZES"),
            (4, 0, "IS_PART_OF"),
            (5, 6, "IS_PART_OF"),
        ],
    ),
    (
        "FlashAttention introduced an IO-aware attention algorithm and corresponding CUDA kernel for GPUs. It minimizes memory reads and writes between GPU high bandwidth memory (HBM) and on-chip SRAM, and is asymptotically the fastest attention implementation.",
        [
            ("FlashAttention", "algorithm"),
            ("IO-aware attention", "optimization_technique"),
            ("CUDA kernel", "kernel_operation"),
            ("HBM", "hardware"),
            ("SRAM", "hardware"),
            ("memory reads and writes", "performance_metric"),
        ],
        [
            (0, 1, "IMPLEMENTS"),
            (0, 2, "USES"),
            (1, 5, "REDUCES"),
            (1, 3, "TARGETS"),
            (1, 4, "TARGETS"),
        ],
    ),
    # GPU optimization domain examples (from memory/knowledge base)
    (
        "The DELTA_NET_RECURRENCE kernel takes 115 microseconds per dispatch on Radeon 8060S with RDNA 3.5 architecture. This is the number one non-matmul cost at 2760 microseconds per token across 24 SSM layers.",
        [
            ("DELTA_NET_RECURRENCE", "kernel_operation"),
            ("115 microseconds", "performance_metric"),
            ("Radeon 8060S", "hardware"),
            ("RDNA 3.5", "gpu_feature"),
            ("2760 microseconds per token", "performance_metric"),
            ("SSM layers", "model_architecture"),
        ],
        [
            (0, 1, "MEASURES"),
            (0, 2, "TARGETS"),
            (2, 3, "IS_FEATURE_OF"),
            (0, 5, "IS_PART_OF"),
        ],
    ),
    (
        "Vulkan shared memory tiling improved SSM inference from 58.44 to 67.09 tokens per second, a 14.8% speedup. The technique uses transposed layout in LDS with TILE_K=64 to fit within the 32 KB shared memory limit.",
        [
            ("shared memory tiling", "optimization_technique"),
            ("SSM inference", "kernel_operation"),
            ("67.09 tokens per second", "performance_metric"),
            ("14.8% speedup", "performance_metric"),
            ("transposed layout", "memory_pattern"),
            ("LDS", "hardware"),
            ("TILE_K=64", "constraint"),
            ("32 KB", "constraint"),
        ],
        [
            (0, 1, "IMPROVES"),
            (0, 2, "MEASURES"),
            (0, 3, "MEASURES"),
            (0, 4, "USES"),
            (4, 5, "TARGETS"),
            (6, 7, "LIMITS"),
        ],
    ),
    (
        "The batched elementwise mega-kernel fuses SILU, SIGMOID, SCALE, and MUL operations into a single Vulkan compute dispatch. This reduces per-dispatch overhead and achieves a 7% improvement in token generation speed.",
        [
            ("batched elementwise mega-kernel", "optimization_technique"),
            ("SILU", "kernel_operation"),
            ("SIGMOID", "kernel_operation"),
            ("SCALE", "kernel_operation"),
            ("MUL", "kernel_operation"),
            ("Vulkan", "software_framework"),
            ("7% improvement", "performance_metric"),
            ("token generation speed", "performance_metric"),
        ],
        [
            (0, 1, "USES"),
            (0, 2, "USES"),
            (0, 3, "USES"),
            (0, 4, "USES"),
            (0, 5, "TARGETS"),
            (0, 6, "MEASURES"),
            (0, 7, "IMPROVES"),
        ],
    ),
    (
        "Wave64 configuration is 8.7% faster than wave32 on RDNA 3.5 for soft_max, im2col, and flash_attn shaders. The performance regression with wave32 confirms that RDNA 3.5 benefits from wider wavefronts.",
        [
            ("Wave64", "optimization_technique"),
            ("wave32", "optimization_technique"),
            ("RDNA 3.5", "gpu_feature"),
            ("soft_max", "kernel_operation"),
            ("im2col", "kernel_operation"),
            ("flash_attn", "kernel_operation"),
            ("8.7%", "performance_metric"),
        ],
        [
            (0, 2, "TARGETS"),
            (0, 3, "IMPROVES"),
            (0, 4, "IMPROVES"),
            (0, 5, "IMPROVES"),
            (0, 6, "MEASURES"),
            (1, 0, "COMPETES_WITH"),
        ],
    ),
    (
        "The Qwen3.5-35B-A3B model is a hybrid SSM plus MoE architecture with both MoE transformer layers and Delta-Net SSM layers. It has 24 SSM layers per token with DELTA_NET_RECURRENCE and SSM_CONV dispatches.",
        [
            ("Qwen3.5-35B-A3B", "model_architecture"),
            ("SSM", "algorithm"),
            ("MoE", "algorithm"),
            ("Delta-Net", "algorithm"),
            ("DELTA_NET_RECURRENCE", "kernel_operation"),
            ("SSM_CONV", "kernel_operation"),
            ("24 SSM layers", "constraint"),
        ],
        [
            (0, 1, "USES"),
            (0, 2, "USES"),
            (0, 3, "USES"),
            (4, 0, "IS_PART_OF"),
            (5, 0, "IS_PART_OF"),
            (6, 0, "LIMITS"),
        ],
    ),
    (
        "KV cache quantization using int8 is not supported on the Vulkan backend, causing context creation failures. Flash attention with -fa 1 has no effect on token generation speed for this model because SSM layers dominate.",
        [
            ("KV cache quantization", "optimization_technique"),
            ("int8", "optimization_technique"),
            ("Vulkan", "software_framework"),
            ("Flash attention", "algorithm"),
            ("token generation speed", "performance_metric"),
            ("SSM layers", "model_architecture"),
        ],
        [
            (0, 2, "LIMITS"),
            (3, 4, "MEASURES"),
            (5, 4, "LIMITS"),
        ],
    ),
    (
        "UMA HostVisible allocation causes a 2 tok/s regression due to uncached write-combined memory access patterns. The correct approach is to prefer eDeviceLocal only, which gives GPU-optimized caching on UMA system RAM.",
        [
            ("UMA HostVisible allocation", "memory_pattern"),
            ("2 tok/s regression", "performance_metric"),
            ("write-combined memory", "memory_pattern"),
            ("eDeviceLocal", "memory_pattern"),
            ("UMA system RAM", "hardware"),
        ],
        [
            (0, 1, "MEASURES"),
            (0, 2, "USES"),
            (3, 0, "COMPETES_WITH"),
            (3, 4, "TARGETS"),
        ],
    ),
    (
        "MUL_MAT_ID only reads active expert weights, 8 out of 256 total experts, making MoE inference bandwidth-efficient. The TOPK_MOE kernel selects which experts to activate based on router logits.",
        [
            ("MUL_MAT_ID", "kernel_operation"),
            ("expert weights", "data_structure"),
            ("MoE inference", "algorithm"),
            ("TOPK_MOE", "kernel_operation"),
            ("router logits", "data_structure"),
        ],
        [
            (0, 1, "USES"),
            (0, 2, "IMPROVES"),
            (3, 4, "USES"),
            (3, 0, "ENABLES"),
        ],
    ),
    (
        "The fused RMS_NORM_MUL operation combines root mean square normalization with element-wise multiplication into a single kernel dispatch, reducing memory bandwidth overhead in transformer inference.",
        [
            ("RMS_NORM_MUL", "kernel_operation"),
            ("root mean square normalization", "algorithm"),
            ("element-wise multiplication", "kernel_operation"),
            ("memory bandwidth", "performance_metric"),
            ("transformer inference", "algorithm"),
        ],
        [
            (0, 1, "IMPLEMENTS"),
            (0, 2, "IMPLEMENTS"),
            (0, 3, "REDUCES"),
            (0, 4, "IMPROVES"),
        ],
    ),
    (
        "Speculative decoding achieved only 4.65 tokens per second because M-RoPE is incompatible with the draft model architecture. The overhead of verification rounds negates any potential speedup for SSM-heavy models.",
        [
            ("Speculative decoding", "optimization_technique"),
            ("4.65 tokens per second", "performance_metric"),
            ("M-RoPE", "algorithm"),
            ("draft model", "model_architecture"),
            ("verification rounds", "constraint"),
            ("SSM-heavy models", "model_architecture"),
        ],
        [
            (0, 1, "MEASURES"),
            (2, 3, "LIMITS"),
            (4, 0, "LIMITS"),
            (0, 5, "TARGETS"),
        ],
    ),
    # More FlashRNN paper content
    (
        "To realize the shown speed-ups, we fuse the recurrent matrix-multiplication part with the point-wise activation part, both wrapped in the sequential loop into one kernel. This can be used on different GPUs and with different state and gate variants.",
        [
            ("recurrent matrix-multiplication", "kernel_operation"),
            ("point-wise activation", "kernel_operation"),
            ("sequential loop", "algorithm"),
            ("kernel", "optimization_technique"),
        ],
        [
            (3, 0, "USES"),
            (3, 1, "USES"),
            (3, 2, "USES"),
        ],
    ),
    (
        "For the auto-optimization we introduce an integer constraint satisfaction library ConstrINT. With this library, one can model generic integer CSP problems with equality, inequality and divisibility constraints as these can model size constraints on modern hardware with specific tensor-core, register and SRAM memory sizes.",
        [
            ("ConstrINT", "software_framework"),
            ("integer CSP", "algorithm"),
            ("tensor-core", "hardware"),
            ("register", "hardware"),
            ("SRAM", "hardware"),
        ],
        [
            (0, 1, "IMPLEMENTS"),
            (0, 2, "TARGETS"),
            (0, 3, "TARGETS"),
            (0, 4, "TARGETS"),
        ],
    ),
    (
        "Our kernels can achieve 50x speed-ups over a vanilla PyTorch implementation and allow 40x larger hidden sizes compared to our Triton implementation. The CUDA kernels operate at near peak FLOPS on NVIDIA A100 and H100 GPUs.",
        [
            ("50x speed-ups", "performance_metric"),
            ("PyTorch", "software_framework"),
            ("40x larger hidden sizes", "performance_metric"),
            ("Triton", "software_framework"),
            ("CUDA kernels", "kernel_operation"),
            ("NVIDIA A100", "hardware"),
            ("H100", "hardware"),
            ("FLOPS", "performance_metric"),
        ],
        [
            (4, 0, "MEASURES"),
            (4, 1, "COMPETES_WITH"),
            (4, 2, "MEASURES"),
            (4, 3, "COMPETES_WITH"),
            (4, 5, "TARGETS"),
            (4, 6, "TARGETS"),
            (4, 7, "MEASURES"),
        ],
    ),
    (
        "The forward pass stores intermediate activations in HBM to be reloaded during the backward pass. FlashRNN minimizes these HBM accesses by keeping data in registers and SRAM as long as possible.",
        [
            ("forward pass", "algorithm"),
            ("HBM", "hardware"),
            ("backward pass", "algorithm"),
            ("FlashRNN", "software_framework"),
            ("registers", "hardware"),
            ("SRAM", "hardware"),
        ],
        [
            (0, 1, "USES"),
            (3, 1, "REDUCES"),
            (3, 4, "USES"),
            (3, 5, "USES"),
        ],
    ),
    # Additional GPU optimization domain examples
    (
        "AMD Ryzen AI Max+ 395 processor features the Radeon 8060S integrated GPU with RDNA 3.5 architecture and 40 compute units. The system uses 68GB LPDDR5X unified memory with approximately 212 GB/s measured bandwidth.",
        [
            ("AMD Ryzen AI Max+ 395", "hardware"),
            ("Radeon 8060S", "hardware"),
            ("RDNA 3.5", "gpu_feature"),
            ("40 compute units", "gpu_feature"),
            ("68GB LPDDR5X", "hardware"),
            ("212 GB/s", "performance_metric"),
        ],
        [
            (1, 0, "IS_PART_OF"),
            (2, 1, "IS_FEATURE_OF"),
            (3, 1, "IS_FEATURE_OF"),
            (4, 0, "IS_PART_OF"),
            (5, 4, "MEASURES"),
        ],
    ),
    (
        "The Vulkan backend achieves 67.09 tokens per second with warm GPU on the shared memory tiled SSM shader. This represents 60% bandwidth efficiency at 127 out of 212 GB/s theoretical peak.",
        [
            ("Vulkan", "software_framework"),
            ("67.09 tokens per second", "performance_metric"),
            ("shared memory tiled SSM shader", "optimization_technique"),
            ("60% bandwidth efficiency", "performance_metric"),
            ("127 out of 212 GB/s", "performance_metric"),
        ],
        [
            (0, 1, "MEASURES"),
            (2, 1, "MEASURES"),
            (0, 3, "MEASURES"),
        ],
    ),
    (
        "HIP optimized build reaches 56.46 tokens per second with fused SSM plus spin scheduling and TILE_K=128. The master baseline without optimizations achieves only 43.84 tokens per second.",
        [
            ("HIP", "software_framework"),
            ("56.46 tokens per second", "performance_metric"),
            ("fused SSM", "optimization_technique"),
            ("spin scheduling", "optimization_technique"),
            ("TILE_K=128", "constraint"),
            ("43.84 tokens per second", "performance_metric"),
        ],
        [
            (0, 1, "MEASURES"),
            (2, 1, "IMPROVES"),
            (3, 1, "IMPROVES"),
            (4, 2, "ENABLES"),
        ],
    ),
    (
        "The GLU fused operation combines gated linear unit computation into a single dispatch. Similarly, MUL_MAT_ADD fuses matrix multiplication with bias addition to reduce kernel launch overhead.",
        [
            ("GLU", "kernel_operation"),
            ("gated linear unit", "algorithm"),
            ("MUL_MAT_ADD", "kernel_operation"),
            ("matrix multiplication", "kernel_operation"),
            ("bias addition", "kernel_operation"),
            ("kernel launch overhead", "performance_metric"),
        ],
        [
            (0, 1, "IMPLEMENTS"),
            (2, 3, "IMPLEMENTS"),
            (2, 4, "IMPLEMENTS"),
            (2, 5, "REDUCES"),
        ],
    ),
    (
        "The CPY operations account for 1571 microseconds across 240 dispatches per iteration. GET_ROWS adds another 1165 microseconds for 248 dispatches, primarily from embedding lookups during token generation.",
        [
            ("CPY", "kernel_operation"),
            ("1571 microseconds", "performance_metric"),
            ("GET_ROWS", "kernel_operation"),
            ("1165 microseconds", "performance_metric"),
            ("embedding lookups", "kernel_operation"),
            ("token generation", "algorithm"),
        ],
        [
            (0, 1, "MEASURES"),
            (2, 3, "MEASURES"),
            (4, 2, "IS_PART_OF"),
            (4, 5, "IS_PART_OF"),
        ],
    ),
    (
        "GGUF tensor data pointers are not 4096-aligned within the file, which prevents VK_EXT_external_memory_host from working for zero-copy model loading. The buffer_from_host_ptr function fails silently when alignment requirements are not met.",
        [
            ("GGUF", "data_structure"),
            ("4096-aligned", "constraint"),
            ("VK_EXT_external_memory_host", "gpu_feature"),
            ("zero-copy model loading", "optimization_technique"),
            ("buffer_from_host_ptr", "kernel_operation"),
        ],
        [
            (1, 2, "LIMITS"),
            (0, 3, "LIMITS"),
            (2, 3, "ENABLES"),
            (4, 2, "USES"),
        ],
    ),
    (
        "Direct memory access through HostVisible plus HostCoherent buffer flags on UMA systems causes performance degradation because the GPU accesses uncached write-combined memory regions instead of using the L2 cache.",
        [
            ("HostVisible", "memory_pattern"),
            ("HostCoherent", "memory_pattern"),
            ("UMA", "gpu_feature"),
            ("write-combined memory", "memory_pattern"),
            ("L2 cache", "hardware"),
        ],
        [
            (0, 2, "TARGETS"),
            (1, 2, "TARGETS"),
            (3, 4, "COMPETES_WITH"),
        ],
    ),
    (
        "The ROPE positional encoding is applied as a kernel operation taking 213 microseconds across 80 dispatches. The CONCAT operation adds 328 microseconds for 120 dispatches combining attention heads.",
        [
            ("ROPE", "kernel_operation"),
            ("positional encoding", "algorithm"),
            ("213 microseconds", "performance_metric"),
            ("CONCAT", "kernel_operation"),
            ("328 microseconds", "performance_metric"),
            ("attention heads", "data_structure"),
        ],
        [
            (0, 1, "IMPLEMENTS"),
            (0, 2, "MEASURES"),
            (3, 4, "MEASURES"),
            (3, 5, "USES"),
        ],
    ),
    (
        "Mamba is a selective state space model architecture that uses input-dependent gating for efficient sequence processing. Unlike Transformers, SSMs have linear scaling with sequence length but lack in-context learning ability.",
        [
            ("Mamba", "model_architecture"),
            ("selective state space model", "algorithm"),
            ("input-dependent gating", "optimization_technique"),
            ("Transformers", "model_architecture"),
            ("SSMs", "algorithm"),
            ("sequence length", "constraint"),
            ("in-context learning", "gpu_feature"),
        ],
        [
            (0, 1, "IMPLEMENTS"),
            (0, 2, "USES"),
            (4, 5, "IMPROVES"),
            (3, 6, "ENABLES"),
            (0, 3, "COMPETES_WITH"),
        ],
    ),
    (
        "The block-diagonal recurrent matrix structure reduces both parameter count and computation by processing multiple smaller RNN heads in parallel. Each head has independent recurrent weights enabling head-wise parallelism.",
        [
            ("block-diagonal recurrent matrix", "data_structure"),
            ("parameter count", "performance_metric"),
            ("RNN heads", "model_architecture"),
            ("head-wise parallelism", "optimization_technique"),
            ("recurrent weights", "data_structure"),
        ],
        [
            (0, 1, "REDUCES"),
            (0, 2, "ENABLES"),
            (3, 2, "TARGETS"),
            (4, 2, "IS_PART_OF"),
        ],
    ),
    (
        "DirectML execution provider enables GPU acceleration for ONNX Runtime inference on Windows AMD GPUs. However, the MatMulInteger operations in int8 quantized models produce incorrect results on DirectML.",
        [
            ("DirectML", "software_framework"),
            ("ONNX Runtime", "software_framework"),
            ("AMD GPUs", "hardware"),
            ("MatMulInteger", "kernel_operation"),
            ("int8 quantized models", "optimization_technique"),
        ],
        [
            (0, 1, "EXTENDS"),
            (0, 2, "TARGETS"),
            (3, 0, "LIMITS"),
            (4, 0, "LIMITS"),
        ],
    ),
    (
        "The RMS_NORM_MUL kernel takes 988 microseconds per iteration across 524 dispatches. This is an already-fused operation combining root mean square normalization with element-wise scaling.",
        [
            ("RMS_NORM_MUL", "kernel_operation"),
            ("988 microseconds", "performance_metric"),
            ("524 dispatches", "performance_metric"),
            ("root mean square normalization", "algorithm"),
            ("element-wise scaling", "kernel_operation"),
        ],
        [
            (0, 1, "MEASURES"),
            (0, 2, "MEASURES"),
            (0, 3, "IMPLEMENTS"),
            (0, 4, "IMPLEMENTS"),
        ],
    ),
    (
        "Flash attention with row_split enabled provides better occupancy on AMD Strix Halo UMA iGPUs. The occupancy tuning is specific to RDNA 3 architecture where shared memory is limited to 32 KB per workgroup.",
        [
            ("Flash attention", "algorithm"),
            ("row_split", "optimization_technique"),
            ("AMD Strix Halo", "hardware"),
            ("UMA", "gpu_feature"),
            ("RDNA 3", "gpu_feature"),
            ("32 KB", "constraint"),
        ],
        [
            (1, 0, "IMPROVES"),
            (0, 2, "TARGETS"),
            (3, 2, "IS_FEATURE_OF"),
            (4, 2, "IS_FEATURE_OF"),
            (5, 4, "LIMITS"),
        ],
    ),
    (
        "Nystrom approximation uses landmark points to create a low-rank approximation of the full attention matrix, reducing quadratic complexity to linear. The random Fourier features approach achieves similar goals through kernel function approximation.",
        [
            ("Nystrom approximation", "algorithm"),
            ("landmark points", "data_structure"),
            ("attention matrix", "data_structure"),
            ("quadratic complexity", "constraint"),
            ("random Fourier features", "algorithm"),
            ("kernel function approximation", "algorithm"),
        ],
        [
            (0, 1, "USES"),
            (0, 2, "TARGETS"),
            (0, 3, "REDUCES"),
            (4, 0, "COMPETES_WITH"),
            (4, 5, "USES"),
        ],
    ),
    (
        "The SSBO running offset fix ensures each flush appends at increasing offset to avoid buffer overwrite in the batched elementwise kernel. The capacity is set to 2048 operations with graceful fallback when full.",
        [
            ("SSBO running offset", "optimization_technique"),
            ("batched elementwise kernel", "kernel_operation"),
            ("2048 operations", "constraint"),
        ],
        [
            (0, 1, "IMPROVES"),
            (2, 1, "LIMITS"),
        ],
    ),
    (
        "HIPBLASLT is slower for token generation on this model at 46.88 versus 47.79 tokens per second. HIP graphs also regress to 45.50 tokens per second due to graph capture overhead exceeding dispatch savings.",
        [
            ("HIPBLASLT", "software_framework"),
            ("46.88", "performance_metric"),
            ("47.79 tokens per second", "performance_metric"),
            ("HIP graphs", "optimization_technique"),
            ("45.50 tokens per second", "performance_metric"),
            ("graph capture overhead", "constraint"),
        ],
        [
            (0, 1, "MEASURES"),
            (3, 4, "MEASURES"),
            (5, 3, "LIMITS"),
        ],
    ),
    (
        "The polyhedral model used by ConstrINT represents hardware constraints as a system of linear inequalities over integer variables. This includes register file sizes, shared memory capacity, and warp occupancy limits.",
        [
            ("polyhedral model", "algorithm"),
            ("ConstrINT", "software_framework"),
            ("register file sizes", "constraint"),
            ("shared memory capacity", "constraint"),
            ("warp occupancy", "performance_metric"),
        ],
        [
            (0, 1, "IS_PART_OF"),
            (1, 2, "USES"),
            (1, 3, "USES"),
            (1, 4, "USES"),
        ],
    ),
    (
        "CUDA thread block configuration directly impacts occupancy and performance on NVIDIA GPUs. A100 tensor cores operate at peak throughput with specific tile sizes aligned to warp dimensions.",
        [
            ("thread block configuration", "optimization_technique"),
            ("occupancy", "performance_metric"),
            ("NVIDIA GPUs", "hardware"),
            ("A100 tensor cores", "hardware"),
            ("tile sizes", "constraint"),
            ("warp dimensions", "constraint"),
        ],
        [
            (0, 1, "IMPROVES"),
            (0, 2, "TARGETS"),
            (3, 4, "REQUIRES"),
            (4, 5, "REQUIRES"),
        ],
    ),
    (
        "Token merging reduces the number of tokens processed through transformer layers by progressively merging similar tokens. This technique trades a small accuracy loss for significant speedup in vision transformers and LLM inference.",
        [
            ("Token merging", "optimization_technique"),
            ("transformer layers", "model_architecture"),
            ("accuracy loss", "performance_metric"),
            ("vision transformers", "model_architecture"),
            ("LLM inference", "algorithm"),
        ],
        [
            (0, 1, "IMPROVES"),
            (0, 2, "MEASURES"),
            (0, 3, "TARGETS"),
            (0, 4, "TARGETS"),
        ],
    ),
    (
        "The alternating kernel strategy for FlashRNN uses two kernel calls per time step. The first kernel performs the recurrent matrix multiplication, and the second handles point-wise nonlinearities, enabling maximum hidden size support.",
        [
            ("alternating kernel strategy", "optimization_technique"),
            ("FlashRNN", "software_framework"),
            ("recurrent matrix multiplication", "kernel_operation"),
            ("point-wise nonlinearities", "kernel_operation"),
            ("hidden size", "constraint"),
        ],
        [
            (0, 1, "IS_PART_OF"),
            (0, 2, "USES"),
            (0, 3, "USES"),
            (0, 4, "IMPROVES"),
        ],
    ),
    (
        "Submodular function optimization provides theoretical guarantees for selecting representative subsets of tokens or attention heads for pruning. The greedy algorithm achieves a 1-1/e approximation ratio.",
        [
            ("Submodular function optimization", "algorithm"),
            ("representative subsets", "data_structure"),
            ("attention heads", "data_structure"),
            ("pruning", "optimization_technique"),
            ("greedy algorithm", "algorithm"),
            ("1-1/e approximation ratio", "performance_metric"),
        ],
        [
            (0, 1, "USES"),
            (3, 2, "TARGETS"),
            (4, 0, "IMPLEMENTS"),
            (4, 5, "MEASURES"),
        ],
    ),
    (
        "The Information Bottleneck principle compresses intermediate representations while preserving task-relevant information. Applied to neural network pruning, it identifies which neurons or layers can be removed with minimal performance impact.",
        [
            ("Information Bottleneck", "algorithm"),
            ("intermediate representations", "data_structure"),
            ("neural network pruning", "optimization_technique"),
            ("performance impact", "performance_metric"),
        ],
        [
            (0, 1, "TARGETS"),
            (0, 2, "ENABLES"),
            (2, 3, "MEASURES"),
        ],
    ),
    (
        "CKA (Centered Kernel Alignment) measures the similarity between neural network layer representations. It is used to analyze how different optimization techniques affect internal model representations across training checkpoints.",
        [
            ("CKA", "algorithm"),
            ("Centered Kernel Alignment", "algorithm"),
            ("layer representations", "data_structure"),
            ("optimization techniques", "optimization_technique"),
            ("training checkpoints", "data_structure"),
        ],
        [
            (0, 2, "MEASURES"),
            (0, 3, "VALIDATES"),
            (0, 4, "USES"),
        ],
    ),
    (
        "RPCholesky decomposition provides a unified framework for efficient low-rank approximation of kernel matrices. This connects Nystrom-style approximation with random pivoted Cholesky factorization for scalable attention mechanisms.",
        [
            ("RPCholesky", "algorithm"),
            ("low-rank approximation", "optimization_technique"),
            ("kernel matrices", "data_structure"),
            ("Nystrom", "algorithm"),
            ("Cholesky factorization", "algorithm"),
            ("attention mechanisms", "algorithm"),
        ],
        [
            (0, 1, "IMPLEMENTS"),
            (0, 2, "TARGETS"),
            (3, 0, "BUILDS_ON"),
            (4, 0, "BUILDS_ON"),
            (0, 5, "IMPROVES"),
        ],
    ),
]

# ── Build NER and RE datasets ──────────────────────────────────────────

def build_datasets():
    ner_examples = []
    re_all = []

    for text, entity_defs, relation_defs in EXAMPLES:
        ann = annotate(text, entity_defs, relation_defs)
        entities = ann["entities"]
        relations = ann["relations"]

        # NER example
        ner_examples.append({
            "text": text,
            "entities": entities,
        })

        # RE positive examples
        positive_pairs = set()
        for rel in relations:
            h_idx = rel["head_idx"]
            t_idx = rel["tail_idx"]
            if h_idx < len(entities) and t_idx < len(entities):
                re_all.append({
                    "text": text,
                    "head": entities[h_idx],
                    "tail": entities[t_idx],
                    "relation": rel["relation"],
                })
                positive_pairs.add((h_idx, t_idx))

        # RE negative examples (sample from non-related pairs)
        for i, j in combinations(range(len(entities)), 2):
            if (i, j) not in positive_pairs and (j, i) not in positive_pairs:
                re_all.append({
                    "text": text,
                    "head": entities[i],
                    "tail": entities[j],
                    "relation": "no_relation",
                })

    return ner_examples, re_all


def main():
    ner_examples, re_examples = build_datasets()

    print(f"Generated {len(ner_examples)} NER examples")
    print(f"Generated {len(re_examples)} RE examples")

    # Count entity types
    entity_counts = {}
    total_entities = 0
    for ex in ner_examples:
        for ent in ex["entities"]:
            entity_counts[ent["label"]] = entity_counts.get(ent["label"], 0) + 1
            total_entities += 1
    print(f"\nTotal entities: {total_entities}")
    print("Entity type distribution:")
    for et in sorted(entity_counts, key=entity_counts.get, reverse=True):
        print(f"  {et}: {entity_counts[et]}")

    # Count relation types
    rel_counts = {}
    for ex in re_examples:
        rel_counts[ex["relation"]] = rel_counts.get(ex["relation"], 0) + 1
    print(f"\nRelation type distribution:")
    for rt in sorted(rel_counts, key=rel_counts.get, reverse=True):
        print(f"  {rt}: {rel_counts[rt]}")

    # Split train/val
    random.shuffle(ner_examples)
    val_size = max(1, int(len(ner_examples) * 0.2))
    ner_train = ner_examples[val_size:]
    ner_val = ner_examples[:val_size]

    # RE: group by text, keep groups together
    re_by_text = {}
    for ex in re_examples:
        key = ex["text"][:80]
        re_by_text.setdefault(key, []).append(ex)
    keys = list(re_by_text.keys())
    random.shuffle(keys)
    val_keys = set(keys[:max(1, int(len(keys) * 0.2))])
    re_train = [ex for k in keys if k not in val_keys for ex in re_by_text[k]]
    re_val = [ex for k in keys if k in val_keys for ex in re_by_text[k]]

    # Save
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    for name, data in [
        ("ner_train.json", ner_train),
        ("ner_val.json", ner_val),
        ("re_train.json", re_train),
        ("re_val.json", re_val),
    ]:
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {path} ({len(data)} examples)")


if __name__ == "__main__":
    main()
