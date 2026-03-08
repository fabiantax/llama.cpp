/*
 * Fused Delta-Net SSM recurrence kernel for CUDA/HIP.
 *
 * Port of the Vulkan ssm_recurrence.comp shader.
 * Replaces ~11 individual dispatches per SSM layer with 1 fused kernel.
 *
 * Dispatch: one block per (head, seq) pair, S threads per block.
 * Thread j owns row j of the state matrix.
 *
 * Math per head:
 *   s_dec[j][i] = s[j][i] * exp(gate)
 *   sk[j] = dot(s_dec[j], k)
 *   d[j]  = (v[j] - sk[j]) * beta
 *   s_new[j][i] = s_dec[j][i] + k[i] * d[j]
 *   o[j]  = dot(s_new[j], q)
 *
 * State layout: [S, S, H, n_seqs]
 * Output layout: dst[0..S*H*n_seqs-1] = output, dst[s_off..] = new state
 */

#include "delta-net-recurrence.cuh"

// Tile size for shared memory. On HIP/RDNA 3.5 we have 64 KB LDS,
// on CUDA we typically have 48-96 KB. TILE_K=128 uses 128*128*4=64KB.
// Use 128 on HIP (fits in 64 KB LDS), 64 on CUDA for safety.
#if defined(GGML_USE_HIP)
#define TILE_K 128
#else
#define TILE_K 64
#endif

template <int BLOCK_SIZE>
__global__ void delta_net_recurrence_f32(
    const float * __restrict__ state_in,  // [S, S, H, n_seqs]
    const float * __restrict__ q,         // [S, H, 1, n_seqs]
    const float * __restrict__ k,         // [S, H, 1, n_seqs]
    const float * __restrict__ v,         // [S, H, 1, n_seqs]
    const float * __restrict__ gate,      // [1, H, 1, n_seqs]
    const float * __restrict__ beta_in,   // [1, H, 1, n_seqs]
    float * __restrict__ dst,
    const int S,
    const int H,
    const int n_seqs,
    const int s_off)
{
    // Shared memory tile: TILE_K columns x BLOCK_SIZE rows, transposed layout
    // s_tile[col * BLOCK_SIZE + row] for coalesced access
    __shared__ float s_tile[TILE_K * BLOCK_SIZE];

    const int j  = threadIdx.x;  // Thread j owns row j of state
    const int wg = blockIdx.x;   // Block index = head*n_seqs + seq

    if (j >= S) return;

    // Map block to (head, seq)
    const int head = wg % H;
    const int seq  = wg / H;

    // Base offsets
    const int si_base  = seq * H * S * S + head * S * S;
    const int qkv_base = seq * H * S + head * S;
    const int gh_idx   = seq * H + head;
    const int out_base = seq * H * S + head * S;
    const int so_base  = s_off + si_base;

    const float exp_g    = expf(gate[gh_idx]);
    const float beta_val = beta_in[gh_idx];

    // ===== Pass 1: Compute sk_j = dot(decayed_state_row_j, k) =====
    float sk_j = 0.0f;

    for (int tile = 0; tile < S; tile += TILE_K) {
        const int tile_cols = min(TILE_K, S - tile);

        // Cooperative coalesced load into shared memory
        // 128 threads, TILE_K=64 columns: each column loaded by 2 threads (half rows each)
        const int my_col  = j % TILE_K;
        const int my_half = j / TILE_K;
        const int row_start = my_half * (S / 2);
        const int row_end   = row_start + (S / 2);

        if (my_col < tile_cols) {
            for (int row = row_start; row < row_end; row++) {
                // Consecutive threads access consecutive columns → coalesced
                s_tile[my_col * S + row] = state_in[si_base + row * S + (tile + my_col)] * exp_g;
            }
        }

        __syncthreads();

        // Partial dot product: s_tile[t * S + j] is state[j][tile+t] (transposed)
        for (int t = 0; t < tile_cols; t++) {
            sk_j += s_tile[t * S + j] * k[qkv_base + tile + t];
        }

        __syncthreads();
    }

    // ===== Innovation d[j] = (v[j] - sk[j]) * beta =====
    const float d_j = (v[qkv_base + j] - sk_j) * beta_val;

    // ===== Pass 2: State update + output dot product =====
    float o_j = 0.0f;

    for (int tile = 0; tile < S; tile += TILE_K) {
        const int tile_cols = min(TILE_K, S - tile);

        // Cooperative coalesced load (same pattern)
        const int my_col  = j % TILE_K;
        const int my_half = j / TILE_K;
        const int row_start = my_half * (S / 2);
        const int row_end   = row_start + (S / 2);

        if (my_col < tile_cols) {
            for (int row = row_start; row < row_end; row++) {
                s_tile[my_col * S + row] = state_in[si_base + row * S + (tile + my_col)] * exp_g;
            }
        }

        __syncthreads();

        // Update state + accumulate output
        // Thread j reads/writes position [t * S + j] — no overlap with other threads
        for (int t = 0; t < tile_cols; t++) {
            float s_new = s_tile[t * S + j] + k[qkv_base + tile + t] * d_j;
            s_tile[t * S + j] = s_new;
            o_j += s_new * q[qkv_base + tile + t];
        }

        __syncthreads();

        // Cooperative coalesced writeback
        if (my_col < tile_cols) {
            for (int row = row_start; row < row_end; row++) {
                dst[so_base + row * S + (tile + my_col)] = s_tile[my_col * S + row];
            }
        }

        __syncthreads();
    }

    // Write output
    dst[out_base + j] = o_j;
}

void ggml_cuda_op_delta_net_recurrence(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * state_in = dst->src[0];  // [S, S, H, n_seqs]
    const ggml_tensor * q_t      = dst->src[1];  // [S, H, 1, n_seqs]
    const ggml_tensor * k_t      = dst->src[2];
    const ggml_tensor * v_t      = dst->src[3];
    const ggml_tensor * gate_t   = dst->src[4];  // [1, H, 1, n_seqs]
    const ggml_tensor * beta_t   = dst->src[5];  // [1, H, 1, n_seqs]

    GGML_ASSERT(state_in->type == GGML_TYPE_F32);

    const int S      = q_t->ne[0];
    const int H      = q_t->ne[1];
    const int n_seqs = state_in->ne[3];
    const int s_off  = S * H * n_seqs;  // offset to state output in dst (in floats)

    const float * state_data = (const float *) state_in->data;
    const float * q_data     = (const float *) q_t->data;
    const float * k_data     = (const float *) k_t->data;
    const float * v_data     = (const float *) v_t->data;
    const float * gate_data  = (const float *) gate_t->data;
    const float * beta_data  = (const float *) beta_t->data;
    float * dst_data         = (float *) dst->data;

    const int num_blocks = H * n_seqs;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(S == 128 && "delta_net_recurrence kernel requires S=128");

    delta_net_recurrence_f32<128><<<num_blocks, 128, 0, stream>>>(
        state_data, q_data, k_data, v_data, gate_data, beta_data, dst_data,
        S, H, n_seqs, s_off);
}
