#pragma once

#include "llama.h"

struct llama_context;
struct llama_model;
class  llama_kv_cache;

// Run KV cache compaction for the given context and sequence.
// Returns the number of tokens after compaction, or -1 on error.
int32_t llama_kv_compact_impl(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        llama_compact_params   params);
