Run the Rust NER extraction pipeline using the anno crate (GLiNER zero-shot). Pass a source file path and options.

## Instructions

1. Build and run the pipeline in `graphrag-pipeline/`:
   ```
   cd C:/Users/fabia/Projects/llama.cpp/llama.cpp/graphrag-pipeline
   cargo run -- $ARGUMENTS
   ```

2. If no arguments given, show usage help:
   ```
   cargo run -- --help
   ```

## CLI Options

- `--source <file>` — Input text file (paper, profiling log, code comments)
- `--dry-run` — Print extracted entities/relations without writing to FalkorDB
- `--ner-only` — Skip LLM relation extraction, NER pass only (fast, free)
- `--labels <csv>` — Custom entity types (default: hardware,gpu_feature,optimization_technique,algorithm,software_framework,performance_metric,memory_pattern,kernel_operation,model_architecture,constraint,data_structure,research_paper)

## How It Works

- **anno crate**: Uses `GLiNEROnnx` with model `onnx-community/gliner_small-v2.1`
- **ZeroShotNER trait**: `extract_with_types(text, labels, threshold=0.5)`
- Chunks text at ~600 tokens (2400 chars) with 400-char overlap (GraphRAG optimal)
- Outputs `<source>_extracted.json` with entities and relations

## Typical Workflows

- **Quick NER scan**: `--source paper.txt --ner-only --dry-run`
- **Full pipeline**: `--source paper.txt` (needs ANTHROPIC_API_KEY + FalkorDB running)
- **Custom domain**: `--source log.txt --labels "kernel,bandwidth,latency,occupancy" --ner-only`

## Build

```
cd C:/Users/fabia/Projects/llama.cpp/llama.cpp/graphrag-pipeline
cargo build
```
