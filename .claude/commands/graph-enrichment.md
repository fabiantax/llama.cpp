Run the full knowledge graph enrichment pipeline: arXiv paper -> chunk -> NER -> LLM relations -> dedup -> FalkorDB. Pass a paper ID, URL, or topic.

## Instructions

1. Load tools:
   - Use ToolSearch to load: `select:mcp__arxiv-server__search_papers,mcp__arxiv-server__read_paper,mcp__arxiv-server__download_paper`
   - Use ToolSearch to load: `+falkordb` (for graph merge verification)

2. **Acquire source** from $ARGUMENTS:
   - If arXiv ID (e.g. `2307.08691`): use `mcp__arxiv-server__read_paper` to get full text
   - If search topic: use `mcp__arxiv-server__search_papers`, pick best result, then read it
   - Save text to `C:/Users/fabia/Projects/llama.cpp/llama.cpp/graphrag-pipeline/sources/<id>.txt`

3. **Run extraction pipeline**:
   ```bash
   cd C:/Users/fabia/Projects/llama.cpp/llama.cpp/graphrag-pipeline
   cargo run -- --source sources/<id>.txt
   ```
   - Without ANTHROPIC_API_KEY: add `--ner-only` (NER pass only, no LLM gleaning)
   - For preview: add `--dry-run` (print results, skip FalkorDB merge)

4. **Verify in FalkorDB**:
   - Query new nodes: `MATCH (n) WHERE n.name CONTAINS '<keyword>' RETURN n`
   - Check relations: `MATCH (n)-[r]->(m) RETURN n.name, type(r), m.name ORDER BY n.name LIMIT 20`

5. **Report**: Summarize entities created, relations found, and any dedup merges.

## Pipeline Stages (GraphRAG + LightRAG hybrid)

| Stage | Technique | Detail |
|-------|-----------|--------|
| Chunk | GraphRAG | 600-token chunks, 100-token overlap |
| NER | anno/GLiNER | Zero-shot with 12 GPU-domain entity types |
| Relations | GraphRAG gleaning | Claude Haiku, multi-round extraction per chunk |
| Dedup | LightRAG | Normalize names, merge properties, deduplicate rels |
| Merge | Incremental | MATCH-or-CREATE into FalkorDB `gpu_optimization` graph |
