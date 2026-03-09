Research a topic using web search, arXiv papers, and Hugging Face. Synthesize findings into actionable results.

## Instructions

1. Load research tools first:
   - Use ToolSearch to load: WebSearch, WebFetch
   - Use ToolSearch to load arxiv MCP tools: mcp__arxiv-server__search_papers, mcp__arxiv-server__read_paper, mcp__arxiv-server__list_papers
   - Use ToolSearch to load HuggingFace MCP tools: mcp__claude_ai_Hugging_Face__paper_search, mcp__claude_ai_Hugging_Face__hub_repo_search

2. Search phase — run these in parallel:
   - WebSearch for: $ARGUMENTS
   - mcp__arxiv-server__search_papers for relevant papers
   - mcp__claude_ai_Hugging_Face__paper_search if ML models are relevant

3. Deep-dive phase:
   - For each promising result, use WebFetch to get details (GitHub READMEs, docs)
   - For key papers, use mcp__arxiv-server__read_paper to read full content
   - Use mcp__claude_ai_Hugging_Face__hub_repo_search for relevant models/datasets

4. Compile findings into a structured comparison table with:
   - Project name + URL
   - Language/SDK (TypeScript, Rust, Python, etc.)
   - Key features relevant to the query
   - Maturity (stars, last update, version)
   - How it could apply to our GPU optimization work

5. If FalkorDB is running, suggest new entities/relations to add from findings

6. Eval: verify at least 3 sources were consulted and findings are cross-referenced
