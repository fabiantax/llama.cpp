Search arXiv for papers and read their contents. Pass a search query or paper ID.

## Instructions

1. Load arXiv MCP tools:
   - Use ToolSearch to load: `select:mcp__arxiv-server__search_papers,mcp__arxiv-server__read_paper,mcp__arxiv-server__list_papers,mcp__arxiv-server__download_paper`

2. Execute the user's request: $ARGUMENTS

3. Search phase:
   - Use `mcp__arxiv-server__search_papers` with the query
   - Search tips: prefix `ti:` for title, `au:` for author
   - Category filters: `cs.LG` (ML), `cs.AR` (architecture), `cs.DC` (distributed), `cs.PF` (performance)
   - Example: `ti:flash attention au:dao cat:cs.LG`

4. Read phase:
   - For each relevant paper, use `mcp__arxiv-server__read_paper` with the arXiv ID
   - Summarize: problem, method, key results, relevance to GPU optimization

5. Save findings:
   - If FalkorDB is running (use /falkordb), create research_paper nodes and relations
   - Save paper text to `graphrag-pipeline/sources/` for later NER extraction
   - Report paper IDs, titles, and key takeaways

## Quick Examples

- Search: `mcp__arxiv-server__search_papers` with query `"flash attention v3 hopper"`
- Read: `mcp__arxiv-server__read_paper` with id `"2307.08691"`
- List recent: `mcp__arxiv-server__list_papers` with category `"cs.LG"` and max_results `5`
