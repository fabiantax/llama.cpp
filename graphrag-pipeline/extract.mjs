/**
 * GraphRAG-style entity/relation extraction pipeline for GPU optimization research.
 *
 * Borrows from:
 * - GraphRAG: multi-round gleaning, structured extraction prompt, 600-token chunks
 * - LightRAG: entity deduplication/normalization, incremental graph merge
 *
 * Sources: arxiv papers via MCP, local code/logs, profiling data
 * Output: FalkorDB graph (gpu_optimization)
 */

import Anthropic from '@anthropic-ai/sdk';
import { FalkorDB } from 'falkordb';
import { readFileSync, writeFileSync, existsSync } from 'fs';

// ── Config ──────────────────────────────────────────────────────────────
const CHUNK_SIZE = 600;       // tokens (~2400 chars). GraphRAG: 600 > 2400 for extraction
const CHUNK_OVERLAP = 100;    // token overlap between chunks
const MAX_GLEANING_ROUNDS = 2; // GraphRAG uses up to 4, we use 2 for cost
const ANTHROPIC_MODEL = 'claude-haiku-4-5-20251001'; // fast + cheap for extraction

// ── Domain-specific entity types ────────────────────────────────────────
const ENTITY_TYPES = [
  'hardware',           // GPU, CPU, memory subsystem, cache
  'hardware_feature',   // specific feature (LDS, infinity cache, wave mode)
  'model',              // ML model (Qwen, Mamba, RWKV)
  'architecture',       // model architecture pattern (SSM, MoE, Transformer)
  'kernel_technique',   // GPU kernel optimization technique
  'algorithm',          // mathematical algorithm or method
  'software',           // framework, library, compiler
  'metric',             // performance measurement
  'constraint',         // bottleneck, limitation
  'data_structure',     // tensor layout, memory layout
  'operation',          // GGML op, CUDA op, compute primitive
  'paper',              // research paper
];

const RELATION_TYPES = [
  'IMPLEMENTS', 'USES', 'OPTIMIZES', 'TARGETS', 'IMPROVES',
  'REDUCES', 'ELIMINATES', 'MEASURES', 'LIMITS', 'ENABLES',
  'EXTENDS', 'BUILDS_ON', 'VALIDATES', 'COMPETES_WITH',
  'IS_PART_OF', 'IS_FEATURE_OF', 'REQUIRES', 'BLOCKED_BY',
  'COULD_IMPROVE', 'INTRODUCES', 'PORTS_TO',
];

// ── Extraction prompt (GraphRAG-style with few-shot examples) ───────────
const EXTRACTION_PROMPT = `You are an expert entity and relationship extractor for GPU kernel optimization research.

ENTITY TYPES: ${ENTITY_TYPES.join(', ')}
RELATION TYPES: ${RELATION_TYPES.join(', ')}

Extract ALL entities and relationships from the text below. Be thorough — extract every technical concept, hardware detail, algorithm name, performance number, and optimization technique mentioned.

For each entity provide:
- name: normalized CamelCase name (e.g., "SharedMemoryTiling", "RDNA35", "DeltaNet")
- type: one of the entity types above
- description: 1-2 sentence description from the text
- properties: key-value pairs for any numeric values or specific attributes

For each relationship provide:
- from: entity name
- to: entity name
- type: one of the relation types above
- description: brief description of the relationship

## Few-shot examples:

Text: "FlashRNN uses ConstrINT polyhedral constraints to find optimal tile sizes on NVIDIA H100, achieving 50x speedup over vanilla PyTorch."
Entities:
- {name: "FlashRNN", type: "software", description: "Hardware-optimized RNN kernel framework"}
- {name: "ConstrINT", type: "algorithm", description: "Polyhedral constraint solver for optimal tile size selection"}
- {name: "H100", type: "hardware", description: "NVIDIA H100 GPU"}
- {name: "TileOptimization", type: "kernel_technique", description: "Finding optimal tile dimensions for GPU kernel performance"}
Relations:
- {from: "FlashRNN", to: "ConstrINT", type: "USES", description: "Uses ConstrINT for tile optimization"}
- {from: "FlashRNN", to: "H100", type: "TARGETS", description: "Optimized for H100 GPU"}
- {from: "ConstrINT", to: "TileOptimization", type: "IMPLEMENTS", description: "Implements tile size optimization via polyhedral constraints"}

## Now extract from this text:

{TEXT}

Respond ONLY with valid JSON:
{
  "entities": [...],
  "relations": [...]
}`;

// ── Gleaning prompt (GraphRAG multi-round) ──────────────────────────────
const GLEANING_PROMPT = `Review the text again carefully. Many entities were missed in the previous extraction.

Previously extracted entities: {PREV_ENTITIES}

Look specifically for:
1. Hardware features (cache sizes, bandwidth, register counts, wave sizes)
2. Specific numeric metrics (tok/s, microseconds, speedup factors)
3. Memory access patterns (coalesced, strided, tiled, bank conflicts)
4. Optimization techniques not yet captured
5. Relationships between already-extracted entities that were missed

Text: {TEXT}

Return ONLY the NEW entities and relations not in the previous list.
Respond with valid JSON: {"entities": [...], "relations": [...]}`;

// ── Chunker (GraphRAG: 600 tokens ≈ 2400 chars) ────────────────────────
function chunkText(text, chunkChars = CHUNK_SIZE * 4, overlapChars = CHUNK_OVERLAP * 4) {
  const chunks = [];
  let start = 0;
  // Hard limit: at most one chunk per character (prevents infinite loops from logic bugs)
  const maxIterations = text.length + 1;

  for (let iter = 0; iter < maxIterations && start < text.length; iter++) {
    let end = Math.min(start + chunkChars, text.length);
    // Try to break at paragraph or sentence boundary (search only within this chunk)
    if (end < text.length) {
      const slice = text.slice(start, end);
      const lastPara = slice.lastIndexOf('\n\n');
      const lastSentence = slice.lastIndexOf('. ');
      if (lastPara > chunkChars * 0.5) end = start + lastPara + 2;
      else if (lastSentence > chunkChars * 0.5) end = start + lastSentence + 2;
    }
    chunks.push(text.slice(start, end));
    if (end >= text.length) break;
    // Guarantee forward progress: new start must be past old start
    const newStart = end > overlapChars ? end - overlapChars : end;
    start = Math.max(newStart, start + 1);
  }
  return chunks;
}

// ── LLM extraction ─────────────────────────────────────────────────────
async function extractFromChunk(client, text, chunkIdx, totalChunks) {
  const prompt = EXTRACTION_PROMPT.replace('{TEXT}', text);

  let allEntities = [];
  let allRelations = [];

  // Round 1: Initial extraction
  console.log(`  [${chunkIdx + 1}/${totalChunks}] Round 1: extracting...`);
  const r1 = await client.messages.create({
    model: ANTHROPIC_MODEL,
    max_tokens: 4096,
    messages: [{ role: 'user', content: prompt }],
  });

  try {
    const parsed = JSON.parse(r1.content[0].text);
    allEntities.push(...(parsed.entities || []));
    allRelations.push(...(parsed.relations || []));
  } catch (e) {
    // Try to extract JSON from markdown code block
    const match = r1.content[0].text.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (match) {
      const parsed = JSON.parse(match[1]);
      allEntities.push(...(parsed.entities || []));
      allRelations.push(...(parsed.relations || []));
    } else {
      console.log(`  [${chunkIdx + 1}] WARNING: failed to parse round 1`);
    }
  }

  // Round 2+: Gleaning (GraphRAG-style)
  for (let round = 1; round <= MAX_GLEANING_ROUNDS; round++) {
    if (allEntities.length === 0) break;

    const prevNames = allEntities.map(e => e.name).join(', ');
    const gleanPrompt = GLEANING_PROMPT
      .replace('{PREV_ENTITIES}', prevNames)
      .replace('{TEXT}', text);

    console.log(`  [${chunkIdx + 1}/${totalChunks}] Round ${round + 1}: gleaning...`);
    const rN = await client.messages.create({
      model: ANTHROPIC_MODEL,
      max_tokens: 2048,
      messages: [{ role: 'user', content: gleanPrompt }],
    });

    try {
      const raw = rN.content[0].text;
      const match = raw.match(/```(?:json)?\s*([\s\S]*?)```/) || [null, raw];
      const parsed = JSON.parse(match[1]);
      const newEntities = parsed.entities || [];
      const newRelations = parsed.relations || [];

      if (newEntities.length === 0 && newRelations.length === 0) {
        console.log(`  [${chunkIdx + 1}] Gleaning round ${round + 1}: nothing new, stopping`);
        break;
      }

      allEntities.push(...newEntities);
      allRelations.push(...newRelations);
      console.log(`  [${chunkIdx + 1}] Gleaning found +${newEntities.length} entities, +${newRelations.length} relations`);
    } catch {
      console.log(`  [${chunkIdx + 1}] WARNING: failed to parse gleaning round ${round + 1}`);
    }
  }

  return { entities: allEntities, relations: allRelations };
}

// ── Entity deduplication (LightRAG-style) ───────────────────────────────
function normalizeEntityName(name) {
  return name
    .replace(/[\s\-_]+/g, '')    // Remove spaces, hyphens, underscores
    .replace(/[^\w]/g, '')       // Remove non-word chars
    .toLowerCase();
}

function deduplicateEntities(entities) {
  const seen = new Map(); // normalized name -> best entity
  const nameMap = new Map(); // normalized -> canonical name

  for (const entity of entities) {
    const norm = normalizeEntityName(entity.name);

    if (seen.has(norm)) {
      // Merge: keep the one with longer description
      const existing = seen.get(norm);
      if ((entity.description || '').length > (existing.description || '').length) {
        seen.set(norm, { ...entity, name: nameMap.get(norm) });
      }
      // Merge properties
      if (entity.properties) {
        existing.properties = { ...existing.properties, ...entity.properties };
      }
    } else {
      seen.set(norm, entity);
      nameMap.set(norm, entity.name);
    }
  }

  return { entities: Array.from(seen.values()), nameMap };
}

function deduplicateRelations(relations, nameMap) {
  const normName = (n) => {
    const norm = normalizeEntityName(n);
    return nameMap.get(norm) || n;
  };

  const seen = new Set();
  const deduped = [];

  for (const rel of relations) {
    const from = normName(rel.from);
    const to = normName(rel.to);
    const key = `${normalizeEntityName(from)}|${rel.type}|${normalizeEntityName(to)}`;

    if (!seen.has(key)) {
      seen.add(key);
      deduped.push({ ...rel, from, to });
    }
  }

  return deduped;
}

// ── FalkorDB merge (LightRAG-style incremental) ────────────────────────
async function mergeIntoGraph(db, entities, relations) {
  const g = db.selectGraph('gpu_optimization');

  let nodesCreated = 0;
  let nodesUpdated = 0;
  let relsCreated = 0;

  for (const entity of entities) {
    const safeName = entity.name.replace(/'/g, "\\'");
    const safeDesc = (entity.description || '').replace(/'/g, "\\'");
    const safeType = entity.type || 'unknown';

    // Check if exists
    const existing = await g.query(
      `MATCH (n {name: '${safeName}'}) RETURN n.name`
    );

    if (existing.data.length > 0) {
      // Update description if new one is longer
      if (safeDesc.length > 0) {
        await g.query(
          `MATCH (n {name: '${safeName}'}) SET n.description = '${safeDesc}'`
        );
        nodesUpdated++;
      }
    } else {
      // Create new node
      const props = [`name: '${safeName}'`, `type: '${safeType}'`];
      if (safeDesc) props.push(`description: '${safeDesc}'`);
      if (entity.properties) {
        for (const [k, v] of Object.entries(entity.properties)) {
          if (typeof v === 'number') props.push(`${k}: ${v}`);
          else if (typeof v === 'string') props.push(`${k}: '${String(v).replace(/'/g, "\\'")}'`);
        }
      }
      try {
        await g.query(`CREATE (:${safeType} {${props.join(', ')}})`);
        nodesCreated++;
      } catch (e) {
        console.log(`  ! Failed to create node ${safeName}: ${e.message}`);
      }
    }
  }

  for (const rel of relations) {
    const safeFrom = rel.from.replace(/'/g, "\\'");
    const safeTo = rel.to.replace(/'/g, "\\'");

    try {
      // Check if both nodes exist
      const check = await g.query(
        `MATCH (a {name: '${safeFrom}'}), (b {name: '${safeTo}'}) RETURN a.name, b.name`
      );
      if (check.data.length === 0) continue;

      // Check if relation already exists
      const existingRel = await g.query(
        `MATCH (a {name: '${safeFrom}'})-[r:${rel.type}]->(b {name: '${safeTo}'}) RETURN type(r)`
      );
      if (existingRel.data.length > 0) continue;

      await g.query(
        `MATCH (a {name: '${safeFrom}'}), (b {name: '${safeTo}'}) CREATE (a)-[:${rel.type}]->(b)`
      );
      relsCreated++;
    } catch (e) {
      // Relation type or node type issue — skip silently
    }
  }

  return { nodesCreated, nodesUpdated, relsCreated };
}

// ── Main pipeline ───────────────────────────────────────────────────────
async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log(`Usage: node extract.mjs <source-file-or-dir> [--dry-run]`);
    console.log(`       node extract.mjs paper.txt`);
    console.log(`       node extract.mjs graphrag-pipeline/sources/`);
    console.log(`\nPut paper text or code in graphrag-pipeline/sources/ and run.`);
    process.exit(1);
  }

  const dryRun = args.includes('--dry-run');
  const sourcePath = args[0];

  // Read source
  let sourceText;
  if (existsSync(sourcePath)) {
    sourceText = readFileSync(sourcePath, 'utf-8');
  } else {
    console.error(`File not found: ${sourcePath}`);
    process.exit(1);
  }

  console.log(`Source: ${sourcePath} (${sourceText.length} chars)`);

  // Chunk
  const chunks = chunkText(sourceText);
  console.log(`Chunked into ${chunks.length} pieces (~${CHUNK_SIZE} tokens each)\n`);

  // Extract
  const client = new Anthropic();
  let allEntities = [];
  let allRelations = [];

  for (let i = 0; i < chunks.length; i++) {
    const result = await extractFromChunk(client, chunks[i], i, chunks.length);
    allEntities.push(...result.entities);
    allRelations.push(...result.relations);
    console.log(`  → Cumulative: ${allEntities.length} entities, ${allRelations.length} relations\n`);
  }

  // Deduplicate (LightRAG-style)
  console.log(`\nDeduplicating ${allEntities.length} raw entities...`);
  const { entities: dedupedEntities, nameMap } = deduplicateEntities(allEntities);
  const dedupedRelations = deduplicateRelations(allRelations, nameMap);
  console.log(`After dedup: ${dedupedEntities.length} entities, ${dedupedRelations.length} relations`);

  // Save raw extraction (for debugging)
  const outputPath = sourcePath.replace(/\.\w+$/, '') + '_extracted.json';
  writeFileSync(outputPath, JSON.stringify({ entities: dedupedEntities, relations: dedupedRelations }, null, 2));
  console.log(`Saved raw extraction to ${outputPath}`);

  if (dryRun) {
    console.log('\n--dry-run: skipping FalkorDB merge');
    console.log('\nEntities:');
    for (const e of dedupedEntities) console.log(`  [${e.type}] ${e.name}: ${e.description}`);
    console.log('\nRelations:');
    for (const r of dedupedRelations) console.log(`  ${r.from} --${r.type}--> ${r.to}`);
    return;
  }

  // Merge into FalkorDB (LightRAG-style incremental)
  console.log('\nMerging into FalkorDB (gpu_optimization)...');
  const db = await FalkorDB.connect({ socket: { host: 'localhost', port: 6379 } });
  const stats = await mergeIntoGraph(db, dedupedEntities, dedupedRelations);
  console.log(`  Nodes created: ${stats.nodesCreated}, updated: ${stats.nodesUpdated}`);
  console.log(`  Relations created: ${stats.relsCreated}`);
  await db.close();

  console.log('\nDone! Open http://localhost:3000 to explore the graph.');
}

main().catch(console.error);
