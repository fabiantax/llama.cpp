//! GraphRAG-style extraction pipeline for GPU optimization knowledge graph.
//!
//! Pipeline:
//!   1. Read source text (paper, code, profiling log)
//!   2. Chunk into ~600-token pieces (GraphRAG optimal)
//!   3. NER pass via anno/GLiNER (Rust, local, fast, zero-shot custom labels)
//!   4. LLM relation extraction + gleaning via Claude API (GraphRAG multi-round)
//!   5. Entity deduplication (LightRAG-style normalization)
//!   6. Incremental merge into FalkorDB
//!
//! Usage:
//!   cargo run -- --source paper.txt [--dry-run] [--ner-only]

use composable::{composed, Composable};
use gliner::model::GLiNER;
use gliner::model::params::Parameters;
use gliner::model::input::text::TextInput;
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::pipeline::span::SpanMode;
use gliner::model::pipeline::token::TokenPipeline;
use gliner::model::pipeline::relation::RelationPipeline;
use orp::model::Model;
use orp::pipeline::Pipeline;
use orp::params::RuntimeParameters;
use ort::execution_providers::DirectMLExecutionProvider;
use clap::Parser;
use falkordb::{FalkorClientBuilder, FalkorConnectionInfo};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;

/// GraphRAG extraction pipeline
#[derive(Parser)]
#[command(name = "graphrag-pipeline")]
struct Args {
    /// Source file to extract from
    #[arg(short, long)]
    source: String,

    /// Don't write to FalkorDB, just print results
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Skip LLM gleaning (NER only)
    #[arg(long, default_value_t = false)]
    ner_only: bool,

    /// Skip local NER (LLM-only extraction, zero local memory)
    #[arg(long, default_value_t = false)]
    skip_ner: bool,

    /// Custom entity labels (comma-separated)
    #[arg(
        short,
        long,
        default_value = "hardware,gpu_feature,optimization_technique,algorithm,software_framework,performance_metric,memory_pattern,kernel_operation,model_architecture,constraint,data_structure,research_paper"
    )]
    labels: String,

    /// Path to GLiNER ONNX model directory (must contain tokenizer.json + model.onnx)
    #[arg(short, long, default_value = "models/gliner_small-v2.1")]
    model_dir: String,

    /// Number of ONNX inference threads (default: 2, keeps CPU free)
    #[arg(short, long, default_value_t = 2)]
    threads: usize,

    /// ONNX model variant: model, model_int8, model_fp16, model_q4, model_q4f16 (default: model_int8)
    #[arg(long, default_value = "model_int8")]
    onnx_variant: String,

    /// Enable local relation extraction via GLiNER multitask model
    #[arg(long, default_value_t = false)]
    extract_relations: bool,

    /// Path to GLiNER multitask model directory (for NER + relation extraction)
    #[arg(long, default_value = "models/gliner-multitask-large-v0.5")]
    multitask_model_dir: String,

    /// ONNX variant for multitask model (default: model_quantized)
    #[arg(long, default_value = "model_quantized")]
    multitask_variant: String,

    /// Use GPU (DirectML) for ONNX inference
    #[arg(long, default_value_t = false)]
    gpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Entity {
    name: String,
    #[serde(rename = "type")]
    entity_type: String,
    description: String,
    #[serde(default)]
    properties: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Relation {
    from: String,
    to: String,
    #[serde(rename = "type")]
    rel_type: String,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExtractionResult {
    entities: Vec<Entity>,
    relations: Vec<Relation>,
}

// ── Chunker (GraphRAG: 600 tokens ≈ 2400 chars) ────────────────────────

fn chunk_text(text: &str, chunk_chars: usize, overlap_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;
    // Hard limit: at most one chunk per byte (prevents infinite loops from logic bugs)
    let max_iterations = text.len().saturating_add(1);

    for _iter in 0..max_iterations {
        if start >= text.len() {
            break;
        }

        let mut end = (start + chunk_chars).min(text.len());

        // Snap to char boundary (UTF-8 continuation bytes are at most 3)
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }

        // Try to break at paragraph or sentence (search only within this chunk's slice)
        if end < text.len() {
            let slice = &text[start..end];
            if let Some(pos) = slice.rfind("\n\n") {
                if pos > chunk_chars / 2 {
                    end = start + pos + 2;
                }
            } else if let Some(pos) = slice.rfind(". ") {
                if pos > chunk_chars / 2 {
                    end = start + pos + 2;
                }
            }
        }

        chunks.push(text[start..end].to_string());
        if end >= text.len() {
            break;
        }

        // Guarantee forward progress: advance at least one full character
        let min_advance = text[start..].chars().next().map_or(1, |c| c.len_utf8());
        let mut new_start = if end > overlap_chars {
            end - overlap_chars
        } else {
            end
        };
        // Snap to char boundary (go backwards to preserve overlap)
        while new_start > 0 && !text.is_char_boundary(new_start) {
            new_start -= 1;
        }
        start = new_start.max(start + min_advance);
    }

    chunks
}

// ── NER via gline-rs/GLiNER (zero-shot, custom entity types) ────────────

fn build_runtime_params(threads: usize, gpu: bool) -> RuntimeParameters {
    let params = RuntimeParameters::default().with_threads(threads);
    if gpu {
        params.with_execution_providers([DirectMLExecutionProvider::default().build()])
    } else {
        params
    }
}

fn run_ner(chunks: &[String], labels: &[&str], model_dir: &str, threads: usize, onnx_variant: &str, gpu: bool) -> Vec<Entity> {
    let accel = if gpu { ", GPU: DirectML" } else { "" };
    eprintln!("Loading GLiNER model from {model_dir} (variant: {onnx_variant}, threads: {threads}{accel})...");
    std::io::stderr().flush().ok();

    // Try onnx/ subdir first, then root
    let model_file = format!("{onnx_variant}.onnx");
    let model_path = if std::path::Path::new(&format!("{model_dir}/onnx/{model_file}")).exists() {
        format!("{model_dir}/onnx/{model_file}")
    } else {
        format!("{model_dir}/{model_file}")
    };

    let runtime_params = build_runtime_params(threads, gpu);

    let model = match GLiNER::<SpanMode>::new(
        Parameters::default(),
        runtime_params,
        &format!("{model_dir}/tokenizer.json"),
        &model_path,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load GLiNER model: {e}");
            eprintln!("Download a model first: huggingface-cli download onnx-community/gliner_small-v2.1-ONNX --local-dir {model_dir}");
            eprintln!("Falling back to LLM-only extraction");
            return Vec::new();
        }
    };

    println!("GLiNER loaded. Extracting with labels: {:?}\n", labels);

    let mut all_entities = Vec::new();

    for (i, chunk) in chunks.iter().enumerate() {
        print!("[{}/{}] NER... ", i + 1, chunks.len());

        let texts: Vec<&str> = vec![chunk.as_str()];
        match TextInput::from_str(&texts, labels) {
            Ok(input) => match model.inference(input) {
                Ok(output) => {
                    let mut count = 0;
                    for spans in &output.spans {
                        for span in spans {
                            all_entities.push(Entity {
                                name: span.text().to_string(),
                                entity_type: span.class().to_string(),
                                description: format!("GLiNER NER (conf: {:.2})", span.probability()),
                                properties: {
                                    let mut m = HashMap::new();
                                    m.insert("confidence".into(), serde_json::json!(span.probability()));
                                    m.insert("source_chunk".into(), serde_json::json!(i));
                                    m
                                },
                            });
                            count += 1;
                        }
                    }
                    println!("{count} entities");
                }
                Err(e) => println!("inference failed: {e}"),
            },
            Err(e) => println!("input failed: {e}"),
        }
    }

    all_entities
}

// ── Local relation extraction via GLiNER multitask (NER + RE) ────────────

/// Domain-specific relation schema for GPU optimization research
fn build_relation_schema() -> RelationSchema {
    let mut schema = RelationSchema::new();
    // Hardware relations
    schema.push_with_allowed_labels("TARGETS", &["optimization_technique", "software_framework", "algorithm"], &["hardware", "gpu_feature"]);
    schema.push_with_allowed_labels("IS_FEATURE_OF", &["gpu_feature"], &["hardware"]);
    // Software/algorithm relations
    schema.push_with_allowed_labels("IMPLEMENTS", &["software_framework"], &["algorithm", "optimization_technique"]);
    schema.push_with_allowed_labels("USES", &["software_framework", "algorithm", "optimization_technique"], &["algorithm", "data_structure", "kernel_operation", "gpu_feature"]);
    // Performance relations
    schema.push_with_allowed_labels("IMPROVES", &["optimization_technique", "algorithm"], &["performance_metric", "kernel_operation"]);
    schema.push_with_allowed_labels("REDUCES", &["optimization_technique", "algorithm"], &["performance_metric", "constraint"]);
    schema.push_with_allowed_labels("MEASURES", &["performance_metric"], &["hardware", "software_framework", "kernel_operation"]);
    schema.push_with_allowed_labels("LIMITS", &["constraint"], &["performance_metric", "optimization_technique"]);
    // Architecture relations
    schema.push_with_allowed_labels("IS_PART_OF", &["kernel_operation", "gpu_feature"], &["model_architecture", "hardware"]);
    schema.push_with_allowed_labels("BUILDS_ON", &["algorithm", "optimization_technique"], &["algorithm", "research_paper"]);
    schema.push_with_allowed_labels("EXTENDS", &["software_framework", "algorithm"], &["software_framework", "algorithm"]);
    // Unconstrained relations (let the model decide)
    schema.push("ENABLES");
    schema.push("REQUIRES");
    schema.push("INTRODUCES");
    schema.push("VALIDATES");
    schema.push("COMPETES_WITH");
    schema
}

fn run_ner_and_relations(
    chunks: &[String],
    labels: &[&str],
    model_dir: &str,
    threads: usize,
    onnx_variant: &str,
    gpu: bool,
) -> (Vec<Entity>, Vec<Relation>) {
    let accel = if gpu { ", GPU: DirectML" } else { "" };
    eprintln!("Loading GLiNER multitask model from {model_dir} (variant: {onnx_variant}, threads: {threads}{accel})...");
    std::io::stderr().flush().ok();

    let model_file = format!("{onnx_variant}.onnx");
    let model_path = if std::path::Path::new(&format!("{model_dir}/onnx/{model_file}")).exists() {
        format!("{model_dir}/onnx/{model_file}")
    } else {
        format!("{model_dir}/{model_file}")
    };
    let tokenizer_path = format!("{model_dir}/tokenizer.json");

    let runtime_params = build_runtime_params(threads, gpu);
    let params = Parameters::default();
    let schema = build_relation_schema();

    let model = match Model::new(&model_path, runtime_params) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load multitask model: {e}");
            eprintln!("Download: python -c \"from huggingface_hub import hf_hub_download; ...\"");
            return (Vec::new(), Vec::new());
        }
    };

    let (ner_pipeline, re_pipeline) = match (|| -> gliner::util::result::Result<_> {
        let ner = TokenPipeline::new(&tokenizer_path)?.to_composable(&model, &params);
        let re = composed![
            TokenPipeline::new(&tokenizer_path)?.to_composable(&model, &params),
            RelationPipeline::default(&tokenizer_path, &schema)?.to_composable(&model, &params)
        ];
        Ok((ner, re))
    })() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to build pipeline: {e}");
            return (Vec::new(), Vec::new());
        }
    };

    println!("GLiNER multitask loaded. Extracting entities + relations with labels: {:?}\n", labels);

    let mut all_entities = Vec::new();
    let mut all_relations = Vec::new();

    // Batched NER: send all chunks in one inference call
    let batch_size = 1;
    let mut entity_type_maps: Vec<HashMap<String, String>> = vec![HashMap::new(); chunks.len()];

    print!("NER batched ({} chunks, batch_size={batch_size})... ", chunks.len());
    std::io::stdout().flush().ok();
    let ner_start = std::time::Instant::now();

    for batch_start in (0..chunks.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(chunks.len());
        let batch_texts: Vec<&str> = chunks[batch_start..batch_end].iter().map(|s| s.as_str()).collect();

        match TextInput::from_str(&batch_texts, labels) {
            Ok(input) => match ner_pipeline.apply(input) {
                Ok(ner_output) => {
                    for (batch_idx, spans) in ner_output.spans.iter().enumerate() {
                        let chunk_idx = batch_start + batch_idx;
                        for span in spans {
                            entity_type_maps[chunk_idx].insert(
                                span.text().to_lowercase(),
                                span.class().to_string(),
                            );
                            all_entities.push(Entity {
                                name: span.text().to_string(),
                                entity_type: span.class().to_string(),
                                description: format!("GLiNER NER (conf: {:.2})", span.probability()),
                                properties: {
                                    let mut m = HashMap::new();
                                    m.insert("confidence".into(), serde_json::json!(span.probability()));
                                    m.insert("source_chunk".into(), serde_json::json!(chunk_idx));
                                    m
                                },
                            });
                        }
                    }
                }
                Err(e) => eprint!("NER batch failed: {e} "),
            },
            Err(e) => eprint!("NER input failed: {e} "),
        }
    }
    println!("{} entities in {:.1}s", all_entities.len(), ner_start.elapsed().as_secs_f32());

    // Batched RE: only on chunks that found entities
    let re_chunks: Vec<(usize, &str)> = chunks.iter().enumerate()
        .filter(|(i, _)| !entity_type_maps[*i].is_empty())
        .map(|(i, s)| (i, s.as_str()))
        .collect();

    print!("RE batched ({} chunks with entities, batch_size={batch_size})... ", re_chunks.len());
    std::io::stdout().flush().ok();
    let re_start = std::time::Instant::now();

    for batch in re_chunks.chunks(batch_size) {
        let batch_texts: Vec<&str> = batch.iter().map(|(_, s)| *s).collect();

        match TextInput::from_str(&batch_texts, labels) {
            Ok(input) => match re_pipeline.apply(input) {
                Ok(output) => {
                    for (batch_idx, rels) in output.relations.iter().enumerate() {
                        let chunk_idx = batch[batch_idx].0;
                        let type_map = &entity_type_maps[chunk_idx];
                        for rel in rels {
                            let subj_type = type_map.get(&rel.subject().to_lowercase()).cloned().unwrap_or_default();
                            let obj_type = type_map.get(&rel.object().to_lowercase()).cloned().unwrap_or_default();

                            all_entities.push(Entity {
                                name: rel.subject().to_string(),
                                entity_type: subj_type,
                                description: format!("GLiNER RE subject (conf: {:.2})", rel.probability()),
                                properties: {
                                    let mut m = HashMap::new();
                                    m.insert("confidence".into(), serde_json::json!(rel.probability()));
                                    m.insert("source_chunk".into(), serde_json::json!(chunk_idx));
                                    m
                                },
                            });
                            all_entities.push(Entity {
                                name: rel.object().to_string(),
                                entity_type: obj_type,
                                description: format!("GLiNER RE object (conf: {:.2})", rel.probability()),
                                properties: {
                                    let mut m = HashMap::new();
                                    m.insert("confidence".into(), serde_json::json!(rel.probability()));
                                    m.insert("source_chunk".into(), serde_json::json!(chunk_idx));
                                    m
                                },
                            });

                            all_relations.push(Relation {
                                from: rel.subject().to_string(),
                                to: rel.object().to_string(),
                                rel_type: rel.class().to_string(),
                                description: format!("GLiNER RE (conf: {:.2})", rel.probability()),
                            });
                        }
                    }
                }
                Err(e) => eprint!("RE batch failed: {e} "),
            },
            Err(e) => eprint!("RE input failed: {e} "),
        }
    }
    println!("{} relations in {:.1}s", all_relations.len(), re_start.elapsed().as_secs_f32());

    let before = all_relations.len();
    all_relations.retain(|r| {
        !r.from.is_empty()
            && !r.to.is_empty()
            && !r.from.eq_ignore_ascii_case(&r.to)
    });
    let filtered = before - all_relations.len();
    if filtered > 0 {
        eprintln!("Filtered {filtered} invalid relations (empty or self-referential)");
    }

    (all_entities, all_relations)
}

// ── LLM relation extraction + gleaning (GraphRAG multi-round) ───────────

#[derive(Deserialize)]
struct ClaudeResponse {
    content: Vec<ClaudeContent>,
}

#[derive(Deserialize)]
struct ClaudeContent {
    text: Option<String>,
}

async fn llm_extract(
    chunk: &str,
    ner_entities: &[Entity],
    api_key: &str,
    chunk_idx: usize,
    total_chunks: usize,
) -> ExtractionResult {
    let entity_list: String = ner_entities
        .iter()
        .map(|e| format!("{} [{}]", e.name, e.entity_type))
        .collect::<Vec<_>>()
        .join(", ");

    let prompt = format!(
        r#"You are an expert at extracting entities and relationships from GPU kernel optimization research.

NER-extracted entities from this chunk: [{entity_list}]

Source text:
---
{chunk}
---

Tasks:
1. Extract ADDITIONAL entities missed by NER (numbers, parameters, speedups, specific techniques)
2. Extract ALL relationships between entities

Entity types: hardware, gpu_feature, optimization_technique, algorithm, software_framework, performance_metric, memory_pattern, kernel_operation, model_architecture, constraint, data_structure, research_paper

Relation types: IMPLEMENTS, USES, OPTIMIZES, TARGETS, IMPROVES, REDUCES, ELIMINATES, MEASURES, LIMITS, ENABLES, EXTENDS, BUILDS_ON, VALIDATES, COMPETES_WITH, IS_PART_OF, IS_FEATURE_OF, REQUIRES, COULD_IMPROVE, INTRODUCES, PORTS_TO

Respond ONLY with valid JSON (no markdown):
{{"entities": [{{"name": "CamelCase", "type": "entity_type", "description": "brief"}}], "relations": [{{"from": "Name", "to": "Name", "type": "REL_TYPE", "description": "brief"}}]}}"#
    );

    println!(
        "  [{}/{}] LLM extraction...",
        chunk_idx + 1,
        total_chunks
    );

    let client = reqwest::Client::new();
    let resp = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&serde_json::json!({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}]
        }))
        .send()
        .await;

    match resp {
        Ok(r) if r.status().is_success() => {
            if let Ok(body) = r.json::<ClaudeResponse>().await {
                if let Some(content) = body.content.first() {
                    if let Some(text) = &content.text {
                        // Extract JSON from response
                        let json_str = if let Some(start) = text.find('{') {
                            if let Some(end) = text.rfind('}') {
                                &text[start..=end]
                            } else {
                                text.as_str()
                            }
                        } else {
                            text.as_str()
                        };

                        match serde_json::from_str::<ExtractionResult>(json_str) {
                            Ok(result) => {
                                println!(
                                    "  [{}/{}] +{} entities, +{} relations",
                                    chunk_idx + 1,
                                    total_chunks,
                                    result.entities.len(),
                                    result.relations.len()
                                );
                                return result;
                            }
                            Err(e) => eprintln!("  JSON parse error: {e}"),
                        }
                    }
                }
            }
        }
        Ok(r) => eprintln!("  API error: {}", r.status()),
        Err(e) => eprintln!("  Request failed: {e}"),
    }

    ExtractionResult {
        entities: Vec::new(),
        relations: Vec::new(),
    }
}

// ── Entity deduplication (LightRAG-style) ───────────────────────────────

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

fn deduplicate(entities: Vec<Entity>, relations: Vec<Relation>) -> ExtractionResult {
    let mut seen: HashMap<String, Entity> = HashMap::new();
    let mut name_map: HashMap<String, String> = HashMap::new();

    for entity in entities {
        let norm = normalize_name(&entity.name);
        if let Some(existing) = seen.get_mut(&norm) {
            if entity.description.len() > existing.description.len() {
                existing.description = entity.description.clone();
            }
            for (k, v) in &entity.properties {
                existing.properties.entry(k.clone()).or_insert(v.clone());
            }
        } else {
            name_map.insert(norm.clone(), entity.name.clone());
            seen.insert(norm, entity);
        }
    }

    let mut rel_set: HashSet<String> = HashSet::new();
    let deduped_relations: Vec<Relation> = relations
        .into_iter()
        .map(|mut r| {
            let from_norm = normalize_name(&r.from);
            let to_norm = normalize_name(&r.to);
            if let Some(canon) = name_map.get(&from_norm) {
                r.from = canon.clone();
            }
            if let Some(canon) = name_map.get(&to_norm) {
                r.to = canon.clone();
            }
            r
        })
        .filter(|r| {
            let key = format!(
                "{}|{}|{}",
                normalize_name(&r.from),
                r.rel_type,
                normalize_name(&r.to)
            );
            rel_set.insert(key)
        })
        .collect();

    ExtractionResult {
        entities: seen.into_values().collect(),
        relations: deduped_relations,
    }
}

// ── FalkorDB merge (LightRAG-style incremental) ────────────────────────

async fn merge_to_falkordb(
    result: &ExtractionResult,
) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let connection_info: FalkorConnectionInfo = "falkor://127.0.0.1:6379"
        .try_into()
        .expect("Invalid connection info");

    let client = FalkorClientBuilder::new()
        .with_connection_info(connection_info)
        .build()?;

    let mut graph = client.select_graph("gpu_optimization");

    let mut nodes_created = 0usize;
    let mut rels_created = 0usize;

    for entity in &result.entities {
        let safe_name = entity.name.replace('\'', "\\'");
        let safe_desc = entity.description.replace('\'', "\\'");
        let safe_type = entity.entity_type.replace('\'', "\\'");

        // Check if node exists
        let existing = graph
            .query(format!(
                "MATCH (n {{name: '{safe_name}'}}) RETURN n.name"
            ))
            .execute()?;

        if existing.data.is_empty() {
            let q = format!(
                "CREATE (:{safe_type} {{name: '{safe_name}', description: '{safe_desc}', type: '{safe_type}'}})"
            );
            match graph.query(&q).execute() {
                Ok(_) => nodes_created += 1,
                Err(e) => eprintln!("  ! Create {safe_name}: {e}"),
            }
        }
    }

    for rel in &result.relations {
        let sf = rel.from.replace('\'', "\\'");
        let st = rel.to.replace('\'', "\\'");

        let q = format!(
            "MATCH (a {{name: '{sf}'}}), (b {{name: '{st}'}}) \
             CREATE (a)-[:{}]->(b)",
            rel.rel_type
        );
        match graph.query(&q).execute() {
            Ok(r) => {
                // stats is Vec<String> like ["Relationships created: 1", ...]
                let created = r.stats.iter().any(|s| {
                    s.contains("Relationships created") && !s.contains(": 0")
                });
                if created {
                    rels_created += 1;
                }
            }
            Err(_) => {}
        }
    }

    Ok((nodes_created, rels_created))
}

// ── Main ────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let source_text = fs::read_to_string(&args.source)?;
    eprintln!("Source: {} ({} chars)", args.source, source_text.len());

    let chunks = chunk_text(&source_text, 2400, 400);
    eprintln!("Chunked into {} pieces (~600 tokens each)", chunks.len());
    std::io::stderr().flush().ok();

    let labels: Vec<&str> = args.labels.split(',').collect();

    // Step 1: Local extraction (NER-only or NER+RE)
    let (mut all_entities, mut all_relations): (Vec<Entity>, Vec<Relation>) = if args.skip_ner {
        println!("\nSkipping local NER (--skip-ner), using LLM-only extraction");
        (Vec::new(), Vec::new())
    } else if args.extract_relations {
        // Use multitask model for NER + relation extraction
        let (ents, rels) = run_ner_and_relations(
            &chunks, &labels,
            &args.multitask_model_dir, args.threads, &args.multitask_variant, args.gpu,
        );
        println!("\nLocal NER+RE total: {} entities, {} relations", ents.len(), rels.len());
        (ents, rels)
    } else {
        // NER-only with small model
        let ner_entities = run_ner(&chunks, &labels, &args.model_dir, args.threads, &args.onnx_variant, args.gpu);
        println!("\nNER total: {} raw entities", ner_entities.len());
        (ner_entities, Vec::new())
    };

    // Step 2: LLM extraction + gleaning (if not --ner-only)
    if !args.ner_only {
        let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
        if let Some(key) = api_key {
            println!("\nStarting LLM relation extraction + gleaning...");
            for (i, chunk) in chunks.iter().enumerate() {
                // Filter NER entities for this chunk
                let chunk_entities: Vec<Entity> = all_entities
                    .iter()
                    .filter(|e| {
                        e.properties
                            .get("source_chunk")
                            .and_then(|v| v.as_u64())
                            .map_or(false, |c| c == i as u64)
                    })
                    .cloned()
                    .collect();

                let result = llm_extract(chunk, &chunk_entities, &key, i, chunks.len()).await;
                all_entities.extend(result.entities);
                all_relations.extend(result.relations);
            }
        } else {
            println!("\nNo ANTHROPIC_API_KEY — skipping LLM extraction");
        }
    }

    // Step 3: Deduplicate (LightRAG-style)
    println!(
        "\nDeduplicating {} entities, {} relations...",
        all_entities.len(),
        all_relations.len()
    );
    let result = deduplicate(all_entities, all_relations);
    println!(
        "After dedup: {} entities, {} relations",
        result.entities.len(),
        result.relations.len()
    );

    // Save JSON (use /tmp fallback if source dir is read-only)
    let base_path = format!(
        "{}_extracted.json",
        args.source
            .trim_end_matches(".txt")
            .trim_end_matches(".md")
    );
    let json_path = match fs::write(&base_path, serde_json::to_string_pretty(&result)?) {
        Ok(()) => base_path,
        Err(_) => {
            let fallback = format!("/tmp/{}", std::path::Path::new(&base_path)
                .file_name().unwrap_or_default().to_string_lossy());
            fs::write(&fallback, serde_json::to_string_pretty(&result)?)?;
            fallback
        }
    };
    println!("Saved to {json_path}");

    if args.dry_run {
        println!("\n=== Entities ({}) ===", result.entities.len());
        for e in &result.entities {
            println!("  [{}] {}: {}", e.entity_type, e.name, e.description);
        }
        let entity_type_lookup: HashMap<String, &str> = result
            .entities
            .iter()
            .filter(|e| !e.entity_type.is_empty())
            .map(|e| (e.name.to_lowercase(), e.entity_type.as_str()))
            .collect();
        println!("\n=== Relations ({}) ===", result.relations.len());
        for r in &result.relations {
            let from_type = entity_type_lookup
                .get(&r.from.to_lowercase())
                .copied()
                .unwrap_or("?");
            let to_type = entity_type_lookup
                .get(&r.to.to_lowercase())
                .copied()
                .unwrap_or("?");
            println!(
                "  {} [{}] --{}--> {} [{}]",
                r.from, from_type, r.rel_type, r.to, to_type
            );
        }
        return Ok(());
    }

    // Step 4: Merge into FalkorDB
    println!("\nMerging into FalkorDB...");
    match merge_to_falkordb(&result).await {
        Ok((nodes, rels)) => {
            println!("  Nodes created: {nodes}");
            println!("  Relations created: {rels}");
        }
        Err(e) => eprintln!("FalkorDB merge failed: {e}"),
    }

    println!("\nDone! Open http://localhost:3000 to explore.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ── Deterministic edge-case tests ────────────────────────────────────

    #[test]
    fn chunk_empty_string() {
        let chunks = chunk_text("", 2400, 400);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_single_char() {
        let chunks = chunk_text("x", 2400, 400);
        assert_eq!(chunks, vec!["x"]);
    }

    #[test]
    fn chunk_shorter_than_overlap() {
        // This was the original infinite loop trigger: text.len() < overlap_chars
        let chunks = chunk_text("short", 2400, 400);
        assert_eq!(chunks, vec!["short"]);
    }

    #[test]
    fn chunk_exactly_chunk_size() {
        let text = "a".repeat(2400);
        let chunks = chunk_text(&text, 2400, 400);
        assert_eq!(chunks, vec![text]);
    }

    #[test]
    fn chunk_overlap_larger_than_chunk() {
        // Pathological: overlap > chunk_chars — must still terminate
        let text = "a".repeat(100);
        let chunks = chunk_text(&text, 10, 50);
        assert!(!chunks.is_empty());
        // First chunk starts at 0, last chunk must include final byte
        assert!(chunks.first().unwrap().starts_with('a'));
        assert!(chunks.last().unwrap().ends_with('a'));
    }

    #[test]
    fn chunk_multibyte_utf8() {
        // 3-byte chars: ensure char boundary handling works
        let text = "日本語のテスト文字列です。これはテストです。";
        let chunks = chunk_text(text, 10, 3);
        // Must cover the full text and not panic
        let reconstructed_len: usize = chunks.iter().map(|c| c.len()).max().unwrap_or(0);
        assert!(reconstructed_len > 0);
        // First and last chars must be present
        assert!(chunks.first().unwrap().starts_with('日'));
        assert!(chunks.last().unwrap().ends_with('。'));
    }

    #[test]
    fn chunk_paragraph_boundary() {
        let text = format!("{}.\n\n{}", "a".repeat(1500), "b".repeat(1500));
        let chunks = chunk_text(&text, 2400, 400);
        // Should break at the paragraph boundary
        assert!(chunks.len() >= 2);
        assert!(chunks[0].ends_with("\n\n"));
    }

    #[test]
    fn chunk_sentence_boundary() {
        let text = format!("{}. {}", "a".repeat(1500), "b".repeat(1500));
        let chunks = chunk_text(&text, 2400, 400);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].ends_with(". "));
    }

    #[test]
    fn chunk_all_newlines() {
        // Dense paragraph boundaries everywhere
        let text = "\n\n".repeat(500);
        let chunks = chunk_text(&text, 100, 50);
        assert!(!chunks.is_empty());
        let total_unique: usize = chunks.len();
        // Should not explode into millions of chunks
        assert!(total_unique < text.len());
    }

    #[test]
    fn chunk_coverage_no_gaps() {
        // Use non-repeating text so find() is unambiguous
        let text: String = (0..500).map(|i| format!("w{i} ")).collect();
        let chunks = chunk_text(&text, 100, 20);
        // First chunk starts at 0, last chunk ends at text.len()
        assert!(text.starts_with(&chunks[0]), "First chunk must be a prefix");
        assert!(text.ends_with(chunks.last().unwrap()), "Last chunk must be a suffix");
        // Verify sequential overlap: each chunk's start must be within the previous chunk
        let mut prev_end = chunks[0].len();
        for (i, chunk) in chunks.iter().enumerate().skip(1) {
            let pos = text.find(chunk.as_str()).expect("Chunk not found in text");
            assert!(pos < prev_end, "Gap between chunk {} (ends {}) and chunk {} (starts {})", i - 1, prev_end, i, pos);
            prev_end = pos + chunk.len();
        }
        assert_eq!(prev_end, text.len(), "Chunks don't reach end of text");
    }

    // ── Property-based tests (proptest) ──────────────────────────────────

    proptest! {
        /// chunk_text must always terminate and return non-empty for non-empty input.
        #[test]
        fn prop_terminates_and_nonempty(
            text in ".{1,5000}",
            chunk_chars in 1usize..5000,
            overlap_chars in 0usize..5000,
        ) {
            let chunks = chunk_text(&text, chunk_chars, overlap_chars);
            prop_assert!(!chunks.is_empty(), "Non-empty text must produce at least one chunk");
        }

        /// The first chunk must start at the beginning of the text.
        #[test]
        fn prop_first_chunk_starts_at_beginning(
            text in ".{1,5000}",
            chunk_chars in 1usize..5000,
            overlap_chars in 0usize..5000,
        ) {
            let chunks = chunk_text(&text, chunk_chars, overlap_chars);
            prop_assert!(text.starts_with(&chunks[0]),
                "First chunk must be a prefix of the text");
        }

        /// The last chunk must end at the end of the text.
        #[test]
        fn prop_last_chunk_ends_at_end(
            text in ".{1,5000}",
            chunk_chars in 1usize..5000,
            overlap_chars in 0usize..5000,
        ) {
            let chunks = chunk_text(&text, chunk_chars, overlap_chars);
            prop_assert!(text.ends_with(chunks.last().unwrap()),
                "Last chunk must be a suffix of the text");
        }

        /// Number of chunks must be bounded: at most text.len() chunks.
        #[test]
        fn prop_bounded_chunk_count(
            text in ".{1,5000}",
            chunk_chars in 1usize..5000,
            overlap_chars in 0usize..5000,
        ) {
            let chunks = chunk_text(&text, chunk_chars, overlap_chars);
            prop_assert!(chunks.len() <= text.len(),
                "Got {} chunks for text of {} bytes", chunks.len(), text.len());
        }

        /// Every chunk must be a valid substring of the original text.
        #[test]
        fn prop_chunks_are_substrings(
            text in ".{1,3000}",
            chunk_chars in 10usize..3000,
            overlap_chars in 0usize..1000,
        ) {
            let chunks = chunk_text(&text, chunk_chars, overlap_chars);
            for (i, chunk) in chunks.iter().enumerate() {
                prop_assert!(text.contains(chunk.as_str()),
                    "Chunk {} is not a substring of the original text", i);
            }
        }

        /// No chunk may be empty.
        #[test]
        fn prop_no_empty_chunks(
            text in ".{1,5000}",
            chunk_chars in 1usize..5000,
            overlap_chars in 0usize..5000,
        ) {
            let chunks = chunk_text(&text, chunk_chars, overlap_chars);
            for (i, chunk) in chunks.iter().enumerate() {
                prop_assert!(!chunk.is_empty(), "Chunk {} is empty", i);
            }
        }

        /// Stress test with extreme overlap ratios.
        #[test]
        fn prop_extreme_overlap(
            text in ".{1,1000}",
            chunk_chars in 1usize..100,
        ) {
            // overlap = 10x chunk (pathological)
            let overlap = chunk_chars * 10;
            let chunks = chunk_text(&text, chunk_chars, overlap);
            prop_assert!(!chunks.is_empty());
            prop_assert!(chunks.len() <= text.len());
        }
    }
}
