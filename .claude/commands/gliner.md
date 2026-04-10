GLiNER zero-shot NER and relation extraction reference. Use when working with GLiNER models, gline-rs, or the graphrag-pipeline.

## GLiNER Model Modes

GLiNER supports multiple extraction modes via different model architectures:

### 1. Span Mode (NER only)
- Models: `gliner_small-v2.1`, `gliner_large-v2.1`, `gliner_multi-v2.1`
- API: `GLiNER::<SpanMode>::new(params, runtime_params, tokenizer, model)`
- Input: `TextInput::from_str(&texts, &labels)`
- Output: `SpanOutput` — list of entity spans with class, text, probability
- Fast, small models (175MB int8). Good for high-throughput NER.

### 2. Token Mode (NER, multitask models)
- Models: `gliner-multitask-large-v0.5`, `gliner-relex-large-v0.5`
- API: `TokenPipeline::new(tokenizer)?.to_composable(&model, &params)`
- Same input/output as span mode but uses token-level classification
- Required for multitask models that also support relation extraction

### 3. Relation Extraction (via composed pipeline)
- Model: `gliner-multitask-large-v0.5` (same model does both NER + RE)
- Requires `TokenPipeline` (NER) chained with `RelationPipeline` (RE)
- Relations are schema-driven: define allowed subject/object entity types per relation

## gline-rs API (Rust crate v1.0.1)

### NER (Span Mode)
```rust
use gliner::model::GLiNER;
use gliner::model::pipeline::span::SpanMode;
use gliner::model::params::Parameters;
use gliner::model::input::text::TextInput;
use orp::params::RuntimeParameters;

let model = GLiNER::<SpanMode>::new(
    Parameters::default(),
    RuntimeParameters::default().with_threads(2),
    "models/gliner_small-v2.1/tokenizer.json",
    "models/gliner_small-v2.1/onnx/model_int8.onnx",
)?;
let input = TextInput::from_str(&["some text"], &["person", "company"])?;
let output = model.inference(input)?;
for spans in &output.spans {
    for span in spans {
        println!("{} [{}] {:.0}%", span.text(), span.class(), span.probability() * 100.0);
    }
}
```

### NER + Relation Extraction (Composed Pipeline)
```rust
use composable::*;
use orp::model::Model;
use orp::params::RuntimeParameters;
use gliner::model::params::Parameters;
use gliner::model::pipeline::{token::TokenPipeline, relation::RelationPipeline};
use gliner::model::input::{text::TextInput, relation::schema::RelationSchema};

let params = Parameters::default();
let model = Model::new("models/gliner-multitask-large-v0.5/onnx/model_q4f16.onnx",
                        RuntimeParameters::default())?;

let mut schema = RelationSchema::new();
schema.push_with_allowed_labels("USES", &["software_framework"], &["algorithm"]);
schema.push_with_allowed_labels("TARGETS", &["optimization_technique"], &["hardware"]);
// Or unconstrained:
schema.push("IMPROVES");

let pipeline = composed![
    TokenPipeline::new("models/gliner-multitask-large-v0.5/tokenizer.json")?
        .to_composable(&model, &params),
    RelationPipeline::default("models/gliner-multitask-large-v0.5/tokenizer.json", &schema)?
        .to_composable(&model, &params),
];

let input = TextInput::from_str(&["text"], &["person", "company"])?;
pipeline.apply(input)?;
```

### Output Structures
```rust
// Entity (from SpanOutput or TokenPipeline)
span.text()        -> &str      // "Bill Gates"
span.class()       -> &str      // "person"
span.probability() -> f32       // 0.999
span.offsets()     -> (usize, usize)

// Relation (from RelationOutput)
relation.subject()     -> &str  // "Bill Gates"
relation.object()      -> &str  // "Microsoft"
relation.class()       -> &str  // "founded"
relation.probability() -> f32   // 0.997
```

### Parameters
```rust
Parameters::default()
    .with_threshold(0.5)       // confidence threshold
    .with_flat_ner(true)       // no overlapping entities
    .with_multi_label(false)   // no overlapping different-class spans
    .with_max_length(Some(512)) // max sequence length
```

## Available Models (local)

| Model | Path | Size | Mode | Capabilities |
|-------|------|------|------|-------------|
| gliner_small-v2.1 | `models/gliner_small-v2.1/` | 175MB (int8) | Span | NER only |
| gliner-multitask-large-v0.5 | `models/gliner-multitask-large-v0.5/` | 519MB (q4f16) | Token | NER + Relations |

## ONNX Models on HuggingFace

| Repo | Tasks | License |
|------|-------|---------|
| `onnx-community/gliner_small-v2.1` | NER | Apache 2.0 |
| `onnx-community/gliner_large-v2.1` | NER | Apache 2.0 |
| `onnx-community/gliner-multitask-large-v0.5` | NER + RE | Apache 2.0 |
| `knowledgator/gliner-relex-large-v0.5` | NER + RE (needs ONNX conversion) | Apache 2.0 |

## Domain Entity Types (GPU optimization)

```
hardware, gpu_feature, optimization_technique, algorithm,
software_framework, performance_metric, memory_pattern,
kernel_operation, model_architecture, constraint,
data_structure, research_paper
```

## Domain Relation Types

```
IMPLEMENTS, USES, OPTIMIZES, TARGETS, IMPROVES, REDUCES,
ELIMINATES, MEASURES, LIMITS, ENABLES, EXTENDS, BUILDS_ON,
VALIDATES, COMPETES_WITH, IS_PART_OF, IS_FEATURE_OF,
REQUIRES, COULD_IMPROVE, INTRODUCES, PORTS_TO
```

## Docker

```bash
# NER only (fast, no API key needed)
docker compose run --rm graphrag --source sources/paper.txt --ner-only --dry-run

# Full pipeline (NER + LLM relations + FalkorDB)
ANTHROPIC_API_KEY=sk-... docker compose run --rm graphrag --source sources/paper.txt

# Skip local NER, LLM-only
docker compose run --rm graphrag --source sources/paper.txt --skip-ner
```
