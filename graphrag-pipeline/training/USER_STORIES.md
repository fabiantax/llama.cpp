# User Stories: ModernBERT NER+RE Fine-Tuning Pipeline

## Completed

### US-TRAIN-1: Domain Schema Definition
**As a** ML engineer,
**I want** a well-defined entity and relation schema for GPU optimization knowledge graphs,
**So that** NER and RE models can be trained with consistent, domain-specific labels.

**Acceptance Criteria:**
- [x] 12 entity types covering hardware, software, algorithms, metrics, and research
- [x] 21 relation types covering technical relationships (USES, IMPLEMENTS, IMPROVES, etc.)
- [x] BIO tagging scheme (25 labels: O + B/I for each entity type)
- [x] RE entity markers ([E1], [/E1], [E2], [/E2]) as special tokens
- [x] Type-to-ID and ID-to-type bidirectional mappings

**Files:** `schema.py`

---

### US-TRAIN-2: Domain Training Data Generation
**As a** ML engineer,
**I want** hand-annotated NER and RE training data from GPU optimization papers,
**So that** models can learn domain-specific entity and relation patterns.

**Acceptance Criteria:**
- [x] 41 annotated text chunks from FlashRNN paper and GPU optimization domain
- [x] 236 entity annotations with character-level offsets
- [x] 584 RE examples (175 positive relations, 409 negative)
- [x] 80/20 train/val split: NER 33/8, RE 464/120
- [x] Offset validation and entity text verification
- [x] Alternative API-based generator (generate_data.py) for scaling with Claude Haiku

**Files:** `generate_data_inline.py`, `generate_data.py`

---

### US-TRAIN-3: NER Model Training
**As a** ML engineer,
**I want** to fine-tune ModernBERT-base for token classification with BIO tags,
**So that** the model can identify GPU optimization entities in text.

**Acceptance Criteria:**
- [x] ModernBERT-base (149M params) with 25-label token classification head
- [x] Subword-to-character alignment via offset_mapping
- [x] seqeval F1 metric for evaluation
- [x] Early stopping (patience=3) on validation F1
- [x] Best model + label_map.json saved to checkpoints
- [x] Domain-only baseline: F1=0.39

**Files:** `train_ner.py`

---

### US-TRAIN-4: RE Model Training with Class Weighting
**As a** ML engineer,
**I want** to fine-tune ModernBERT-base for relation classification with typed entity markers,
**So that** the model can classify relationships between entity pairs.

**Acceptance Criteria:**
- [x] Typed entity marker approach: `"{head_type} [SEP] {tail_type} [SEP] text with [E1]...[/E1] and [E2]...[/E2]"`
- [x] 4 special tokens added to tokenizer, embeddings resized
- [x] Inverse-frequency class weighting (WeightedTrainer) for imbalanced data
- [x] macro_F1 metric, early stopping (patience=4)
- [x] Domain-only baseline: macro_F1=0.096

**Files:** `train_re.py`

---

### US-TRAIN-5: HuggingFace Pre-training Data Loading
**As a** ML engineer,
**I want** to download and map remote NER/RE datasets from HuggingFace,
**So that** models can be pre-trained on larger datasets before domain fine-tuning.

**Acceptance Criteria:**
- [x] Few-NERD (DFKI-SLT/few-nerd): 9,000 NER examples loaded and mapped
- [x] SciERC (hrithikpiyush/scierc): 4,184 RE examples loaded and mapped
- [x] Fine-grained type mapping: Few-NERD 66 types → our 12 entity types
- [x] Relation mapping: SciERC 7 types → our 21 relation types
- [x] Flat-tag-to-entity conversion (Few-NERD uses no BIO prefix)
- [x] SciERC `[[ head ]]` / `<< tail >>` marker parsing
- [x] 90/10 train/val split saved as pretrain_*.json

**Files:** `load_pretrain_data.py`

---

### US-TRAIN-6: Two-Stage Training Pipeline
**As a** ML engineer,
**I want** to pre-train on large remote datasets then fine-tune on domain data,
**So that** models learn general NER/RE patterns before adapting to GPU optimization.

**Acceptance Criteria:**
- [x] `--pretrain` flag enables two-stage training
- [x] Stage 1: Lower learning rate (2e-5), larger batch size, shorter epochs
- [x] Stage 2: Higher learning rate (5e-5), domain data, full epochs
- [x] Best model from Stage 1 carries forward to Stage 2
- [x] Configurable: `--pretrain-epochs`, `--pretrain-lr`, `--pretrain-batch-size`
- [ ] Full training run completed (stopped at 15% of Stage 1 — CPU too slow)

**Files:** `train_ner.py`, `train_re.py`

---

### US-TRAIN-7: ONNX Export with INT8 Quantization
**As a** ML engineer,
**I want** to export trained models to ONNX format with INT8 quantization,
**So that** they can be loaded by the Rust pipeline via the `ort` crate for fast inference.

**Acceptance Criteria:**
- [x] NER: ORTModelForTokenClassification export
- [x] RE: ORTModelForSequenceClassification export
- [x] Dynamic INT8 quantization (onnxruntime.quantization)
- [x] Validation with test inputs and human-readable output
- [x] Label map copied to output directory
- [ ] Actual export run (waiting for better models from two-stage training)

**Files:** `export_onnx.py`

---

## Planned

### US-TRAIN-8: Rust Integration
**As a** developer,
**I want** a `--modernbert` flag in the Rust pipeline that uses ONNX models for NER+RE,
**So that** inference is 44x faster than the current GLiNER pipeline (72s → <2s).

**Acceptance Criteria:**
- [ ] `tokenizers` and `ndarray` crates added to Cargo.toml
- [ ] ModernBERT tokenizer loaded from exported model directory
- [ ] NER ONNX inference via `ort` crate with INT8 model
- [ ] RE ONNX inference with entity marker insertion
- [ ] Entity deduplication and relation merging (reuse existing logic)
- [ ] End-to-end benchmark: <2s for 77 chunks

**Files:** `src/main.rs`, `Cargo.toml`

---

### US-TRAIN-9: Training Data Scaling
**As a** ML engineer,
**I want** to generate more domain training data using the Claude Haiku API,
**So that** model quality improves beyond the current 41-example baseline.

**Acceptance Criteria:**
- [ ] Run `generate_data.py` with ANTHROPIC_API_KEY on multiple source papers
- [ ] Target: 200+ annotated chunks (currently 41)
- [ ] Validate offset accuracy and entity type coverage
- [ ] Retrain and measure F1 improvement

**Files:** `generate_data.py`
