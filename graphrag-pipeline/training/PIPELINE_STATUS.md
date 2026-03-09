# ModernBERT Fine-Tuning Pipeline — Status Report

**Date:** 2026-03-09
**Goal:** Replace slow GLiNER multitask pipeline (72s for 77 chunks) with two specialized ModernBERT-base models for NER + RE, targeting <2s total inference via ONNX INT8.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Two-Stage Training                      │
│                                                          │
│  Stage 1: Pre-train on HuggingFace remote datasets       │
│    NER: Few-NERD (9,000 examples) → entity boundaries    │
│    RE:  SciERC  (4,184 examples) → relation patterns     │
│                       │                                   │
│  Stage 2: Fine-tune on domain data                       │
│    NER: 33 GPU optimization examples → 12 entity types   │
│    RE:  464 GPU optimization examples → 21 relation types│
│                       │                                   │
│  Export: ONNX + INT8 quantization                        │
│    NER: ~150 MB INT8 (ModernBERT token classification)   │
│    RE:  ~150 MB INT8 (ModernBERT sequence classification)│
│                       │                                   │
│  Integrate into Rust pipeline (main.rs)                  │
│    ort crate + tokenizers crate → ~21ms/chunk inference  │
└─────────────────────────────────────────────────────────┘
```

## Model Choice: ModernBERT-base

**Why ModernBERT over TiME (Tiny Monolingual Encoders):**
- ModernBERT-base: 22 layers, 768 hidden, 149M params, 21.3ms/chunk ONNX
- TiME-en-xs: 4 layers, 384 hidden, 103M params, 1.89ms/chunk ONNX
- TiME is faster but has poor embedding discrimination (cosine 0.96+ for everything)
- ModernBERT correctly separates unrelated concepts (GPU vs cooking = 0.39 cosine)
- 21ms × 77 chunks = 1.64s total — still **44x faster** than GLiNER's 72s

## Entity Types (12)

| Type | Description |
|------|-------------|
| hardware | GPUs, CPUs, chips (Radeon 8060S, A100) |
| gpu_feature | Hardware features (RDNA 3.5, Tensor Cores) |
| optimization_technique | Methods (tiling, loop unrolling) |
| algorithm | Algorithms (FlashAttention, Delta-Net) |
| software_framework | Software (PyTorch, CUDA, ROCm) |
| performance_metric | Numbers (67 tok/s, 212 GB/s) |
| memory_pattern | Memory access (coalesced, bank conflicts) |
| kernel_operation | Compute ops (GEMM, DELTA_NET_RECURRENCE) |
| model_architecture | Models (Qwen3.5, GPT-4, Mamba) |
| constraint | Limits (32 KB shared mem, wave64) |
| data_structure | Data structures (KV cache, state matrix) |
| research_paper | Papers (FlashRNN, Mamba2) |

## Relation Types (21)

`no_relation`, `TARGETS`, `IS_FEATURE_OF`, `IMPLEMENTS`, `USES`, `IMPROVES`, `REDUCES`, `MEASURES`, `LIMITS`, `IS_PART_OF`, `BUILDS_ON`, `EXTENDS`, `ENABLES`, `REQUIRES`, `INTRODUCES`, `VALIDATES`, `COMPETES_WITH`, `OPTIMIZES`, `ELIMINATES`, `COULD_IMPROVE`, `PORTS_TO`

RE uses typed entity markers: `"{head_type} [SEP] {tail_type} [SEP] text with [E1] head [/E1] and [E2] tail [/E2]"`

---

## Current State

### Data (all saved in `data/`)

| File | Examples | Source | Status |
|------|----------|--------|--------|
| `ner_train.json` | 33 | Hand-annotated (generate_data_inline.py) | Ready |
| `ner_val.json` | 8 | Hand-annotated | Ready |
| `re_train.json` | 464 | Hand-annotated (175 pos, 289 neg) | Ready |
| `re_val.json` | 120 | Hand-annotated | Ready |
| `pretrain_ner_train.json` | 9,000 | Few-NERD (DFKI-SLT/few-nerd) | Ready |
| `pretrain_ner_val.json` | 1,000 | Few-NERD | Ready |
| `pretrain_re_train.json` | 4,184 | SciERC (hrithikpiyush/scierc) | Ready |
| `pretrain_re_val.json` | 464 | SciERC | Ready |

### Pre-training Data Type Mappings

**NER (Few-NERD → our types):**
- `product-software` → software_framework (11,626 entities)
- `product-engine`, `product-other` → hardware (3,922)
- `other-chemicalthing` → data_structure (2,541)
- `organization-company` → software_framework
- Note: `other-algorithm` / `other-scientificterm` NOT in DFKI-SLT version

**RE (SciERC → our types):**
- USED-FOR → USES (233)
- FEATURE-OF → IS_FEATURE_OF (269)
- PART-OF + HYPONYM-OF → IS_PART_OF (991)
- COMPARE → COMPETES_WITH (264)
- no_relation (2,891)

### Checkpoints

| Model | Status | Metrics | Notes |
|-------|--------|---------|-------|
| NER (domain-only) | Trained | F1=0.39 | 10 epochs, early stopped at 9 |
| NER (two-stage) | **NOT STARTED** | — | Stopped at step 125/846 of Stage 1 |
| RE (domain-only) | Trained | macro_F1=0.096 | 15 epochs, early stopped at 7 |
| RE (two-stage) | **NOT STARTED** | — | Waiting for NER to finish |

The domain-only models have low metrics due to only 33/464 training examples. Two-stage training should significantly improve this.

### Previous Training Results (domain-only, for reference)

**NER:** Loss dropped 2.14 → 0.07, best F1=0.39 at epoch 9
**RE:** Best macro_F1=0.096 at epoch 3. Per-class: USES=0.75, TARGETS=0.19, most others=0.00. Model struggles with class imbalance (no_relation dominates).

---

## Files

```
graphrag-pipeline/training/
├── schema.py                  # Entity types, relation types, BIO tags, RE markers
├── generate_data_inline.py    # 41 hand-annotated examples (FlashRNN + GPU opt)
├── generate_data.py           # Claude Haiku API annotation (needs ANTHROPIC_API_KEY)
├── load_pretrain_data.py      # Downloads Few-NERD + SciERC from HuggingFace
├── train_ner.py               # NER training (--pretrain for two-stage)
├── train_re.py                # RE training (--pretrain for two-stage)
├── export_onnx.py             # ONNX export + INT8 quantization
├── requirements.txt           # Python dependencies
├── data/                      # Training data (domain + pretrain)
│   ├── ner_train.json         # 33 domain NER examples
│   ├── ner_val.json           # 8 domain NER examples
│   ├── re_train.json          # 464 domain RE examples
│   ├── re_val.json            # 120 domain RE examples
│   ├── pretrain_ner_train.json # 9,000 Few-NERD examples
│   ├── pretrain_ner_val.json   # 1,000 Few-NERD examples
│   ├── pretrain_re_train.json  # 4,184 SciERC examples
│   ├── pretrain_re_val.json    # 464 SciERC examples
│   └── raw_annotations/       # Raw annotation JSONL
└── checkpoints/
    ├── ner/best/               # Domain-only NER model (F1=0.39)
    └── re/best/                # Domain-only RE model (macro_F1=0.096)
```

---

## How to Resume

### 1. Run two-stage NER training (~4h on CPU)
```bash
cd graphrag-pipeline/training
python train_ner.py --pretrain --pretrain-epochs 3 --pretrain-lr 2e-5 --pretrain-batch-size 32 --epochs 10 --learning-rate 5e-5 --batch-size 16
```

### 2. Run two-stage RE training
```bash
python train_re.py --pretrain --pretrain-epochs 5 --pretrain-lr 2e-5 --pretrain-batch-size 32 --epochs 15 --learning-rate 3e-5 --batch-size 32
```

### 3. Export to ONNX + INT8
```bash
python export_onnx.py
# Outputs: ../models/modernbert_ner/ and ../models/modernbert_re/
```

### 4. Integrate into Rust pipeline
- Add `tokenizers` and `ndarray` to `Cargo.toml`
- Add `--modernbert` flag to `main.rs`
- Load ONNX models via `ort` crate
- Replace GLiNER inference path with ModernBERT NER → RE pipeline

### Tips
- Add `--dry-run` to any training script for a quick sanity check
- Pre-training data is already downloaded (cached in `data/pretrain_*.json`)
- To regenerate pretrain data: `python load_pretrain_data.py --max-ner-examples 10000`
- GPU training (if available): remove `fp16=False` in training scripts

---

## Known Issues

1. **CPU training is slow** — ~17s/step, Stage 1 NER takes ~4h. Consider GPU or reducing `--max-ner-examples`.
2. **Few-NERD type coverage** — DFKI-SLT version lacks `other-algorithm` and `other-scientificterm`. Only 4 of our 12 types get pre-training data (software_framework, hardware, data_structure). The model still learns general entity boundary detection.
3. **Deprecated HF datasets** — TACRED, FewRel, DocRED, REBEL, CrossNER, CrossRE all use deprecated loading scripts. Use Parquet-based uploads only (Few-NERD, SciERC work).
4. **RE class imbalance** — `no_relation` dominates. WeightedTrainer with inverse-frequency class weights is implemented but may need tuning.
5. **Domain data is small** — 33 NER / 464 RE examples. Consider running `generate_data.py` with ANTHROPIC_API_KEY for more data, or adding more examples to `generate_data_inline.py`.

---

## Remaining TODO

- [ ] Complete two-stage NER training (Stage 1 + Stage 2)
- [ ] Complete two-stage RE training
- [ ] Evaluate two-stage models (target: NER F1 > 0.7, RE macro_F1 > 0.3)
- [ ] Export ONNX + INT8 quantized models
- [ ] Add ModernBERT inference path to `main.rs` (`--modernbert` flag)
- [ ] Add `tokenizers`, `ndarray` deps to `Cargo.toml`
- [ ] End-to-end benchmark: GLiNER (72s) vs ModernBERT (<2s)
- [ ] Optional: add more domain training data via Claude Haiku API
- [ ] Optional: try NuNER (4.38M examples) for richer NER pre-training
