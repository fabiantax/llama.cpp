"""Fine-tune ModernBERT-base for NER (token classification with BIO tags).

Supports two-stage training:
  Stage 1 (--pretrain): Pre-train on Few-NERD + CrossNER (general NER patterns)
  Stage 2: Fine-tune on domain-specific data (GPU optimization entities)

Usage:
    python train_ner.py                          # domain-only (stage 2)
    python train_ner.py --pretrain               # two-stage training
    python train_ner.py --epochs 3 --dry-run     # quick test
"""
import argparse
import json
import os
import numpy as np
import evaluate
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset
from schema import BIO_TAGS, BIO_TAG2ID, BIO_ID2TAG, ENTITY_TYPES


class NerDataset(Dataset):
    """NER dataset with subword-aligned BIO labels."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example["text"]
        entities = example["entities"]

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0)

        # Build character-level label array
        char_labels = ["O"] * len(text)
        for ent in entities:
            start = ent["start"]
            end = ent["end"]
            label = ent["label"]
            if start < 0 or end > len(text) or label not in ENTITY_TYPES:
                continue
            char_labels[start] = f"B-{label}"
            for c in range(start + 1, end):
                char_labels[c] = f"I-{label}"

        # Align to subword tokens
        labels = []
        for offset in offset_mapping:
            start, end = offset[0].item(), offset[1].item()
            if start == 0 and end == 0:
                labels.append(-100)
            else:
                labels.append(BIO_TAG2ID.get(char_labels[start], 0))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def compute_metrics(eval_preds):
    """Compute seqeval F1 for NER."""
    seqeval = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_tags = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_seq.append(BIO_ID2TAG.get(l, "O"))
            pred_seq_tags.append(BIO_ID2TAG.get(p, "O"))
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_tags)

    results = seqeval.compute(predictions=pred_labels, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }


def run_training_stage(
    model,
    tokenizer,
    train_data: list[dict],
    val_data: list[dict],
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    warmup_ratio: float,
    stage_name: str,
    patience: int = 3,
):
    """Run a single training stage (used for both pre-train and fine-tune)."""
    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"  LR: {learning_rate}, Epochs: {epochs}, Batch: {batch_size}")
    print(f"{'='*60}")

    train_dataset = NerDataset(train_data, tokenizer, max_length)
    val_dataset = NerDataset(val_data, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\n{stage_name} metrics: {metrics}")

    return trainer, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="checkpoints/ner")
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    # Two-stage training options
    parser.add_argument("--pretrain", action="store_true",
                        help="Enable two-stage training with pre-train data")
    parser.add_argument("--pretrain-epochs", type=int, default=3,
                        help="Epochs for pre-training stage")
    parser.add_argument("--pretrain-lr", type=float, default=2e-5,
                        help="Learning rate for pre-training (lower than fine-tune)")
    parser.add_argument("--pretrain-batch-size", type=int, default=32,
                        help="Batch size for pre-training")
    args = parser.parse_args()

    # Load domain data
    train_data = json.loads(Path(os.path.join(args.data_dir, "ner_train.json")).read_text(encoding="utf-8"))
    val_data = json.loads(Path(os.path.join(args.data_dir, "ner_val.json")).read_text(encoding="utf-8"))
    print(f"Domain NER train: {len(train_data)}, val: {len(val_data)}")

    if args.dry_run:
        train_data = train_data[:10]
        val_data = val_data[:5]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(BIO_TAGS),
        id2label=BIO_ID2TAG,
        label2id=BIO_TAG2ID,
    )
    print(f"Model: {args.model_name}, labels: {len(BIO_TAGS)}")

    if args.pretrain:
        # ── Stage 1: Pre-train on remote datasets ──────────────────
        pretrain_train_path = os.path.join(args.data_dir, "pretrain_ner_train.json")
        pretrain_val_path = os.path.join(args.data_dir, "pretrain_ner_val.json")

        if not os.path.exists(pretrain_train_path):
            print(f"ERROR: Pre-train data not found at {pretrain_train_path}")
            print("Run: python load_pretrain_data.py --ner-only")
            return

        pretrain_train = json.loads(Path(pretrain_train_path).read_text(encoding="utf-8"))
        pretrain_val = json.loads(Path(pretrain_val_path).read_text(encoding="utf-8"))

        if args.dry_run:
            pretrain_train = pretrain_train[:100]
            pretrain_val = pretrain_val[:20]

        stage1_dir = os.path.join(args.output_dir, "stage1_pretrain")
        trainer, _ = run_training_stage(
            model=model,
            tokenizer=tokenizer,
            train_data=pretrain_train,
            val_data=pretrain_val,
            output_dir=stage1_dir,
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
            learning_rate=args.pretrain_lr,
            max_length=args.max_length,
            warmup_ratio=args.warmup_ratio,
            stage_name="STAGE 1: Pre-training on Few-NERD + CrossNER",
            patience=2,
        )

        # Model is now the best from stage 1 (loaded by load_best_model_at_end)
        model = trainer.model
        print("\nStage 1 complete. Model now has general NER knowledge.")

    # ── Stage 2 (or single stage): Fine-tune on domain data ────────
    stage_name = "STAGE 2: Fine-tuning on domain data" if args.pretrain else "Training on domain data"
    stage2_dir = os.path.join(args.output_dir, "stage2_finetune") if args.pretrain else args.output_dir

    trainer, metrics = run_training_stage(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        output_dir=stage2_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        stage_name=stage_name,
        patience=3,
    )

    # Save best model
    best_dir = os.path.join(args.output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Save label map
    label_map = {"bio_tags": BIO_TAGS, "tag2id": BIO_TAG2ID, "id2tag": {str(k): v for k, v in BIO_ID2TAG.items()}}
    with open(os.path.join(best_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nFinal NER metrics: {metrics}")
    print(f"Model saved to {best_dir}")


if __name__ == "__main__":
    main()
