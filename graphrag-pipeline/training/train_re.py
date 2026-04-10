"""Fine-tune ModernBERT-base for Relation Extraction (sequence classification).

Uses typed entity markers approach:
  "{head_type} [SEP] {tail_type} [SEP] text with [E1] head [/E1] and [E2] tail [/E2]"

Supports two-stage training:
  Stage 1 (--pretrain): Pre-train on CrossRE AI (general relation patterns)
  Stage 2: Fine-tune on domain-specific data (GPU optimization relations)

Usage:
    python train_re.py                          # domain-only (stage 2)
    python train_re.py --pretrain               # two-stage training
    python train_re.py --epochs 3 --dry-run     # quick test
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, classification_report

from schema import RELATION_TYPES, REL_TYPE2ID, REL_ID2TYPE, RE_MARKERS


class ReDataset(Dataset):
    """RE dataset with typed entity marker insertion."""

    def __init__(self, examples: list[dict], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["text"]
        head = ex["head"]
        tail = ex["tail"]
        relation = ex["relation"]

        # Insert entity markers into text
        marked_text = self._insert_markers(text, head, tail)

        # Prepend entity type info: "head_type [SEP] tail_type [SEP] marked_text"
        type_prefix = f"{head['label']} {self.tokenizer.sep_token} {tail['label']} {self.tokenizer.sep_token} "
        input_text = type_prefix + marked_text

        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(REL_TYPE2ID[relation], dtype=torch.long),
        }

    def _insert_markers(self, text: str, head: dict, tail: dict) -> str:
        """Insert [E1]/[E2] markers around entity mentions in text."""
        h_start, h_end = head["start"], head["end"]
        t_start, t_end = tail["start"], tail["end"]

        # Handle overlapping entities (shouldn't happen, but be safe)
        if h_start <= t_start:
            first, second = (h_start, h_end, "[E1]", "[/E1]"), (t_start, t_end, "[E2]", "[/E2]")
        else:
            first, second = (t_start, t_end, "[E2]", "[/E2]"), (h_start, h_end, "[E1]", "[/E1]")

        # Insert from right to left to preserve offsets
        result = text[:second[0]] + second[2] + text[second[0]:second[1]] + second[3] + text[second[1]:]
        # Adjust first entity's position (it's before second, so no offset change needed)
        result = result[:first[0]] + first[2] + result[first[0]:first[1] + len(second[2]) + len(second[3]) if first[1] > second[0] else first[1]] + first[3] + result[first[1] + (len(second[2]) + len(second[3]) if first[1] > second[0] else 0):]

        # Simpler approach: just rebuild
        parts = []
        positions = sorted([
            (h_start, "h_start"), (h_end, "h_end"),
            (t_start, "t_start"), (t_end, "t_end"),
        ])

        prev = 0
        for pos, marker_type in positions:
            parts.append(text[prev:pos])
            if marker_type == "h_start":
                parts.append("[E1] ")
            elif marker_type == "h_end":
                parts.append(" [/E1]")
            elif marker_type == "t_start":
                parts.append("[E2] ")
            elif marker_type == "t_end":
                parts.append(" [/E2]")
            prev = pos
        parts.append(text[prev:])

        return "".join(parts)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    # Accuracy excluding no_relation
    mask = labels != REL_TYPE2ID["no_relation"]
    if mask.sum() > 0:
        pos_acc = (preds[mask] == labels[mask]).mean()
    else:
        pos_acc = 0.0
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "positive_accuracy": pos_acc,
    }


def make_weighted_trainer(class_weights):
    """Create a Trainer subclass with weighted loss for the given class weights."""
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


def compute_class_weights(data: list[dict]) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced RE data."""
    all_labels = [REL_TYPE2ID[ex["relation"]] for ex in data]
    label_counts = Counter(all_labels)
    total = len(all_labels)
    n_classes = len(RELATION_TYPES)
    weights = torch.tensor([
        total / (n_classes * label_counts.get(i, 1)) for i in range(n_classes)
    ], dtype=torch.float32)
    print(f"  Class weights (max={weights.max():.1f}, min={weights.min():.2f})")
    return weights


def run_re_training_stage(
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
    patience: int = 4,
):
    """Run a single RE training stage."""
    print(f"\n{'='*60}")
    print(f"  {stage_name}")
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"  LR: {learning_rate}, Epochs: {epochs}, Batch: {batch_size}")
    print(f"{'='*60}")

    # Print class distribution
    train_counts = Counter(ex["relation"] for ex in train_data)
    print("  Train distribution:")
    for rel, count in train_counts.most_common():
        print(f"    {rel}: {count}")

    class_weights = compute_class_weights(train_data)
    WeightedTrainer = make_weighted_trainer(class_weights)

    train_dataset = ReDataset(train_data, tokenizer, max_length)
    val_dataset = ReDataset(val_data, tokenizer, max_length)

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
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=50,
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = WeightedTrainer(
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
    parser.add_argument("--output-dir", default="checkpoints/re")
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true")
    # Two-stage training options
    parser.add_argument("--pretrain", action="store_true",
                        help="Enable two-stage training with pre-train data")
    parser.add_argument("--pretrain-epochs", type=int, default=5,
                        help="Epochs for pre-training stage")
    parser.add_argument("--pretrain-lr", type=float, default=2e-5,
                        help="Learning rate for pre-training (lower than fine-tune)")
    parser.add_argument("--pretrain-batch-size", type=int, default=32)
    args = parser.parse_args()

    # Load domain data
    train_data = json.loads(Path(os.path.join(args.data_dir, "re_train.json")).read_text(encoding="utf-8"))
    val_data = json.loads(Path(os.path.join(args.data_dir, "re_val.json")).read_text(encoding="utf-8"))
    print(f"Domain RE train: {len(train_data)}, val: {len(val_data)}")

    if args.dry_run:
        train_data = train_data[:50]
        val_data = val_data[:20]

    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": RE_MARKERS})
    print(f"Added {num_added} special tokens: {RE_MARKERS}")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(RELATION_TYPES),
        id2label=REL_ID2TYPE,
        label2id=REL_TYPE2ID,
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model: {args.model_name}, labels: {len(RELATION_TYPES)}")

    if args.pretrain:
        # ── Stage 1: Pre-train on remote RE datasets ───────────────
        pretrain_train_path = os.path.join(args.data_dir, "pretrain_re_train.json")
        pretrain_val_path = os.path.join(args.data_dir, "pretrain_re_val.json")

        if not os.path.exists(pretrain_train_path):
            print(f"ERROR: Pre-train data not found at {pretrain_train_path}")
            print("Run: python load_pretrain_data.py --re-only")
            return

        pretrain_train = json.loads(Path(pretrain_train_path).read_text(encoding="utf-8"))
        pretrain_val = json.loads(Path(pretrain_val_path).read_text(encoding="utf-8"))

        if args.dry_run:
            pretrain_train = pretrain_train[:100]
            pretrain_val = pretrain_val[:20]

        stage1_dir = os.path.join(args.output_dir, "stage1_pretrain")
        trainer, _ = run_re_training_stage(
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
            stage_name="STAGE 1: Pre-training on CrossRE AI",
            patience=2,
        )

        model = trainer.model
        print("\nStage 1 complete. Model now has general RE knowledge.")

    # ── Stage 2 (or single stage): Fine-tune on domain data ────────
    stage_name = "STAGE 2: Fine-tuning on domain data" if args.pretrain else "Training on domain data"
    stage2_dir = os.path.join(args.output_dir, "stage2_finetune") if args.pretrain else args.output_dir

    trainer, metrics = run_re_training_stage(
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
        patience=4,
    )

    # Save best model
    best_dir = os.path.join(args.output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Save label map
    label_map = {
        "relation_types": RELATION_TYPES,
        "type2id": REL_TYPE2ID,
        "id2type": {str(k): v for k, v in REL_ID2TYPE.items()},
        "entity_markers": RE_MARKERS,
    }
    with open(os.path.join(best_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Final eval with classification report
    val_dataset = ReDataset(val_data, tokenizer, args.max_length)
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    target_names = [REL_ID2TYPE[i] for i in sorted(set(labels) | set(preds))]
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=target_names, zero_division=0))

    print(f"\nFinal RE metrics: {metrics}")
    print(f"Model saved to {best_dir}")


if __name__ == "__main__":
    main()
