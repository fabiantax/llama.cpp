"""Download and prepare pre-training datasets from HuggingFace.

Downloads Few-NERD (NER) and SciERC (RE) datasets, maps their
entity/relation types to our domain schema, and saves in our training format.

Two-stage training strategy:
  Stage 1 (pre-train): Learn general NER/RE patterns on large remote datasets
  Stage 2 (fine-tune): Adapt to our 12 entity types + 21 relation types

Usage:
    python load_pretrain_data.py                         # download all
    python load_pretrain_data.py --ner-only              # NER datasets only
    python load_pretrain_data.py --re-only               # RE datasets only
    python load_pretrain_data.py --max-ner-examples 5000 # limit NER size
"""
import argparse
import json
import os
import re
import random
from collections import Counter

from datasets import load_dataset

from schema import ENTITY_TYPES, RELATION_TYPES

# ── Few-NERD fine-grained type → our entity type mapping ──────────────
# Few-NERD uses 66 fine-grained types WITHOUT BIO prefix.
# Consecutive tokens with the same type form one entity span.

FEWNERD_TYPE_MAP = {
    # product types → hardware / software
    "product-software": "software_framework",
    "product-engine": "hardware",
    "product-other": "hardware",
    # other types → algorithm / scientific
    "other-algorithm": "algorithm",
    "other-scientificterm": "algorithm",
    "other-chemicalthing": "data_structure",
    # organization → software_framework (tech companies)
    "organization-company": "software_framework",
}

# ── SciERC relation labels → our relation types ──────────────────────
# SciERC (hrithikpiyush/scierc) uses [[ head ]] and << tail >> markers.
# Labels 0-6 map to: no_relation, USED-FOR, FEATURE-OF, PART-OF,
#                     COMPARE, HYPONYM-OF, CONJUNCTION
# (Exact mapping confirmed from SciERC paper + label distribution)

SCIERC_REL_MAP = {
    0: "no_relation",
    1: "USES",           # USED-FOR → USES
    2: "IS_FEATURE_OF",  # FEATURE-OF → IS_FEATURE_OF
    3: "IS_PART_OF",     # PART-OF → IS_PART_OF
    4: "COMPETES_WITH",  # COMPARE → COMPETES_WITH
    5: "IS_PART_OF",     # HYPONYM-OF → IS_PART_OF (subtype relation)
    6: "no_relation",    # CONJUNCTION → no_relation (too vague)
}


def tokens_to_text_and_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join word tokens into text and track character offsets for each token."""
    text_parts = []
    offsets = []
    pos = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            text_parts.append(" ")
            pos += 1
        start = pos
        text_parts.append(tok)
        pos += len(tok)
        offsets.append((start, pos))
    return "".join(text_parts), offsets


def flat_tags_to_entities(
    tokens: list[str],
    tag_ids: list[int],
    tag_names: list[str],
    type_map: dict[str, str],
    offsets: list[tuple[int, int]],
) -> list[dict]:
    """Convert flat entity-type tags (no BIO prefix) to character-level spans.

    Few-NERD uses flat tags where consecutive tokens with the same non-O type
    form a single entity. Type boundaries are detected by tag changes.
    """
    entities = []
    current_type = None
    current_start = None
    current_end = None

    for i, tag_id in enumerate(tag_ids):
        tag_name = tag_names[tag_id] if tag_id < len(tag_names) else "O"

        if tag_name == "O":
            if current_type is not None:
                mapped = type_map.get(current_type)
                if mapped and mapped in ENTITY_TYPES:
                    entities.append({
                        "start": current_start,
                        "end": current_end,
                        "label": mapped,
                    })
                current_type = None
            continue

        if tag_name == current_type:
            # Extend current entity
            current_end = offsets[i][1]
        else:
            # Close previous entity
            if current_type is not None:
                mapped = type_map.get(current_type)
                if mapped and mapped in ENTITY_TYPES:
                    entities.append({
                        "start": current_start,
                        "end": current_end,
                        "label": mapped,
                    })
            # Start new entity
            current_type = tag_name
            current_start = offsets[i][0]
            current_end = offsets[i][1]

    # Close final entity
    if current_type is not None:
        mapped = type_map.get(current_type)
        if mapped and mapped in ENTITY_TYPES:
            entities.append({
                "start": current_start,
                "end": current_end,
                "label": mapped,
            })

    return entities


# ── Few-NERD loading ─────────────────────────────────────────────────

def load_fewnerd(max_examples: int = 0) -> list[dict]:
    """Load Few-NERD supervised split using fine-grained tags."""
    print("Downloading Few-NERD (supervised)...")
    ds = load_dataset("DFKI-SLT/few-nerd", "supervised")

    fine_tag_names = ds["train"].features["fine_ner_tags"].feature.names
    print(f"  Few-NERD: train={len(ds['train'])}, val={len(ds['validation'])}")
    print(f"  Fine-grained types: {len(fine_tag_names)}")

    mapped_types = [t for t in fine_tag_names if t in FEWNERD_TYPE_MAP]
    print(f"  Mapped types ({len(mapped_types)}): {mapped_types}")

    examples = []
    skipped = 0

    for split_name in ["train", "validation"]:
        for row in ds[split_name]:
            tokens = row["tokens"]
            fine_tags = row["fine_ner_tags"]
            text, offsets = tokens_to_text_and_offsets(tokens)

            entities = flat_tags_to_entities(
                tokens, fine_tags, fine_tag_names, FEWNERD_TYPE_MAP, offsets
            )

            if len(entities) > 0:
                examples.append({"text": text, "entities": entities})
            else:
                skipped += 1

            if max_examples > 0 and len(examples) >= max_examples:
                break
        if max_examples > 0 and len(examples) >= max_examples:
            break

    print(f"  Result: {len(examples)} examples with mapped entities (skipped {skipped})")

    type_counts = Counter()
    for ex in examples:
        for ent in ex["entities"]:
            type_counts[ent["label"]] += 1
    print("  Entity distribution:")
    for et, count in type_counts.most_common():
        print(f"    {et}: {count}")

    return examples


# ── SciERC loading ────────────────────────────────────────────────────

def parse_scierc_example(text: str, label: int) -> dict | None:
    """Parse a SciERC example with [[ head ]] and << tail >> markers.

    Returns an RE training example in our format, or None if parsing fails.
    """
    mapped_rel = SCIERC_REL_MAP.get(label, "no_relation")
    if mapped_rel not in RELATION_TYPES:
        return None

    # Find [[ head ]] markers
    head_match = re.search(r'\[\[\s*(.+?)\s*\]\]', text)
    tail_match = re.search(r'<<\s*(.+?)\s*>>', text)

    if not head_match or not tail_match:
        return None

    head_text = head_match.group(1).strip()
    tail_text = tail_match.group(1).strip()

    # Remove all markers to get clean text
    clean_text = text
    clean_text = re.sub(r'\[\[\s*', '', clean_text)
    clean_text = re.sub(r'\s*\]\]', '', clean_text)
    clean_text = re.sub(r'<<\s*', '', clean_text)
    clean_text = re.sub(r'\s*>>', '', clean_text)

    # Find entity positions in clean text
    head_idx = clean_text.find(head_text)
    tail_idx = clean_text.find(tail_text)

    if head_idx < 0 or tail_idx < 0:
        return None

    head = {
        "start": head_idx,
        "end": head_idx + len(head_text),
        "label": "algorithm",  # SciERC is about scientific methods
    }
    tail = {
        "start": tail_idx,
        "end": tail_idx + len(tail_text),
        "label": "algorithm",
    }

    return {
        "text": clean_text,
        "head": head,
        "tail": tail,
        "relation": mapped_rel,
    }


def load_scierc() -> list[dict]:
    """Load SciERC and convert to our RE format."""
    print("Downloading SciERC...")
    ds = load_dataset("hrithikpiyush/scierc")

    print(f"  SciERC: train={len(ds['train'])}, val={len(ds['validation'])}, test={len(ds['test'])}")

    examples = []
    failed = 0

    for split_name in ["train", "validation", "test"]:
        for row in ds[split_name]:
            text = row["text"]
            label = row["label"]

            result = parse_scierc_example(text, label)
            if result:
                examples.append(result)
            else:
                failed += 1

    print(f"  Result: {len(examples)} RE examples (failed to parse: {failed})")

    rel_counts = Counter(ex["relation"] for ex in examples)
    print("  Relation distribution:")
    for rt, count in rel_counts.most_common():
        print(f"    {rt}: {count}")

    return examples


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--max-ner-examples", type=int, default=10000,
                        help="Max NER pre-train examples (0=all)")
    parser.add_argument("--ner-only", action="store_true")
    parser.add_argument("--re-only", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.re_only:
        fewnerd = load_fewnerd(max_examples=args.max_ner_examples)
        all_ner = fewnerd
        random.shuffle(all_ner)
        print(f"\nTotal NER pre-train examples: {len(all_ner)}")

        if len(all_ner) == 0:
            print("WARNING: No NER examples generated.")
        else:
            val_size = max(1, int(len(all_ner) * args.val_split))
            ner_val = all_ner[:val_size]
            ner_train = all_ner[val_size:]

            for name, data in [
                ("pretrain_ner_train.json", ner_train),
                ("pretrain_ner_val.json", ner_val),
            ]:
                path = os.path.join(args.output_dir, name)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                print(f"Saved {path} ({len(data)} examples)")

    if not args.ner_only:
        scierc = load_scierc()

        if len(scierc) > 0:
            random.shuffle(scierc)
            val_size = max(1, int(len(scierc) * args.val_split))
            re_val = scierc[:val_size]
            re_train = scierc[val_size:]

            for name, data in [
                ("pretrain_re_train.json", re_train),
                ("pretrain_re_val.json", re_val),
            ]:
                path = os.path.join(args.output_dir, name)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                print(f"Saved {path} ({len(data)} examples)")
        else:
            print("WARNING: No RE examples generated.")

    print(f"\n{'='*60}")
    print("Pre-training data ready. Next steps:")
    print("  python train_ner.py --pretrain          # two-stage NER training")
    print("  python train_re.py --pretrain            # two-stage RE training")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
