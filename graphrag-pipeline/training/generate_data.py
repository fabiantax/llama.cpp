"""Generate NER + RE training data using Claude Haiku API.

Reads source texts, chunks them, and calls Claude Haiku to produce
character-level entity annotations and relation labels.

Usage:
    python generate_data.py --source ../sources/flashrnn.txt
    python generate_data.py --source ../sources/flashrnn.txt --max-chunks 10  # quick test
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from itertools import combinations

from schema import ENTITY_TYPES, RELATION_TYPES

# ── Chunker (mirrors main.rs) ──────────────────────────────────────────

def chunk_text(text: str, chunk_chars: int = 2400, overlap_chars: int = 400) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            # Try paragraph break
            slice_ = text[start:end]
            pos = slice_.rfind("\n\n")
            if pos is not None and pos > chunk_chars // 2:
                end = start + pos + 2
            else:
                pos = slice_.rfind(". ")
                if pos is not None and pos > chunk_chars // 2:
                    end = start + pos + 2
        chunks.append(text[start:end])
        if end >= len(text):
            break
        new_start = max(end - overlap_chars, start + 1)
        start = new_start
    return chunks


# ── Claude Haiku annotation ────────────────────────────────────────────

ANNOTATION_PROMPT = """You are an expert NLP annotator for GPU kernel optimization research.

Given the text below, extract:
1. All named entities with their character-level start/end offsets and entity type
2. All relationships between entity pairs

Entity types: {entity_types}

Relation types: {relation_types}

CRITICAL: Character offsets must be EXACT. The substring text[start:end] must match the entity text exactly. Count characters carefully including spaces and newlines.

Text:
---
{text}
---

Respond with ONLY valid JSON (no markdown, no explanation):
{{
  "entities": [
    {{"start": 0, "end": 10, "label": "entity_type", "text": "exact text"}}
  ],
  "relations": [
    {{"head_idx": 0, "tail_idx": 1, "relation": "RELATION_TYPE"}}
  ]
}}

Where head_idx and tail_idx are 0-based indices into the entities array.
Extract ALL entities you can find, including specific numbers/metrics, model names, techniques, etc."""


def annotate_chunk(text: str, client, chunk_idx: int, total: int) -> dict | None:
    """Call Claude Haiku to annotate a single chunk."""
    prompt = ANNOTATION_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES),
        relation_types=", ".join(t for t in RELATION_TYPES if t != "no_relation"),
        text=text,
    )

    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Extract JSON if wrapped in markdown
            if raw.startswith("```"):
                raw = re.sub(r"^```\w*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            data = json.loads(raw)
            return data
        except json.JSONDecodeError as e:
            print(f"  [!] JSON parse error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(1)
        except Exception as e:
            print(f"  [!] API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2)
    return None


# ── Validation ─────────────────────────────────────────────────────────

def validate_annotation(text: str, annotation: dict) -> dict:
    """Validate and clean annotation. Returns corrected version."""
    valid_entities = []
    for ent in annotation.get("entities", []):
        start = ent.get("start", -1)
        end = ent.get("end", -1)
        label = ent.get("label", "")
        ent_text = ent.get("text", "")

        # Validate entity type
        if label not in ENTITY_TYPES:
            continue

        # Validate offsets
        if start < 0 or end <= start or end > len(text):
            # Try to find the text in the chunk
            if ent_text:
                idx = text.find(ent_text)
                if idx >= 0:
                    start = idx
                    end = idx + len(ent_text)
                else:
                    # Case-insensitive search
                    idx = text.lower().find(ent_text.lower())
                    if idx >= 0:
                        start = idx
                        end = idx + len(ent_text)
                        ent_text = text[start:end]
                    else:
                        continue

        # Verify text matches offsets
        actual = text[start:end]
        if actual != ent_text:
            # Try to fix by searching
            if ent_text:
                idx = text.find(ent_text)
                if idx >= 0:
                    start = idx
                    end = idx + len(ent_text)
                else:
                    continue

        valid_entities.append({
            "start": start,
            "end": end,
            "label": label,
            "text": text[start:end],
        })

    # Validate relations
    valid_relations = []
    for rel in annotation.get("relations", []):
        head_idx = rel.get("head_idx", -1)
        tail_idx = rel.get("tail_idx", -1)
        relation = rel.get("relation", "")

        if relation not in RELATION_TYPES or relation == "no_relation":
            continue
        if head_idx < 0 or head_idx >= len(valid_entities):
            continue
        if tail_idx < 0 or tail_idx >= len(valid_entities):
            continue
        if head_idx == tail_idx:
            continue

        valid_relations.append({
            "head_idx": head_idx,
            "tail_idx": tail_idx,
            "relation": relation,
        })

    return {"entities": valid_entities, "relations": valid_relations}


# ── Convert to NER + RE training format ────────────────────────────────

def to_ner_examples(text: str, entities: list[dict]) -> dict:
    """Convert to NER training format (text + character-level spans)."""
    return {
        "text": text,
        "entities": entities,
    }


def to_re_examples(text: str, entities: list[dict], relations: list[dict]) -> list[dict]:
    """Convert to RE training examples (one per entity pair)."""
    examples = []

    # Positive examples from annotated relations
    positive_pairs = set()
    for rel in relations:
        head = entities[rel["head_idx"]]
        tail = entities[rel["tail_idx"]]
        examples.append({
            "text": text,
            "head": head,
            "tail": tail,
            "relation": rel["relation"],
        })
        positive_pairs.add((rel["head_idx"], rel["tail_idx"]))

    # Negative examples: entity pairs with no relation
    for i, j in combinations(range(len(entities)), 2):
        if (i, j) not in positive_pairs and (j, i) not in positive_pairs:
            examples.append({
                "text": text,
                "head": entities[i],
                "tail": entities[j],
                "relation": "no_relation",
            })

    return examples


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source text file(s), comma-separated")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit chunks (0 = all)")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--resume", action="store_true", help="Resume from saved raw annotations")
    args = parser.parse_args()

    # Load API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.resume:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "raw_annotations"), exist_ok=True)

    # Collect and chunk source texts
    all_chunks = []
    for src in args.source.split(","):
        src = src.strip()
        text = Path(src).read_text(encoding="utf-8")
        chunks = chunk_text(text)
        print(f"Source: {src} ({len(text)} chars) -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if args.max_chunks > 0:
        all_chunks = all_chunks[:args.max_chunks]
    print(f"Total chunks to annotate: {len(all_chunks)}")

    # Annotate chunks
    raw_path = os.path.join(args.output_dir, "raw_annotations", "annotations.jsonl")

    if args.resume and os.path.exists(raw_path):
        print(f"Resuming from {raw_path}")
        annotations = []
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                annotations.append(json.loads(line))
        print(f"Loaded {len(annotations)} existing annotations")
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        annotations = []
        with open(raw_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(all_chunks):
                print(f"[{i+1}/{len(all_chunks)}] Annotating ({len(chunk)} chars)...", end=" ")
                raw = annotate_chunk(chunk, client, i, len(all_chunks))
                if raw is None:
                    print("FAILED")
                    continue

                validated = validate_annotation(chunk, raw)
                n_ent = len(validated["entities"])
                n_rel = len(validated["relations"])
                print(f"{n_ent} entities, {n_rel} relations")

                record = {"chunk_idx": i, "text": chunk, **validated}
                annotations.append(record)
                f.write(json.dumps(record) + "\n")
                f.flush()

                # Rate limiting (Haiku is fast but let's be safe)
                if i < len(all_chunks) - 1:
                    time.sleep(0.2)

    print(f"\nTotal annotations: {len(annotations)}")

    # Convert to NER and RE training format
    ner_examples = []
    re_examples = []

    for ann in annotations:
        text = ann["text"]
        entities = ann["entities"]
        relations = ann["relations"]

        ner_examples.append(to_ner_examples(text, entities))
        re_examples.extend(to_re_examples(text, entities, relations))

    print(f"NER examples: {len(ner_examples)}")
    print(f"RE examples: {len(re_examples)} (positive: {sum(1 for e in re_examples if e['relation'] != 'no_relation')}, negative: {sum(1 for e in re_examples if e['relation'] == 'no_relation')})")

    # Split train/val (stratified for RE)
    import random
    random.seed(42)

    # NER split
    indices = list(range(len(ner_examples)))
    random.shuffle(indices)
    val_size = max(1, int(len(indices) * args.val_split))
    val_indices = set(indices[:val_size])
    ner_train = [ner_examples[i] for i in indices if i not in val_indices]
    ner_val = [ner_examples[i] for i in indices if i in val_indices]

    # RE split (keep examples from same chunk together)
    chunk_groups = {}
    for ex in re_examples:
        key = ex["text"][:100]
        chunk_groups.setdefault(key, []).append(ex)
    keys = list(chunk_groups.keys())
    random.shuffle(keys)
    val_keys = set(keys[:max(1, int(len(keys) * args.val_split))])
    re_train = [ex for k in keys if k not in val_keys for ex in chunk_groups[k]]
    re_val = [ex for k in keys if k in val_keys for ex in chunk_groups[k]]

    # Save
    for name, data in [
        ("ner_train.json", ner_train),
        ("ner_val.json", ner_val),
        ("re_train.json", re_train),
        ("re_val.json", re_val),
    ]:
        path = os.path.join(args.output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {path} ({len(data)} examples)")

    # Print stats
    entity_counts = {}
    for ann in annotations:
        for ent in ann["entities"]:
            entity_counts[ent["label"]] = entity_counts.get(ent["label"], 0) + 1
    print("\nEntity type distribution:")
    for et in sorted(entity_counts, key=entity_counts.get, reverse=True):
        print(f"  {et}: {entity_counts[et]}")

    rel_counts = {}
    for ex in re_examples:
        rel_counts[ex["relation"]] = rel_counts.get(ex["relation"], 0) + 1
    print("\nRelation type distribution:")
    for rt in sorted(rel_counts, key=rel_counts.get, reverse=True):
        print(f"  {rt}: {rel_counts[rt]}")


if __name__ == "__main__":
    main()
