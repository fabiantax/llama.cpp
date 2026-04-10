"""Export fine-tuned ModernBERT NER + RE models to ONNX with INT8 quantization.

Usage:
    python export_onnx.py                                    # export both
    python export_onnx.py --ner-only                         # NER only
    python export_onnx.py --re-only                          # RE only
    python export_onnx.py --skip-quantize                    # FP32 only
"""
import argparse
import json
import os
import shutil
import numpy as np
from pathlib import Path


def export_ner(checkpoint_dir: str, output_dir: str, quantize: bool = True):
    """Export NER model to ONNX."""
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Exporting NER model: {checkpoint_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Export to ONNX
    print("Exporting to ONNX...")
    ort_model = ORTModelForTokenClassification.from_pretrained(
        checkpoint_dir, export=True
    )
    ort_model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(output_dir)

    # Copy label map
    label_map_src = os.path.join(checkpoint_dir, "label_map.json")
    if os.path.exists(label_map_src):
        shutil.copy2(label_map_src, os.path.join(output_dir, "label_map.json"))

    # Check FP32 model size
    onnx_path = os.path.join(output_dir, "model.onnx")
    fp32_size = os.path.getsize(onnx_path) / 1e6
    print(f"FP32 model: {fp32_size:.1f} MB")

    if quantize:
        print("Quantizing to INT8...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(output_dir, "model_int8.onnx")
        quantize_dynamic(
            onnx_path,
            int8_path,
            weight_type=QuantType.QInt8,
        )
        int8_size = os.path.getsize(int8_path) / 1e6
        print(f"INT8 model: {int8_size:.1f} MB (compression: {fp32_size/int8_size:.1f}x)")

    # Validate
    print("Validating ONNX output...")
    test_text = "The DELTA_NET_RECURRENCE kernel runs on Radeon 8060S with RDNA 3.5."
    tokens = tokenizer(test_text, return_tensors="np", padding=True, truncation=True)
    outputs = ort_model(**{k: v for k, v in tokens.items()})
    logits = outputs.logits
    preds = np.argmax(logits, axis=-1)
    print(f"Test input: '{test_text}'")
    print(f"Output shape: {logits.shape}, predictions: {preds[0][:20]}...")

    # Load label map for human-readable output
    if os.path.exists(os.path.join(output_dir, "label_map.json")):
        lm = json.loads(Path(os.path.join(output_dir, "label_map.json")).read_text())
        id2tag = {int(k): v for k, v in lm["id2tag"].items()}
        token_labels = [id2tag.get(p, "O") for p in preds[0]]
        toks = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        entities_found = [(t, l) for t, l in zip(toks, token_labels) if l != "O" and l != ""]
        if entities_found:
            print(f"Entities found: {entities_found}")
        else:
            print("No entities found in test (model may need training data)")

    print(f"NER model exported to {output_dir}")


def export_re(checkpoint_dir: str, output_dir: str, quantize: bool = True):
    """Export RE model to ONNX."""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Exporting RE model: {checkpoint_dir}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Export to ONNX
    print("Exporting to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        checkpoint_dir, export=True
    )
    ort_model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(output_dir)

    # Copy label map
    label_map_src = os.path.join(checkpoint_dir, "label_map.json")
    if os.path.exists(label_map_src):
        shutil.copy2(label_map_src, os.path.join(output_dir, "label_map.json"))

    # Check sizes
    onnx_path = os.path.join(output_dir, "model.onnx")
    fp32_size = os.path.getsize(onnx_path) / 1e6
    print(f"FP32 model: {fp32_size:.1f} MB")

    if quantize:
        print("Quantizing to INT8...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = os.path.join(output_dir, "model_int8.onnx")
        quantize_dynamic(
            onnx_path,
            int8_path,
            weight_type=QuantType.QInt8,
        )
        int8_size = os.path.getsize(int8_path) / 1e6
        print(f"INT8 model: {int8_size:.1f} MB (compression: {fp32_size/int8_size:.1f}x)")

    # Validate
    print("Validating ONNX output...")
    test_text = "kernel_operation [SEP] hardware [SEP] The [E1] DELTA_NET_RECURRENCE [/E1] kernel runs on [E2] Radeon 8060S [/E2]."
    tokens = tokenizer(test_text, return_tensors="np", padding=True, truncation=True)
    outputs = ort_model(**{k: v for k, v in tokens.items()})
    logits = outputs.logits
    pred_class = np.argmax(logits, axis=-1)[0]

    if os.path.exists(os.path.join(output_dir, "label_map.json")):
        lm = json.loads(Path(os.path.join(output_dir, "label_map.json")).read_text())
        id2type = {int(k): v for k, v in lm["id2type"].items()}
        print(f"Test: predicted relation = {id2type.get(pred_class, '?')} (class {pred_class})")

    print(f"RE model exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ner-checkpoint", default="checkpoints/ner/best")
    parser.add_argument("--re-checkpoint", default="checkpoints/re/best")
    parser.add_argument("--ner-output", default="../models/modernbert_ner")
    parser.add_argument("--re-output", default="../models/modernbert_re")
    parser.add_argument("--ner-only", action="store_true")
    parser.add_argument("--re-only", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    args = parser.parse_args()

    quantize = not args.skip_quantize

    if not args.re_only:
        export_ner(args.ner_checkpoint, args.ner_output, quantize)

    if not args.ner_only:
        export_re(args.re_checkpoint, args.re_output, quantize)

    if not args.ner_only and not args.re_only:
        print(f"\n{'='*60}")
        print("Both models exported. Integration command:")
        print(f"  cargo run -- --source paper.txt --modernbert")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
