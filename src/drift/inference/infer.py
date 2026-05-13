"""Single-example DRIFT inference CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drift.loading import load_drift_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRIFT single multi-context inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--document", default=None)
    parser.add_argument("--document-file", "--document_file", dest="document_file", default=None)
    parser.add_argument("--output-file", "--output_file", dest="output_file", default=None)
    parser.add_argument("--json", action="store_true", help="Print a JSON object instead of raw prediction text.")
    parser.add_argument("--reasoning-model", "--main_model", dest="reasoning_model", default=None)
    parser.add_argument("--knowledge-model", "--aux_model", dest="knowledge_model", default=None)
    parser.add_argument("--instruction", "--user-instruction", "--user_instruction", dest="instruction", default=None)
    parser.add_argument("--answer-prefix", "--answer_prefix", dest="answer_prefix", default=None)
    parser.add_argument("--compress-ratio", "--compress_ratio", dest="compress_ratio", type=int, default=32)
    parser.add_argument("--compress-mode", "--compress_mode", dest="compress_mode", default="small_threshold")
    parser.add_argument("--chunk-size", "--chunk_size", dest="chunk_size", type=int, default=8192)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-attention-heads", "--num_attention_heads", dest="num_attention_heads", type=int, default=8)
    parser.add_argument("--use-layer-norm", "--use_layer_norm", dest="use_layer_norm", action="store_true")
    return parser.parse_args()


def read_document(args: argparse.Namespace) -> str:
    if args.document is not None and args.document_file is not None:
        raise ValueError("Pass either --document or --document-file, not both.")
    if args.document_file is not None:
        return Path(args.document_file).read_text(encoding="utf-8")
    if args.document is not None:
        return args.document
    raise ValueError("Pass --document or --document-file.")


def main() -> None:
    args = parse_args()
    document = read_document(args)

    import torch

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = load_drift_model(
        checkpoint_path=args.checkpoint,
        reasoning_model_name_or_path=args.reasoning_model,
        knowledge_model_name_or_path=args.knowledge_model,
        num_attention_heads=args.num_attention_heads,
        device_map_reasoning={"": device},
        device_map_knowledge={"": device},
        device=device,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        use_layer_norm=args.use_layer_norm,
    )

    prediction = model.chat_multi_sft(
        contexts=[document],
        questions=[args.question],
        instruction_users=[args.instruction],
        answer_prefixs=[args.answer_prefix],
        max_new_tokens=args.max_new_tokens,
        compress_ratio=args.compress_ratio,
        compress_mode=args.compress_mode,
        batch_size=args.batch_size,
    )[0]

    if args.json or args.output_file:
        result = {
            "Question": args.question,
            "Prediction": prediction,
        }
        output = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        output = prediction

    if args.output_file:
        Path(args.output_file).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
