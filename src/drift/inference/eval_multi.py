"""Public DRIFT multi-context evaluation CLI."""

from __future__ import annotations

import argparse
import json
import logging
import re
import traceback
from typing import Any

from drift.loading import load_drift_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def read_jsonl(file_path: str) -> list[dict[str, Any]]:
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON line: %s", exc)
    logger.info("Loaded %d examples from %s", len(data), file_path)
    return data


def write_jsonl(data: list[dict[str, Any]], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Wrote %d results to %s", len(data), file_path)


def cuda_index(device: str) -> int | None:
    match = re.fullmatch(r"cuda(?::(\d+))?", device)
    if not match:
        return None
    return int(match.group(1) or 0)


def run_worker(device: str, data_chunk: list[tuple[int, dict[str, Any]]], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch
    from tqdm import tqdm

    if device.startswith("cuda"):
        index = cuda_index(device)
        if index is not None:
            torch.cuda.set_device(index)

    logger.info("Loading DRIFT model on %s", device)
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

    chunk_results = []
    for i in tqdm(range(0, len(data_chunk), args.batch_size), desc=f"eval {device}"):
        batch_slice = data_chunk[i : i + args.batch_size]
        indices = [item[0] for item in batch_slice]
        items = [item[1] for item in batch_slice]

        contexts = [item.get("Document", "") for item in items]
        questions = [item.get("Question", "") for item in items]
        instructions = [item.get("User_instruction", None) for item in items]
        answer_prefixes = [item.get("answer_prefix", None) for item in items]

        try:
            predictions = model.chat_multi_sft(
                contexts=contexts,
                questions=questions,
                instruction_users=instructions,
                answer_prefixs=answer_prefixes,
                max_new_tokens=args.max_new_tokens,
                compress_ratio=args.compress_ratio,
                compress_mode=args.compress_mode,
                batch_size=args.batch_size,
            )
            for orig_idx, item, pred in zip(indices, items, predictions):
                chunk_results.append({**item, "Prediction": pred, "orig_idx": orig_idx})
        except Exception as exc:
            logger.error("Batch failed on %s: %s", device, exc)
            logger.debug(traceback.format_exc())
            for orig_idx, item in zip(indices, items):
                chunk_results.append({**item, "Prediction": "ERROR", "orig_idx": orig_idx})

    logger.info("%s completed %d samples", device, len(chunk_results))
    return chunk_results


def process_worker(gpu_id: int, data_chunk, args: argparse.Namespace, return_dict) -> None:
    device = f"cuda:{gpu_id}"
    try:
        return_dict[gpu_id] = run_worker(device, data_chunk, args)
    except Exception:
        logger.error("GPU %s fatal error:\n%s", gpu_id, traceback.format_exc())
        return_dict[gpu_id] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRIFT multi-context evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-file", "--input_file", dest="input_file", required=True)
    parser.add_argument("--output-file", "--output_file", dest="output_file", required=True)
    parser.add_argument("--reasoning-model", "--main_model", dest="reasoning_model", default=None)
    parser.add_argument("--knowledge-model", "--aux_model", dest="knowledge_model", default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--compress-ratio", "--compress_ratio", dest="compress_ratio", type=int, default=32)
    parser.add_argument("--chunk-size", "--chunk_size", dest="chunk_size", type=int, default=8192)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--compress-mode", "--compress_mode", dest="compress_mode", default="small_threshold")
    parser.add_argument("--num-gpus", "--num_gpus", dest="num_gpus", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-attention-heads", "--num_attention_heads", dest="num_attention_heads", type=int, default=8)
    parser.add_argument("--use-layer-norm", "--use_layer_norm", dest="use_layer_norm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    all_raw_data = read_jsonl(args.input_file)
    indexed_data = list(enumerate(all_raw_data))

    if args.num_gpus <= 1:
        if args.device is not None:
            device = args.device
        elif torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        results = run_worker(device, indexed_data, args)
    else:
        import torch.multiprocessing as mp

        if args.device is not None:
            logger.warning("--device is ignored when --num-gpus is greater than 1")
        num_gpus = args.num_gpus
        chunks = [indexed_data[i::num_gpus] for i in range(num_gpus)]

        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        logger.info("Spawning %d GPU workers", num_gpus)

        for gpu_id in range(num_gpus):
            process = mp.Process(
                target=process_worker,
                args=(gpu_id, chunks[gpu_id], args, return_dict),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        results = []
        for gpu_id in range(num_gpus):
            results.extend(return_dict.get(gpu_id, []))

    results.sort(key=lambda item: item["orig_idx"])
    for item in results:
        item.pop("orig_idx", None)

    if not results and all_raw_data:
        raise RuntimeError("All evaluation workers failed to return results.")
    write_jsonl(results, args.output_file)


if __name__ == "__main__":
    main()
