"""Public DRIFT training CLI."""

from __future__ import annotations

import argparse
import multiprocessing
import re
from datetime import datetime
from pathlib import Path

from drift.utils.constants import COMPRESSION_TOKEN
from drift.utils.stages import DRIFTStage, normalize_stage


def extract_model_size(model_path: str) -> str:
    match = re.search(r"-(\d+(?:\.\d+)?)B-", model_path)
    if match:
        return f"{match.group(1)}B"

    match = re.search(r"[/-](\d+(?:\.\d+)?)B(?:-|$|/)", model_path)
    if match:
        return f"{match.group(1)}B"

    return "Unknown"


def extract_model_combination_from_checkpoint_path(checkpoint_path: str) -> str | None:
    match = re.search(
        r"(?:MoM|DRIFT)_(?:Qwen2?\.?\d*|Mistral|LLaMA|Llama|Phi)-"
        r"(\d+(?:\.\d+)?Bx\d+(?:\.\d+)?B)",
        checkpoint_path,
    )
    if match:
        return match.group(1)

    match = re.search(r"(\d+(?:\.\d+)?Bx\d+(?:\.\d+)?B)", checkpoint_path)
    if match:
        return match.group(1)

    return None


def generate_model_combination_name(reasoning_model: str, knowledge_model: str) -> str:
    combo = extract_model_combination_from_checkpoint_path(reasoning_model)
    if combo:
        return combo

    combo = extract_model_combination_from_checkpoint_path(knowledge_model)
    if combo:
        return combo

    return f"{extract_model_size(reasoning_model)}x{extract_model_size(knowledge_model)}"


def detect_model_type_from_path(model_path: str) -> str:
    if "Qwen" in model_path:
        return "Qwen2.5"
    if "Mistral" in model_path:
        return "Mistral"
    if "Llama" in model_path or "LLaMA" in model_path:
        return "Llama"
    if "Phi" in model_path:
        return "Phi-3"
    return "Qwen2.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DRIFT training")

    parser.add_argument("--reasoning-model", "--model_path", dest="reasoning_model", required=True)
    parser.add_argument("--knowledge-model", "--aux_model_path", dest="knowledge_model", default=None)
    parser.add_argument("--projector-path", "--projector_path", dest="projector_path", default=None)
    parser.add_argument("--main-device", "--main_device", dest="main_device", default="cuda:0")
    parser.add_argument("--device-reasoning", "--device_main", dest="device_reasoning", default="auto")
    parser.add_argument("--device-knowledge", "--device_aux", dest="device_knowledge", default="auto")

    parser.add_argument("--stage", "--phase", dest="stage", default=DRIFTStage.LFRP.value)
    parser.add_argument("--compress-ratio", "--compress_ratio", dest="compress_ratio", type=int, default=32)
    parser.add_argument("--compress-mode", "--compress_mode", dest="compress_mode", default="small_threshold")
    parser.add_argument("--small-compress-model", "--small_compress_model", dest="small_compress_model", action="store_true")
    parser.add_argument("--use-layer-norm", "--use_layer_norm", dest="use_layer_norm", action="store_true")
    parser.add_argument("--chunk-size", "--chunk_size", dest="chunk_size", type=int, default=4096)
    parser.add_argument("--overlap", type=int, default=200)

    parser.add_argument("--frozen-reasoning", "--frozen_main", dest="frozen_reasoning", action="store_true")
    parser.add_argument("--frozen-projector", "--frozen_projector", dest="frozen_projector", action="store_true")
    parser.add_argument("--frozen-knowledge", "--frozen_auxiliary", dest="frozen_knowledge", action="store_true")

    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.01)
    parser.add_argument("--max-epochs", "--max_epochs", dest="max_epochs", type=int, default=2)
    parser.add_argument("--accumulate-grad-batches", "--accumulate_grad_batches", dest="accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", type=float, default=0.1)
    parser.add_argument("--scheduler-type", "--scheduler_type", dest="scheduler_type", default="cosine")
    parser.add_argument("--gradient-clip-val", "--gradient_clip_val", dest="gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--label-smoothing-factor", "--label_smoothing_factor", dest="label_smoothing_factor", type=float, default=0.1)
    parser.add_argument("--save-steps", "--save_steps", dest="save_steps", type=int, default=150)
    parser.add_argument("--log-steps", "--log_steps", dest="log_steps", type=int, default=10)
    parser.add_argument("--validation-steps", "--validation_steps", dest="validation_steps", type=int, default=10)
    parser.add_argument("--token-range", "--token_range", dest="token_range", default="64~128")
    parser.add_argument("--kl-weight-initial", "--kl_weight_initial", dest="kl_weight_initial", type=float, default=0.0)
    parser.add_argument("--enable-kl-weight-decay", "--enable_kl_weight_decay", dest="enable_kl_weight_decay", action="store_true")
    parser.add_argument(
        "--kl-weight-decay-strategy",
        "--kl_weight_decay_strategy",
        dest="kl_weight_decay_strategy",
        choices=["linear", "sqrt", "exponential", "cosine", "maintain_decay", "no_decay"],
        default="sqrt",
    )

    parser.add_argument("--lora-alpha", "--lora_alpha", dest="lora_alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", "--lora_dropout", dest="lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora-r", "--lora_r", dest="lora_r", type=int, default=16)

    parser.add_argument("--train-file", "--train_file", dest="train_file", required=True)
    parser.add_argument("--val-file", "--val_file", dest="val_file", default=None)
    parser.add_argument("--train-batch-size", "--train_batch_size", dest="train_batch_size", type=int, default=16)
    parser.add_argument("--val-batch-size", "--val_batch_size", dest="val_batch_size", type=int, default=2)
    parser.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=8192)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--dataset-num-proc", "--dataset_num_proc", dest="dataset_num_proc", type=int, default=64)
    parser.add_argument("--num-attention-heads", "--num_attention_heads", dest="num_attention_heads", type=int, default=8)
    parser.add_argument("--response-template", "--response_template", dest="response_template", default="<|im_start|>assistant\n")
    parser.add_argument("--response-end-marker", "--response_end_marker", dest="response_end_marker", default="<|im_end|>")

    parser.add_argument("--use-wandb", "--use_wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="DRIFT_Training")
    parser.add_argument("--wandb-name", "--wandb_name", dest="wandb_name", default=None)
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default=None)

    return parser.parse_args()


def configure_tokenizers(
    model,
    model_type: str,
    *,
    frozen_reasoning: bool = False,
    frozen_knowledge: bool = False,
) -> None:
    from transformers import (
        AddedToken,
        LlamaTokenizer,
        LlamaTokenizerFast,
        Qwen2Tokenizer,
        Qwen2TokenizerFast,
    )

    main_tokenizer = model.main_tokenizer
    aux_tokenizer = model.auxiliary_tokenizer
    main_had_compression_token = COMPRESSION_TOKEN in main_tokenizer.get_vocab()
    aux_had_compression_token = COMPRESSION_TOKEN in aux_tokenizer.get_vocab()

    old_main_size = len(main_tokenizer)
    if isinstance(main_tokenizer, (Qwen2Tokenizer, Qwen2TokenizerFast)):
        main_tokenizer.pad_token = "<|endoftext|>"
    elif isinstance(main_tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
        main_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        main_tokenizer.pad_token = "<pad>"
    else:
        main_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        main_tokenizer.pad_token = "<pad>"

    if not main_had_compression_token:
        if model_type == "Phi-3":
            cps_token = AddedToken(
                COMPRESSION_TOKEN,
                lstrip=False,
                rstrip=False,
                normalized=False,
                special=True,
            )
            main_tokenizer.add_special_tokens({"additional_special_tokens": [cps_token]})
        else:
            main_tokenizer.add_tokens([COMPRESSION_TOKEN])
        if len(main_tokenizer) > old_main_size:
            model.main_model.resize_token_embeddings(len(main_tokenizer))

    old_aux_size = len(aux_tokenizer)
    if frozen_knowledge and not aux_had_compression_token:
        raise ValueError(
            "Cannot freeze the knowledge model when its tokenizer/model does not "
            f"already contain {COMPRESSION_TOKEN}. Use a previous DRIFT knowledge "
            "checkpoint or train the knowledge branch unfrozen first."
        )

    if not aux_had_compression_token:
        aux_tokenizer.add_tokens([COMPRESSION_TOKEN])
        if len(aux_tokenizer) > old_aux_size:
            model.auxiliary_model.resize_token_embeddings(len(aux_tokenizer))

    model.compression_token_id = main_tokenizer.convert_tokens_to_ids(COMPRESSION_TOKEN)
    main_tokenizer.padding_side = "left"
    if aux_tokenizer.pad_token is None:
        aux_tokenizer.pad_token = aux_tokenizer.eos_token or "<|endoftext|>"
    aux_tokenizer.padding_side = "left"


def freeze_module(module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def build_checkpoint_dir(args: argparse.Namespace, stage: DRIFTStage, model_type: str, model_combo: str, time_str: str) -> str:
    if args.checkpoint_dir:
        return args.checkpoint_dir

    if stage == DRIFTStage.QAFT_QA:
        if args.kl_weight_initial > 0.0:
            kl_part = f"with_KL_{args.kl_weight_initial}/{args.kl_weight_decay_strategy}"
        else:
            kl_part = "without_KL"
    else:
        kl_part = ""

    remaining = (
        f"small_{args.small_compress_model}-"
        f"layernorm_{args.use_layer_norm}-"
        f"frozenreasoning_{args.frozen_reasoning}-"
        f"frozenprojector_{args.frozen_projector}"
    )

    base = (
        Path("./checkpoints/final_paper_new")
        / f"stage_{stage.value}"
        / f"DRIFT_{model_type}-{model_combo}"
        / f"token_{args.token_range}"
    )
    if kl_part:
        base = base / kl_part
    return str(base / f"ratio_{args.compress_ratio}" / f"{args.compress_mode}-{time_str}" / remaining)


def main() -> None:
    args = parse_args()
    stage = normalize_stage(args.stage)
    multiprocessing.set_start_method("spawn", force=True)

    import torch
    from peft import LoraConfig, get_peft_model

    from drift.data.datamodule import DRIFTDataModule
    from drift.modeling_drift import DRIFTModel
    from drift.training.trainer import train_model

    try:
        import wandb
    except ImportError:  # pragma: no cover - optional dependency
        wandb = None

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"

    knowledge_model = args.knowledge_model
    if knowledge_model is None:
        if args.small_compress_model:
            raise ValueError(
                "--small-compress-model no longer selects a private default path. "
                "Please pass --knowledge-model explicitly."
            )
        knowledge_model = args.reasoning_model

    model_combo = generate_model_combination_name(args.reasoning_model, knowledge_model)
    model_type = detect_model_type_from_path(args.reasoning_model)
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    wandb_name = args.wandb_name or (
        f"stage_{stage.value}-DRIFT_{model_type}-{model_combo}-token_{args.token_range}-"
        f"{time_str}-ratio_{args.compress_ratio}-{args.compress_mode}-small_{args.small_compress_model}-"
        f"layernorm_{args.use_layer_norm}-frozenreasoning_{args.frozen_reasoning}-"
        f"frozenprojector_{args.frozen_projector}"
    )
    checkpoint_dir = build_checkpoint_dir(args, stage, model_type, model_combo, time_str)

    print(f"Stage: {stage.value}")
    print(f"Reasoning model: {args.reasoning_model}")
    print(f"Knowledge model: {knowledge_model}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is required when --use-wandb is set.")
        wandb.init(project=args.wandb_project, name=wandb_name)

    model = DRIFTModel(
        main_model_name=args.reasoning_model,
        auxiliary_model_name=knowledge_model,
        num_attention_heads=args.num_attention_heads,
        device_map_main=args.device_reasoning,
        device_map_auxiliary=args.device_knowledge,
        device=args.main_device,
        attn_implementation=attn_implementation,
        frozen_main=args.frozen_reasoning,
        frozen_auxiliary=args.frozen_knowledge,
        frozen_projector=args.frozen_projector,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        use_layer_norm=args.use_layer_norm,
        projector_path=args.projector_path,
    )
    model.reasoning_model_name_or_path = args.reasoning_model
    model.knowledge_model_name_or_path = knowledge_model
    configure_tokenizers(
        model,
        model_type,
        frozen_reasoning=args.frozen_reasoning,
        frozen_knowledge=args.frozen_knowledge,
    )
    if args.frozen_reasoning:
        freeze_module(model.main_model)
    if args.frozen_knowledge:
        freeze_module(model.auxiliary_model)

    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        modules_to_save=["lm_head", "embed_tokens"],
    )
    if not args.frozen_reasoning:
        model.main_model = get_peft_model(model.main_model, peft_config)
    if not args.frozen_knowledge:
        model.auxiliary_model = get_peft_model(model.auxiliary_model, peft_config)

    data_module = DRIFTDataModule(
        train_file=args.train_file,
        val_file=args.val_file,
        main_tokenizer=model.main_tokenizer,
        aux_tokenizer=model.auxiliary_tokenizer,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        response_template=args.response_template,
        response_end_marker=args.response_end_marker,
        dataset_num_proc=args.dataset_num_proc,
        stage=stage,
        compress_ratio=args.compress_ratio,
        compress_mode=args.compress_mode,
        dataloader_shuffle=True,
    )
    data_module.setup()

    training_config = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "warmup_ratio": args.warmup_ratio,
        "scheduler_type": args.scheduler_type,
        "gradient_clip_val": args.gradient_clip_val,
        "label_smoothing_factor": args.label_smoothing_factor,
        "save_steps": args.save_steps,
        "checkpoint_dir": checkpoint_dir,
        "log_steps": args.log_steps,
        "validation_steps": args.validation_steps,
        "frozen_auxiliary": args.frozen_knowledge,
        "frozen_main": args.frozen_reasoning,
        "frozen_projector": args.frozen_projector,
        "use_wandb": args.use_wandb,
        "compress_ratio": args.compress_ratio,
        "compress_mode": args.compress_mode,
        "small_compress_model": args.small_compress_model,
        "phase": stage.value,
        "kl_weight_initial": args.kl_weight_initial,
        "enable_kl_weight_decay": args.enable_kl_weight_decay,
        "kl_weight_decay_strategy": args.kl_weight_decay_strategy,
    }

    try:
        stats = train_model(
            model=model,
            train_dataloader=data_module.train_dataloader(),
            val_dataloader=data_module.val_dataloader(),
            main_device=args.main_device,
            config=training_config,
        )
        if args.use_wandb:
            wandb.log(
                {
                    "best_loss": stats["best_loss"],
                    "total_training_time": stats["total_training_time"],
                    "final_epoch_loss": stats["epoch_losses"][-1],
                    "final_epoch_accuracy": stats["epoch_accuracies"][-1],
                }
            )
    finally:
        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
