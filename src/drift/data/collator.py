"""Batch collation for DRIFT training."""

from __future__ import annotations

from typing import Dict, List

from drift.data.labels import create_assistant_labels
from drift.data.processor import DRIFTProcessor


def drift_collate_fn(
    examples: List[Dict],
    processor: DRIFTProcessor,
    max_length: int,
    response_template: str = "<|im_start|>assistant\n",
    response_end_marker: str = "<|im_end|>",
    debug: bool = False,
):
    """Collate DRIFT examples and create assistant-only labels."""

    auxiliary_inputs = [example["auxiliary_input"] for example in examples]
    main_texts = [example["text"] for example in examples]
    kl_texts = [example["kl_text"] for example in examples] if examples[0].get("kl_text", None) else None

    batch = processor(
        main_texts=main_texts,
        auxiliary_inputs=auxiliary_inputs,
        kl_texts=kl_texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length=max_length,
    )

    batch["labels"] = create_assistant_labels(
        input_ids=batch["input_ids"],
        tokenizer=processor.main_tokenizer,
        assistant_start_marker=response_template,
        im_end_marker=response_end_marker,
    )

    if debug:
        num_valid = (batch["labels"] != -100).sum().item()
        print(f"[DEBUG] valid label tokens: {num_valid}")

    if kl_texts:
        batch["kl_input"]["labels"] = create_assistant_labels(
            input_ids=batch["kl_input"]["input_ids"],
            tokenizer=processor.main_tokenizer,
            assistant_start_marker=response_template,
            im_end_marker=response_end_marker,
        )

    return batch
