"""Label masking helpers for assistant-only language-model training."""

from __future__ import annotations

from typing import List

import torch


def find_assistant_sections(
    sequence: torch.Tensor,
    start_tokens: torch.Tensor,
    end_tokens: torch.Tensor,
) -> List[tuple[int, int]]:
    """Find assistant response spans in a token sequence."""

    seq_len = sequence.size(0)
    start_len = start_tokens.size(0)
    end_len = end_tokens.size(0)
    sections = []

    if seq_len < max(start_len, end_len):
        return sections

    start_positions = []
    for i in range(seq_len - start_len + 1):
        if torch.equal(sequence[i : i + start_len], start_tokens):
            start_positions.append(i + start_len)

    end_positions = []
    for i in range(seq_len - end_len + 1):
        if torch.equal(sequence[i : i + end_len], end_tokens):
            end_positions.append(i)

    for start_pos in start_positions:
        valid_ends = [pos for pos in end_positions if pos > start_pos]
        if valid_ends:
            end_pos = min(valid_ends) + end_len
            if start_pos < end_pos:
                sections.append((start_pos, end_pos))
        else:
            sections.append((start_pos, seq_len))

    return sections


def create_assistant_labels(
    input_ids: torch.Tensor,
    tokenizer,
    assistant_start_marker: str = "<|im_start|>assistant\n",
    im_end_marker: str = "<|im_end|>",
) -> torch.Tensor:
    """Create labels where only assistant response tokens are trained."""

    if input_ids.dim() != 2:
        raise ValueError(f"Expected 2D input_ids tensor, got {input_ids.dim()}D")

    batch_size, seq_len = input_ids.shape
    labels = torch.full_like(input_ids, -100)
    assistant_start_marker = assistant_start_marker.replace("\\n", "\n")

    try:
        start_token_ids = tokenizer.encode(
            assistant_start_marker, add_special_tokens=False
        )
        end_token_ids = tokenizer.encode(im_end_marker, add_special_tokens=False)
    except Exception as exc:
        raise ValueError(f"Failed to encode markers: {exc}") from exc

    if not start_token_ids or not end_token_ids:
        raise ValueError("Encoded markers cannot be empty")

    start_tokens = torch.tensor(
        start_token_ids, device=input_ids.device, dtype=input_ids.dtype
    )
    end_tokens = torch.tensor(
        end_token_ids, device=input_ids.device, dtype=input_ids.dtype
    )

    for batch_idx in range(batch_size):
        sequence = input_ids[batch_idx]
        sections = find_assistant_sections(sequence, start_tokens, end_tokens)
        for start_pos, end_pos in sections:
            start_pos = max(0, min(start_pos, seq_len))
            end_pos = max(start_pos, min(end_pos, seq_len))
            if start_pos < end_pos:
                labels[batch_idx, start_pos:end_pos] = sequence[start_pos:end_pos]

    if getattr(tokenizer, "pad_token_id", None) is not None:
        labels[input_ids == tokenizer.pad_token_id] = -100

    return labels

