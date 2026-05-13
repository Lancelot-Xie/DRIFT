"""Tokenizer helpers."""

from __future__ import annotations

from typing import Any

from drift.utils.constants import COMPRESSION_TOKEN


def ensure_compression_token(
    tokenizer: Any,
    model: Any | None = None,
    *,
    compression_token: str = COMPRESSION_TOKEN,
) -> int:
    """Ensure a tokenizer contains the DRIFT compression token.

    If a model is provided and the tokenizer grows, the model embeddings are
    resized to match the new tokenizer length. This mirrors the legacy behavior
    used during both training and inference.
    """

    old_size = len(tokenizer)
    vocab = tokenizer.get_vocab()

    if compression_token not in vocab:
        tokenizer.add_tokens([compression_token])
        if model is not None and len(tokenizer) > old_size:
            model.resize_token_embeddings(len(tokenizer))

    return tokenizer.convert_tokens_to_ids(compression_token)


def ensure_pad_token(tokenizer: Any, *, pad_token: str = "<pad>") -> None:
    """Ensure a tokenizer has a pad token.

    This is intentionally conservative. Model-family-specific pad-token choices
    will be handled in the migrated training entrypoint.
    """

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": pad_token})
        tokenizer.pad_token = pad_token

