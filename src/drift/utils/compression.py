"""Compression-token count helpers."""

from __future__ import annotations


def bucketed_compression_tokens(
    token_length: int,
    compress_ratio: int,
    *,
    mode: str = "small_threshold",
    max_bucket: int = 8192,
) -> int:
    """Compute the number of `<|CPS|>` tokens for a sequence.

    This mirrors the legacy `small_threshold` behavior used across training and
    inference. The helper is deliberately small so future migrations can replace
    duplicated inline logic with one shared implementation.
    """

    if token_length < 0:
        raise ValueError("token_length must be non-negative")
    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")

    if mode == "fix":
        return max(token_length // compress_ratio, 1)

    if mode != "small_threshold":
        raise ValueError(f"Unsupported compression mode: {mode}")

    bucket = 128
    while bucket < token_length and bucket < max_bucket:
        bucket *= 2

    if token_length > max_bucket:
        return max(token_length // compress_ratio, 1)

    return max(bucket // compress_ratio, 1)

