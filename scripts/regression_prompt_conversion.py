#!/usr/bin/env python3
"""Lightweight regression checks for DRIFT prompt conversion.

This script intentionally avoids torch, transformers, datasets, and model
loading. It uses a fake tokenizer to validate that the public preprocessing
functions produce the expected fields and key prompt content.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from drift.data.preprocessing import (  # noqa: E402
    convert_to_messages_and_apply_template_lfrp,
    convert_to_messages_and_apply_template_qaft_dc,
    convert_to_messages_and_apply_template_qaft_qa,
    convert_to_messages_and_apply_template_multi_sft,
)
from drift.utils.constants import COMPRESSION_TOKEN  # noqa: E402


class FakeTokenizer:
    """Small tokenizer stub with the methods used by preprocessing."""

    def encode(self, text, add_special_tokens=True):
        _ = add_special_tokens
        return text.split()

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        _ = tokenize, kwargs
        return "".join(f"<{message['role']}>{message['content']}" for message in messages)

    def __call__(self, texts, **kwargs):
        return {"texts": texts, "kwargs": kwargs}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    main_tokenizer = FakeTokenizer()
    aux_tokenizer = FakeTokenizer()

    lfrp = convert_to_messages_and_apply_template_lfrp(
        {"context": "alpha beta gamma"},
        main_tokenizer,
        aux_tokenizer,
        compress_ratio=8,
        compress_mode="small_threshold",
    )
    assert_true(set(lfrp) == {"text", "auxiliary_input"}, "LFRP fields changed")
    assert_true(COMPRESSION_TOKEN in lfrp["text"], "LFRP main prompt missing CPS")
    assert_true(COMPRESSION_TOKEN in lfrp["auxiliary_input"], "LFRP aux prompt missing CPS")

    qaft_dc = convert_to_messages_and_apply_template_qaft_dc(
        {"Document": "doc words", "Question": "What?", "Evidence": "gold evidence"},
        main_tokenizer,
        aux_tokenizer,
        compress_ratio=32,
        compress_mode="small_threshold",
    )
    assert_true(set(qaft_dc) == {"text", "auxiliary_input"}, "QAFT-DC fields changed")
    assert_true("gold evidence" in qaft_dc["text"], "QAFT-DC target evidence missing")
    assert_true("What?" in qaft_dc["auxiliary_input"], "QAFT-DC aux prompt missing question")

    qaft_qa = convert_to_messages_and_apply_template_qaft_qa(
        {
            "Document": "doc words",
            "Question": "What?",
            "Answer": "answer",
            "Evidence": "gold evidence",
        },
        main_tokenizer,
        aux_tokenizer,
        compress_ratio=32,
        compress_mode="small_threshold",
    )
    assert_true(set(qaft_qa) == {"text", "auxiliary_input", "kl_text"}, "QAFT-QA fields changed")
    assert_true(COMPRESSION_TOKEN in qaft_qa["text"], "QAFT-QA main prompt missing CPS")
    assert_true("gold evidence" in qaft_qa["kl_text"], "QAFT-QA KL prompt missing evidence")
    assert_true(COMPRESSION_TOKEN not in qaft_qa["kl_text"], "QAFT-QA KL prompt should not include CPS")

    multi = convert_to_messages_and_apply_template_multi_sft(
        {
            "Document": ["chunk one", "chunk two"],
            "Question": "Choose?",
            "Answer": "B",
            "Evidence": "gold evidence",
            "answer_prefix": "Answer with a letter: ",
        },
        main_tokenizer,
        aux_tokenizer,
        compress_ratio=32,
        compress_mode="small_threshold",
        answer_prefix="SHOULD_NOT_WIN: ",
    )
    assert_true(set(multi) == {"text", "auxiliary_input", "kl_text"}, "multi-SFT fields changed")
    assert_true("Answer with a letter: " in multi["text"], "Dataset answer_prefix did not win")
    assert_true("SHOULD_NOT_WIN: " not in multi["text"], "Function answer_prefix overrode dataset value")
    assert_true(len(multi["auxiliary_input"]) == 2, "multi-SFT auxiliary chunk count changed")

    print("Prompt conversion regression checks passed.")


if __name__ == "__main__":
    main()

