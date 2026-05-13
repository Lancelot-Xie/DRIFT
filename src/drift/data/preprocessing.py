"""Dataset example conversion for DRIFT training stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from drift.data.templates import (
    ASSISTANT_PROMPT_AUXILIARY,
    ASSISTANT_PROMPT_AUXILIARY_PRETRAIN,
    ASSISTANT_PROMPT_AUXILIARY_PRETRAIN_SECOND,
    SYSTEM_PROMPT_AUXILIARY,
    SYSTEM_PROMPT_AUXILIARY_PRETRAIN,
    SYSTEM_PROMPT_AUXILIARY_PRETRAIN_SECOND,
    SYSTEM_PROMPT_MAIN,
    USER_PROMPT_AUXILIARY,
    USER_PROMPT_AUXILIARY_PRETRAIN,
    USER_PROMPT_AUXILIARY_PRETRAIN_SECOND,
    USER_PROMPT_MAIN,
    USER_PROMPT_MAIN_KL,
    USER_PROMPT_MAIN_MULTI_SFT,
    USER_PROMPT_MAIN_PRETRAIN,
    USER_PROMPT_MAIN_PRETRAIN_SECOND,
    USER_PROMPT_MAIN_SFT,
)
from drift.utils.constants import COMPRESSION_TOKEN

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
else:
    PreTrainedTokenizer = Any


def get_token_length(text: str, tokenizer: PreTrainedTokenizer) -> int:
    return len(tokenizer.encode(text))


def require_answer(example: dict[str, Any], stage_name: str) -> Any:
    answer = example.get("Answer")
    if answer is None:
        identifiers = {
            key: example[key]
            for key in ("id", "idx", "example_id", "Question")
            if key in example
        }
        raise ValueError(
            f"{stage_name} example is missing required field 'Answer'. "
            f"Available identifier fields: {identifiers or 'none'}"
        )
    return answer


def _compression_tokens_for_lfrp(
    length: int,
    compress_ratio: int,
    compress_mode: str,
) -> int:
    if compress_mode == "fix":
        return max(length // compress_ratio, 1)
    if compress_mode == "small_threshold":
        if length <= 128:
            return max(128 // compress_ratio, 1)
        if length <= 256:
            return max(256 // compress_ratio, 1)
        if length <= 512:
            return max(512 // compress_ratio, 1)
        if length <= 1024:
            return max(1024 // compress_ratio, 1)
        print("Warning: length > 1024, using default compress ratio")
        return max(length // compress_ratio, 1)
    raise ValueError(f"Unsupported compress_mode: {compress_mode}")


def _compression_tokens_for_qaft_dc(
    length: int,
    compress_ratio: int,
    compress_mode: str,
) -> int:
    if compress_mode == "fix":
        return max(length // compress_ratio, 1)
    if compress_mode == "small_threshold":
        if length <= 2048:
            return max(2048 // compress_ratio, 1)
        if length <= 4096:
            return max(4096 // compress_ratio, 1)
        if length <= 8192:
            return max(8192 // compress_ratio, 1)
        return max(length // compress_ratio, 1)
    raise ValueError(f"Unsupported compress_mode: {compress_mode}")


def _compression_tokens_for_qaft_qa(
    length: int,
    compress_ratio: int,
    compress_mode: str,
) -> int:
    if compress_mode == "fix":
        return max(length // compress_ratio, 1)
    if compress_mode == "small_threshold":
        if length <= 128:
            return max(128 // compress_ratio, 1)
        if length <= 256:
            return max(256 // compress_ratio, 1)
        if length <= 512:
            return max(512 // compress_ratio, 1)
        if length <= 1024:
            return max(1024 // compress_ratio, 1)
        if length <= 2048:
            return max(2048 // compress_ratio, 1)
        if length <= 4096:
            return max(4096 // compress_ratio, 1)
        if length <= 8192:
            return max(8192 // compress_ratio, 1)
        return max(length // compress_ratio, 1)
    raise ValueError(f"Unsupported compress_mode: {compress_mode}")


def convert_to_messages_and_apply_template_sft(
    example: dict[str, Any],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
) -> dict[str, Any]:
    instruction_main = example.get("instruction_main") or SYSTEM_PROMPT_MAIN
    instruction_aux = example.get("instruction_aux") or SYSTEM_PROMPT_AUXILIARY

    auxiliary_input = []
    for context in example["auxiliary_input_list"]:
        messages_aux = [
            {"role": "system", "content": instruction_aux},
            {
                "role": "user",
                "content": USER_PROMPT_AUXILIARY.format(
                    context=context, question=example["input"]
                ),
            },
            {
                "role": "assistant",
                "content": ASSISTANT_PROMPT_AUXILIARY.format(judgement="Yes"),
            },
        ]
        auxiliary_input.append(aux_tokenizer.apply_chat_template(messages_aux, tokenize=False))

    background = COMPRESSION_TOKEN
    messages_main = [
        {"role": "system", "content": instruction_main},
        {
            "role": "user",
            "content": USER_PROMPT_MAIN.format(
                background=background, question=example["input"]
            ),
        },
        {"role": "assistant", "content": example["output"]},
    ]
    main_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False)
    tokenized_context = aux_tokenizer(
        auxiliary_input,
        padding=True,
        return_tensors="pt",
    )

    return {
        "text": main_text,
        "auxiliary_input": tokenized_context,
        "initial_aux": auxiliary_input,
    }


def convert_to_messages_and_apply_template_lfrp(
    example: dict[str, Any],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
    compress_ratio: int = 8,
    compress_mode: str = "fix",
) -> dict[str, str]:
    instruction_aux = example.get("instruction_aux") or SYSTEM_PROMPT_AUXILIARY_PRETRAIN

    text = example["context"].strip()
    length = get_token_length(text, aux_tokenizer)
    num_compression_tokens = _compression_tokens_for_lfrp(
        length, compress_ratio, compress_mode
    )
    cps_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)

    messages_aux = [
        {"role": "system", "content": instruction_aux.format(num=num_compression_tokens)},
        {
            "role": "user",
            "content": USER_PROMPT_AUXILIARY_PRETRAIN.format(context=text),
        },
        {
            "role": "assistant",
            "content": ASSISTANT_PROMPT_AUXILIARY_PRETRAIN.format(
                CPS_tokens=cps_tokens
            ),
        },
    ]
    temp_aux = aux_tokenizer.apply_chat_template(messages_aux, tokenize=False)

    messages_main = [
        {
            "role": "user",
            "content": USER_PROMPT_MAIN_PRETRAIN.format(
                compressed_information=cps_tokens
            ),
        },
        {"role": "assistant", "content": text},
    ]
    main_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False)

    return {"text": main_text, "auxiliary_input": temp_aux}


def convert_to_messages_and_apply_template_qaft_dc(
    example: dict[str, Any],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
    compress_ratio: int = 8,
    compress_mode: str = "fix",
) -> dict[str, str]:
    instruction_aux = (
        example.get("instruction_aux") or SYSTEM_PROMPT_AUXILIARY_PRETRAIN_SECOND
    )

    text = example["Document"]
    question = example["Question"]
    evidence = example["Evidence"]
    length = get_token_length(text, aux_tokenizer)
    num_compression_tokens = _compression_tokens_for_qaft_dc(
        length, compress_ratio, compress_mode
    )
    cps_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)

    messages_aux = [
        {"role": "system", "content": instruction_aux.format(num=num_compression_tokens)},
        {
            "role": "user",
            "content": USER_PROMPT_AUXILIARY_PRETRAIN_SECOND.format(
                document=text, question=question
            ),
        },
        {
            "role": "assistant",
            "content": ASSISTANT_PROMPT_AUXILIARY_PRETRAIN_SECOND.format(
                CPS_tokens=cps_tokens
            ),
        },
    ]
    temp_aux = aux_tokenizer.apply_chat_template(
        messages_aux, tokenize=False, enable_thinking=False
    )

    messages_main = [
        {
            "role": "user",
            "content": USER_PROMPT_MAIN_PRETRAIN_SECOND.format(
                compressed_information=cps_tokens
            ),
        },
        {"role": "assistant", "content": evidence},
    ]
    main_text = main_tokenizer.apply_chat_template(
        messages_main, tokenize=False, enable_thinking=False
    )

    return {"text": main_text, "auxiliary_input": temp_aux}


def convert_to_messages_and_apply_template_qaft_qa(
    example: dict[str, Any],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
    compress_ratio: int = 8,
    compress_mode: str = "fix",
) -> dict[str, str]:
    instruction_aux = (
        example.get("instruction_aux") or SYSTEM_PROMPT_AUXILIARY_PRETRAIN_SECOND
    )

    text = example["Document"]
    question = example["Question"]
    answer = require_answer(example, "QAFT-QA")
    evidence = example["Evidence"]

    length = get_token_length(text, aux_tokenizer)
    num_compression_tokens = _compression_tokens_for_qaft_qa(
        length, compress_ratio, compress_mode
    )
    cps_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)

    messages_aux = [
        {"role": "system", "content": instruction_aux.format(num=num_compression_tokens)},
        {
            "role": "user",
            "content": USER_PROMPT_AUXILIARY_PRETRAIN_SECOND.format(
                document=text, question=question
            ),
        },
        {
            "role": "assistant",
            "content": ASSISTANT_PROMPT_AUXILIARY_PRETRAIN_SECOND.format(
                CPS_tokens=cps_tokens
            ),
        },
    ]
    temp_aux = aux_tokenizer.apply_chat_template(
        messages_aux, tokenize=False, enable_thinking=False
    )

    messages_main = [
        {
            "role": "user",
            "content": USER_PROMPT_MAIN_SFT.format(
                compressed_information=cps_tokens, question=question
            ),
        },
        {"role": "assistant", "content": answer},
    ]
    main_text = main_tokenizer.apply_chat_template(
        messages_main, tokenize=False, enable_thinking=False
    )

    messages_kl = [
        {
            "role": "user",
            "content": USER_PROMPT_MAIN_KL.format(context=evidence, question=question),
        },
        {"role": "assistant", "content": answer},
    ]
    kl_text = main_tokenizer.apply_chat_template(
        messages_kl, tokenize=False, enable_thinking=False
    )

    return {"text": main_text, "auxiliary_input": temp_aux, "kl_text": kl_text}


def convert_to_messages_and_apply_template_multi_sft(
    example: dict[str, Any],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
    compress_ratio: int = 8,
    compress_mode: str = "fix",
    chunker: Any = None,
    answer_prefix: str | None = None,
) -> dict[str, Any]:
    instruction_aux = (
        example.get("instruction_aux") or SYSTEM_PROMPT_AUXILIARY_PRETRAIN_SECOND
    )

    texts = example["Document"]
    question = example["Question"]
    answer = require_answer(example, "multi-SFT")
    evidence = example["Evidence"]
    answer_prefix = (
        example.get("answer_prefix")
        or answer_prefix
        or "Your answer of this question is: "
    )

    text_list = texts if isinstance(texts, list) else chunker.split_text(texts)
    temp_aux_list = []
    num_main_compression_tokens = []

    for text in text_list:
        length = get_token_length(text, aux_tokenizer)
        num_compression_tokens = _compression_tokens_for_qaft_qa(
            length, compress_ratio, compress_mode
        )
        cps_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)
        num_main_compression_tokens.append(num_compression_tokens)
        messages_aux = [
            {"role": "system", "content": instruction_aux.format(num=num_compression_tokens)},
            {
                "role": "user",
                "content": USER_PROMPT_AUXILIARY_PRETRAIN_SECOND.format(
                    document=text, question=question
                ),
            },
            {
                "role": "assistant",
                "content": ASSISTANT_PROMPT_AUXILIARY_PRETRAIN_SECOND.format(
                    CPS_tokens=cps_tokens
                ),
            },
        ]
        temp_aux_list.append(
            aux_tokenizer.apply_chat_template(
                messages_aux, tokenize=False, enable_thinking=False
            )
        )

    temp_main_cps_tokens = [
        " ".join([COMPRESSION_TOKEN] * count) for count in num_main_compression_tokens
    ]
    main_cps_tokens = "".join(
        f"<Document {i + 1}>{seg}</Document {i + 1}>\n\n"
        for i, seg in enumerate(temp_main_cps_tokens)
    )
    messages_main = [
        {
            "role": "user",
            "content": USER_PROMPT_MAIN_MULTI_SFT.format(
                num=len(temp_aux_list),
                compressed_information=main_cps_tokens,
                question=question,
                answer_prefix=answer_prefix,
            ),
        },
        {"role": "assistant", "content": answer},
    ]
    main_text = main_tokenizer.apply_chat_template(
        messages_main, tokenize=False, enable_thinking=False
    )

    messages_kl = [
        {
            "role": "user",
            "content": USER_PROMPT_MAIN_KL.format(context=evidence, question=question),
        },
        {"role": "assistant", "content": answer},
    ]
    kl_text = main_tokenizer.apply_chat_template(
        messages_kl, tokenize=False, enable_thinking=False
    )

    return {"text": main_text, "auxiliary_input": temp_aux_list, "kl_text": kl_text}


def convert_to_messages_and_apply_template_test_normal_sft(
    example: dict[str, Any],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
    compress_ratio: int = 4,
) -> dict[str, str]:
    _ = aux_tokenizer, compress_ratio
    messages_main = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    main_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False)
    return {"text": main_text, "auxiliary_input": ""}


# Legacy aliases.
convert_to_messages_and_apply_template_pretrain_first = (
    convert_to_messages_and_apply_template_lfrp
)
convert_to_messages_and_apply_template_pretrain_second = (
    convert_to_messages_and_apply_template_qaft_dc
)
convert_to_messages_and_apply_template_simple_sft = (
    convert_to_messages_and_apply_template_qaft_qa
)
