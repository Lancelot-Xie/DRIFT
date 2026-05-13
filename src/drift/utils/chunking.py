"""Chunking utilities."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter


class EnhancedRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """Recursive splitter with robust tokenizer-length handling."""

    @classmethod
    def from_huggingface_tokenizer(
        cls,
        tokenizer: Any,
        max_length: int = 131072,
        **kwargs: Any,
    ) -> TextSplitter:
        try:
            from transformers import PreTrainedTokenizerBase
        except ImportError as exc:
            raise ValueError(
                "Could not import transformers. Please install transformers."
            ) from exc

        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(
                "Tokenizer received was not an instance of PreTrainedTokenizerBase"
            )

        def tokenizer_length(text: str) -> int:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                total_tokens = 0
                for i in range(0, len(text), max_length):
                    total_tokens += len(tokenizer.encode(text[i : i + max_length]))
                return total_tokens

        return cls(length_function=tokenizer_length, **kwargs)

