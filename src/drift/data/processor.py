"""Processor for paired DRIFT main/auxiliary tokenizers."""

from __future__ import annotations

from typing import List, Optional, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack


class DRIFTProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
    }


class DRIFTProcessor(ProcessorMixin):
    """Tokenize reasoning-model and knowledge-model inputs together."""

    attributes = ["aux_tokenizer", "main_tokenizer"]
    valid_kwargs = ["model", "chat_template"]
    aux_tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    main_tokenizer_class = (
        "Qwen2Tokenizer",
        "Qwen2TokenizerFast",
        "LlamaTokenizer",
        "LlamaTokenizerFast",
        "PreTrainedTokenizerFast",
    )

    def __init__(self, aux_tokenizer=None, main_tokenizer=None, chat_template=None):
        self.aux_tokenizer = aux_tokenizer
        self.main_tokenizer = main_tokenizer
        if chat_template is None and hasattr(self.main_tokenizer, "chat_template"):
            chat_template = self.main_tokenizer.chat_template
        super().__init__(aux_tokenizer, main_tokenizer, chat_template=chat_template)

    def __call__(
        self,
        auxiliary_inputs: Optional[Union[List[List[str]], List[str]]] = None,
        main_texts: Optional[Union[List[str], str]] = None,
        kl_texts: Optional[Union[List[str], str]] = None,
        return_tensors: str = "pt",
        max_length=32768,
        device: str = "cuda",
        **kwargs: Unpack[DRIFTProcessorKwargs],
    ) -> BatchFeature:
        _ = device
        padding_side = kwargs.get("padding_side", "left")

        if not isinstance(main_texts, list):
            main_texts = [main_texts]

        if kl_texts:
            if not isinstance(kl_texts, list):
                kl_texts = [kl_texts]
            kl_inputs = self.main_tokenizer(
                kl_texts,
                max_length=max_length,
                return_tensors=return_tensors,
                padding=True,
                padding_side=padding_side,
                truncation=True,
            )
        else:
            kl_inputs = None

        text_inputs = self.main_tokenizer(
            main_texts,
            max_length=max_length,
            return_tensors=return_tensors,
            padding=True,
            padding_side=padding_side,
            truncation=True,
        )

        if auxiliary_inputs is None or len(auxiliary_inputs) == 0:
            processed_auxiliary_inputs = {"auxiliary_input": []}
        elif not isinstance(auxiliary_inputs[0], str):
            aux_input_contexts = []
            for temp_aux in auxiliary_inputs:
                tokenized_aux_context = self.aux_tokenizer(
                    temp_aux,
                    padding=True,
                    padding_side=padding_side,
                    return_tensors="pt",
                )
                aux_input_contexts.append(tokenized_aux_context)
            processed_auxiliary_inputs = {"auxiliary_input": aux_input_contexts}
        else:
            aux_input_contexts = self.aux_tokenizer(
                auxiliary_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )
            processed_auxiliary_inputs = {"auxiliary_input": aux_input_contexts}

        return BatchFeature(
            data={
                **text_inputs,
                **processed_auxiliary_inputs,
                "kl_input": kl_inputs,
            }
        )

