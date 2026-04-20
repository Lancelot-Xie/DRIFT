from typing import Any, Callable, Optional, Sequence, TypeVar, Union
import inspect
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer
from drift.utils.prompts import system_prompt_auxiliary, user_prompt_auxiliary, assistant_prompt_auxiliary, system_prompt_main, user_prompt_main, COMPRESSION_TOKEN, system_prompt_auxiliary_pretrain, user_prompt_auxiliary_pretrain, user_prompt_main_pretrain, assistant_prompt_auxiliary_pretrain, system_prompt_auxiliary_pretrain_second, user_prompt_auxiliary_pretrain_second, assistant_prompt_auxiliary_pretrain_second, user_prompt_main_pretrain_second, user_prompt_main_sft, user_prompt_main_kl, user_prompt_main_multi_sft
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter

import os
os.environ["HF_DATASETS_CACHE"] = "/fs-computility/ai-shen/xiewenxuan/.cache/huggingface/datasets"

class EnhancedRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """增强型文本分割器，解决长文本超出tokenizer最大限制的问题"""
    
    @classmethod
    def from_huggingface_tokenizer(cls, tokenizer: Any, max_length: int = 131072, **kwargs: Any) -> TextSplitter:
        """
        使用HuggingFace tokenizer计算长度，并处理超长文本
        
        参数:
            tokenizer: HuggingFace tokenizer
            max_length: tokenizer能处理的最大字符串长度，默认524288
            **kwargs: 传递给父类的其他参数
        """
        try:
            from transformers import PreTrainedTokenizerBase

            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError(
                    "Tokenizer received was not an instance of PreTrainedTokenizerBase"
                )

            def _enhanced_huggingface_tokenizer_length(text: str) -> int:
                # 如果文本长度超过最大限制，分段处理
                try:
                    # 文本长度在限制范围内，直接计算
                    return len(tokenizer.encode(text))
                except:
                    # 分段并计算总token数
                    total_tokens = 0
                    for i in range(0, len(text), max_length):
                        segment = text[i:i + max_length]
                        total_tokens += len(tokenizer.encode(segment))
                    
                    return total_tokens

        except ImportError:
            raise ValueError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            )
        
        return cls(length_function=_enhanced_huggingface_tokenizer_length, **kwargs)


def get_token_length(text, tokenizer):
    # 将文本转换为token ids
    token_ids = tokenizer.encode(text)
    
    # 返回token ids的长度
    return len(token_ids)



def convert_to_messages_and_apply_template_sft(
    example: dict[str, list[dict[str, str]]],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer
) -> dict[str, str]:
    if not example.get('instruction_main', None):
        instruction_main = system_prompt_main
    else:
        instruction_main = example.get('instruction_main')
    if not example.get('instruction_aux', None):
        instruction_aux = system_prompt_auxiliary
    else:
        instruction_aux = example.get('instruction_aux')
    
    auxiliary_input = []
    for i in range(len(example['auxiliary_input_list'])):
        messages_aux = [{"role": "system", "content": instruction_aux}, {"role": "user", "content": user_prompt_auxiliary.format(context=example['auxiliary_input_list'][i], question=example['input'])}, {"role": "assistant", "content": assistant_prompt_auxiliary.format(judgement='Yes')}]
        temp_aux = aux_tokenizer.apply_chat_template(messages_aux, tokenize=False)
        auxiliary_input.append(temp_aux)
    num_compression_tokens = 1
    background = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)
    messages_main = [{"role": "system", "content": instruction_main}, {"role": "user", "content": user_prompt_main.format(background=background, question=example['input'])}, {"role": "assistant", "content": example['output']}]
    main_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False)
    tokenized_context = aux_tokenizer(
        auxiliary_input,
        padding=True,
        return_tensors="pt"
    )

    output = {}
    output["text"] = main_text 
    print(main_text)
    output["auxiliary_input"] = tokenized_context
    output["initial_aux"] = auxiliary_input
    
    return output


def convert_to_messages_and_apply_template_pretrain_first(
    example: dict[str, list[dict[str, str]]],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
    compress_ratio: int = 8,
    compress_mode: str = 'fix',
    upper_bound: int = 1024
) -> dict[str, str]:
    if not example.get('instruction_aux', None):
        instruction_aux = system_prompt_auxiliary_pretrain
    else:
        instruction_aux = example.get('instruction_aux')
    
    text = example["context"].strip()
    length = get_token_length(text, aux_tokenizer)
    compress_ratio = compress_ratio
    if compress_mode == 'fix':
        num_compression_tokens = max(length // compress_ratio, 1)
    elif compress_mode == 'small_threshold':
        if length <= 128:
            num_compression_tokens = max(128 // compress_ratio, 1)
        elif length <= 256:
            num_compression_tokens = max(256 // compress_ratio, 1)
        elif length <= 512:
            num_compression_tokens = max(512 // compress_ratio, 1)
        elif length <= 1024:
            num_compression_tokens = max(1024 // compress_ratio, 1)
        else:
            print("Warning: length > 1024, using default compress ratio")
            num_compression_tokens = max(length // compress_ratio, 1)
    CPS_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)
    messages_aux = [{"role": "system", "content": instruction_aux.format(num=num_compression_tokens)}, {"role": "user", "content": user_prompt_auxiliary_pretrain.format(context=text)}, {"role": "assistant", "content": assistant_prompt_auxiliary_pretrain.format(CPS_tokens=CPS_tokens)}]
    temp_aux = aux_tokenizer.apply_chat_template(messages_aux, tokenize=False)
    CPS_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)
    messages_main = [{"role": "user", "content": user_prompt_main_pretrain.format(compressed_information=CPS_tokens)}, {"role": "assistant", "content": text}]
    main_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False)
    """
    tokenized_aux_context = aux_tokenizer(
        temp_aux,
        padding=True,
        return_tensors="pt"
    )
    """
    output = {}
    output["text"] = main_text 
    # output["auxiliary_input"] = tokenized_aux_context
    output["auxiliary_input"] = temp_aux
    
    return output

def convert_to_messages_and_apply_template_pretrain_second(
    example: dict[str, list[dict[str, str]]],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
    compress_ratio: int = 8,
    compress_mode: str = 'fix'
) -> dict[str, str]:
    if not example.get('instruction_aux', None):
        instruction_aux = system_prompt_auxiliary_pretrain_second
    else:
        instruction_aux = example.get('instruction_aux')
    
    text = example["Document"]
    question = example["Question"]
    evidence = example["Evidence"]
    length = get_token_length(text, aux_tokenizer)
    compress_ratio = compress_ratio
    if compress_mode == 'fix':
        num_compression_tokens = max(length // compress_ratio, 1)
    elif compress_mode == 'small_threshold':
        if length <= 2048: 
            num_compression_tokens = max(2048 // compress_ratio, 1)
        elif length <= 4096: 
            num_compression_tokens = max(4096 // compress_ratio, 1)
        elif length <= 8192: 
            num_compression_tokens = max(8192 // compress_ratio, 1)
    CPS_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)
    messages_aux = [{"role": "system", "content": instruction_aux.format(num=num_compression_tokens)}, {"role": "user", "content": user_prompt_auxiliary_pretrain_second.format(document=text,question=question)}, {"role": "assistant", "content": assistant_prompt_auxiliary_pretrain_second.format(CPS_tokens=CPS_tokens)}]
    temp_aux = aux_tokenizer.apply_chat_template(messages_aux, tokenize=False, enable_thinking=False)
    CPS_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)
    messages_main = [{"role": "user", "content": user_prompt_main_pretrain_second.format(compressed_information=CPS_tokens)}, {"role": "assistant", "content": evidence}]
    main_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False, enable_thinking=False)
    """
    tokenized_aux_context = aux_tokenizer(
        temp_aux,
        padding=True,
        return_tensors="pt"
    )
    """
    output = {}
    output["text"] = main_text 
    # output["auxiliary_input"] = tokenized_aux_context
    output["auxiliary_input"] = temp_aux

    return output

def convert_to_messages_and_apply_template_simple_sft(
    example: dict[str, list[dict[str, str]]],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
    compress_ratio: int = 8,
    compress_mode: str = 'fix'
) -> dict[str, str]:
    if not example.get('instruction_aux', None):
        instruction_aux = system_prompt_auxiliary_pretrain_second
    else:
        instruction_aux = example.get('instruction_aux')
    
    text = example["Document"]
    question = example["Question"]
    answer = example["Answer"]
    evidence = example["Evidence"]
    
    if answer == None:
        print("救命啊")
    length = get_token_length(text, aux_tokenizer)
    compress_ratio = compress_ratio
    if compress_mode == 'fix':
        num_compression_tokens = max(length // compress_ratio, 1)
    elif compress_mode == 'small_threshold':
        if length <= 128:
            num_compression_tokens = max(128 // compress_ratio, 1)
        elif length <= 256:
            num_compression_tokens = max(256 // compress_ratio, 1)
        elif length <= 512:
            num_compression_tokens = max(512 // compress_ratio, 1)
        elif length <= 1024:
            num_compression_tokens = max(1024 // compress_ratio, 1)
        elif length <= 2048: 
            num_compression_tokens = max(2048 // compress_ratio, 1)
        elif length <= 4096: 
            num_compression_tokens = max(4096 // compress_ratio, 1)
        elif length <= 8192: 
            num_compression_tokens = max(8192 // compress_ratio, 1)
    CPS_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)
    messages_aux = [{"role": "system", "content": instruction_aux.format(num=num_compression_tokens)}, {"role": "user", "content": user_prompt_auxiliary_pretrain_second.format(document=text,question=question)}, {"role": "assistant", "content": assistant_prompt_auxiliary_pretrain_second.format(CPS_tokens=CPS_tokens)}]
    temp_aux = aux_tokenizer.apply_chat_template(messages_aux, tokenize=False, enable_thinking=False)
    CPS_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)
    messages_main = [{"role": "user", "content": user_prompt_main_sft.format(compressed_information=CPS_tokens, question=question)}, {"role": "assistant", "content": answer}]
    main_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False, enable_thinking=False)
    messages_kl = [{"role": "user", "content": user_prompt_main_kl.format(context=evidence, question=question)}, {"role": "assistant", "content": answer}]
    kl_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False, enable_thinking=False)
    """
    tokenized_aux_context = aux_tokenizer(
        temp_aux,
        padding=True,
        return_tensors="pt"
    )
    """
    output = {}
    output["text"] = main_text 
    # output["auxiliary_input"] = tokenized_aux_context
    output["auxiliary_input"] = temp_aux
    output["kl_text"] = kl_text
    
    return output

def convert_to_messages_and_apply_template_multi_sft(
    example: dict[str, list[dict[str, str]]],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
    compress_ratio: int = 8,
    compress_mode: str = 'fix',
    chunker=None
) -> dict[str, str]:
    if not example.get('instruction_aux', None):
        instruction_aux = system_prompt_auxiliary_pretrain_second
    else:
        instruction_aux = example.get('instruction_aux')
    
    texts = example["Document"]
    question = example["Question"]
    answer = example["Answer"]
    evidence = example["Evidence"]
    
    if answer == None:
        print("救命啊")
    if isinstance(texts, list):
        text_list = texts
    else:
        text_list = chunker.split_text(texts)
    temp_aux_list = []
    num_main_compression_tokens = []
    for text in text_list:
        length = get_token_length(text, aux_tokenizer)
        compress_ratio = compress_ratio
        if compress_mode == 'fix':
            num_compression_tokens = max(length // compress_ratio, 1)
        elif compress_mode == 'small_threshold':
            if length <= 128:
                num_compression_tokens = max(128 // compress_ratio, 1)
            elif length <= 256:
                num_compression_tokens = max(256 // compress_ratio, 1)
            elif length <= 512:
                num_compression_tokens = max(512 // compress_ratio, 1)
            elif length <= 1024:
                num_compression_tokens = max(1024 // compress_ratio, 1)
            elif length <= 2048: 
                num_compression_tokens = max(2048 // compress_ratio, 1)
            elif length <= 4096: 
                num_compression_tokens = max(4096 // compress_ratio, 1)
            elif length <= 8192: 
                num_compression_tokens = max(8192 // compress_ratio, 1)
        CPS_tokens = " ".join([COMPRESSION_TOKEN] * num_compression_tokens)
        num_main_compression_tokens.append(num_compression_tokens)
        messages_aux = [{"role": "system", "content": instruction_aux.format(num=num_compression_tokens)}, {"role": "user", "content": user_prompt_auxiliary_pretrain_second.format(document=text,question=question)}, {"role": "assistant", "content": assistant_prompt_auxiliary_pretrain_second.format(CPS_tokens=CPS_tokens)}]
        temp_aux = aux_tokenizer.apply_chat_template(messages_aux, tokenize=False, enable_thinking=False)
        temp_aux_list.append(temp_aux)
    temp_main_CPS_tokens = []
    for num_single_compression_tokens in num_main_compression_tokens:
        temp_main_CPS_tokens.append(" ".join([COMPRESSION_TOKEN] * num_single_compression_tokens))
    main_CPS_tokens = "".join(
        f"<Document {i+1}>{seg}</Document {i+1}>\n\n" for i, seg in enumerate(temp_main_CPS_tokens)
    )
    messages_main = [{"role": "user", "content": user_prompt_main_multi_sft.format(num=len(temp_aux_list), compressed_information=main_CPS_tokens, question=question)}, {"role": "assistant", "content": answer}]
    main_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False, enable_thinking=False)
    messages_kl = [{"role": "user", "content": user_prompt_main_kl.format(context=evidence, question=question)}, {"role": "assistant", "content": answer}]
    kl_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False, enable_thinking=False)
    """
    tokenized_aux_context = aux_tokenizer(
        temp_aux,
        padding=True,
        return_tensors="pt"
    )
    """
    output = {}
    output["text"] = main_text 
    # output["auxiliary_input"] = tokenized_aux_context
    output["auxiliary_input"] = temp_aux_list
    output["kl_text"] = kl_text

    return output



def convert_to_messages_and_apply_template_test_normal_sft(
    example: dict[str, list[dict[str, str]]],
    main_tokenizer: PreTrainedTokenizer,
    aux_tokenizer: PreTrainedTokenizer,
    compress_ratio: int = 4
) -> dict[str, str]:

    messages_main = [{"role": "user", "content":  example['instruction']}, {"role": "assistant", "content":  example['output']}]
    main_text = main_tokenizer.apply_chat_template(messages_main, tokenize=False)
    output = {}
    output["text"] = main_text 
    output["auxiliary_input"] = ""
    
    return output

def debug_model_signature_and_dataset(model, dataset):
    """打印模型签名和数据集列名，帮助调试"""
    print("\n===== Model Signature and Dataset Columns Debug =====")
    
    # 检查模型forward方法的签名
    model_to_inspect = model
    if hasattr(model, "get_base_model"):
        model_to_inspect = model.get_base_model()
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model_to_inspect = model.base_model.model
        
    signature = inspect.signature(model_to_inspect.forward)
    signature_columns = list(signature.parameters.keys())
    
    print(f"Model forward method signature: {signature_columns}")
    print(f"Dataset columns: {dataset.column_names}")
    
    # 检查哪些列会被保留/移除
    kept_columns = [k for k in signature_columns if k in dataset.column_names]
    removed_columns = [k for k in dataset.column_names if k not in signature_columns]
    
    print(f"Columns that will be kept: {kept_columns}")
    print(f"Columns that will be removed: {removed_columns}")
    print("================================================\n")


def get_mom_signature_columns(model: nn.Module) -> list[str]:
    """
    获取MoM模型forward方法的签名参数列表
    
    参数:
        mom_model: MoM模型实例
        
    返回:
        列表: 包含模型forward方法所需的列名
    """
    import inspect
    
    # 获取模型实例
    model_to_inspect = model
    
    # 获取forward方法的签名
    signature = inspect.signature(model_to_inspect.forward)
    signature_columns = list(signature.parameters.keys())
    
    # 添加可能的标签列名
    signature_columns += ["label", "label_ids", "labels"]
    
    # 确保auxiliary_input也被保留
    if "auxiliary_input" not in signature_columns:
        signature_columns.append("auxiliary_input")
    
    return signature_columns


