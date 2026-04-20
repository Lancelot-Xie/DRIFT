# mom_data_module.py
import torch
import inspect
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional, List, Any, Union, Dict
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
import pytorch_lightning as pl
from functools import partial
from drift.utils.data_utils import convert_to_messages_and_apply_template_sft, convert_to_messages_and_apply_template_pretrain_first, convert_to_messages_and_apply_template_pretrain_second, convert_to_messages_and_apply_template_test_normal_sft, convert_to_messages_and_apply_template_simple_sft, convert_to_messages_and_apply_template_multi_sft, EnhancedRecursiveCharacterTextSplitter  # 导入你原有的工具函数
from transformers.processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)

from trl.trainer.utils import DataCollatorForCompletionOnlyLM

import os 
# os.environ["HF_DATASETS_CACHE"] = "/fs-computility/ai-shen/xiewenxuan/.cache/huggingface/datasets"
def tokenize(example, processing_class, dataset_text_field):
    return processing_class(example[dataset_text_field])

def process_auxiliary_input(example, chunker):
    if 'context' in example:
        auxiliary_input_list = chunker.split_text(example['context'])
        return {"auxiliary_input_list": auxiliary_input_list}
    return {"auxiliary_input_list": []}

class DLProcessorKwargs(ProcessingKwargs, total=False):
    """Processing keyword arguments for the DL processor"""
    _defaults = {
        "text_kwargs": {
            "padding": True,
        },
    }


def pack_examples(examples, seq_length=2048):
    """
    打包 input_ids，让多条短句拼成一条长的定长序列
    输入: {'input_ids': [[...], [...], ...]}
    输出: {'input_ids': [[packed_ids_1], [packed_ids_2], ...]}
    """
    # 取出所有 input_ids，拼成一个大长串
    all_ids = []
    for ids in examples["input_ids"]:
        all_ids.extend(ids)
    
    # 按 seq_length 切成一批批
    packed = []
    for i in range(0, len(all_ids), seq_length):
        chunk = all_ids[i:i+seq_length]
        if len(chunk) == seq_length:
            packed.append(chunk)
    return {"input_ids": packed}

class DLProcessor(ProcessorMixin):
    
    attributes = ["aux_tokenizer", "main_tokenizer"]
    valid_kwargs = ["model", "chat_template"]
    aux_tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
    main_tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast", "LlamaTokenizer", "LlamaTokenizerFast","PreTrainedTokenizerFast")

    def __init__(self, aux_tokenizer=None, main_tokenizer=None, chat_template=None):
        self.aux_tokenizer = aux_tokenizer
        self.main_tokenizer = main_tokenizer
        
        # 使用 main_tokenizer 而不是未定义的 tokenizer
        if chat_template is None and hasattr(self.main_tokenizer, "chat_template"):
            chat_template = self.main_tokenizer.chat_template
        
        # 修复 super().__init__ 调用
        super().__init__(aux_tokenizer, main_tokenizer, chat_template=chat_template)
        
    def __call__(
        self,
        auxiliary_inputs: Optional[Union[List[List[str]], List[str]]] = None,
        main_texts: Optional[Union[List[str],str]] = None,
        kl_texts: Optional[Union[List[str],str]] = None,
        return_tensors: str = "pt",
        max_length = 32768,
        device: str = "cuda",
        **kwargs: Unpack[DLProcessorKwargs],
     ) -> BatchFeature:
        # 提取 padding_side 参数，默认为 "left"
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
                truncation=True  
            )
        else:
            kl_inputs = None
        processed_kl_inputs = {"kl_input": kl_inputs}
 
        text_inputs = self.main_tokenizer(
            main_texts, 
            max_length=max_length,
            return_tensors=return_tensors,
            padding=True,
            padding_side=padding_side,
            truncation=True
        )
        if auxiliary_inputs is None or len(auxiliary_inputs) == 0:
            processed_auxiliary_inputs = {"auxiliary_input": []}
        elif not isinstance(auxiliary_inputs[0], str):
            aux_input_contexts = []
            for i in range(len(auxiliary_inputs)):
                temp_aux = auxiliary_inputs[i]
                tokenized_aux_context = self.aux_tokenizer(
                    temp_aux,
                    padding=True,
                    padding_side=padding_side,
                    return_tensors="pt"
                )
                aux_input_contexts.append(tokenized_aux_context)
            processed_auxiliary_inputs = {"auxiliary_input": aux_input_contexts}
        else:
            aux_input_contexts = self.aux_tokenizer(
                auxiliary_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt"       
            )
            processed_auxiliary_inputs = {"auxiliary_input": aux_input_contexts}
        
        return BatchFeature(data={**text_inputs, **processed_auxiliary_inputs, **processed_kl_inputs})

        
def _find_assistant_sections_vectorized(
    sequence: torch.Tensor,
    start_tokens: torch.Tensor,
    end_tokens: torch.Tensor
) -> List[tuple]:
    """
    使用向量化操作找到assistant sections
    
    Args:
        sequence: 单个序列的token ids [seq_len]
        start_tokens: assistant开始标记的token ids
        end_tokens: 结束标记的token ids
    
    Returns:
        List of (start_pos, end_pos) tuples
    """
    seq_len = sequence.size(0)
    start_len = start_tokens.size(0)
    end_len = end_tokens.size(0)
    
    sections = []
    
    # 如果序列太短，直接返回空列表
    if seq_len < max(start_len, end_len):
        return sections
    
    # 使用卷积的思想进行模式匹配 - 找开始位置
    start_positions = []
    if seq_len >= start_len:
        for i in range(seq_len - start_len + 1):
            if torch.equal(sequence[i:i + start_len], start_tokens):
                start_positions.append(i + start_len)  # 存储开始内容的位置
    
    # 找结束位置
    end_positions = []
    if seq_len >= end_len:
        for i in range(seq_len - end_len + 1):
            if torch.equal(sequence[i:i + end_len], end_tokens):
                end_positions.append(i)  # 存储结束标记开始的位置
    
    # 匹配开始和结束位置
    for start_pos in start_positions:
        valid_ends = [pos for pos in end_positions if pos > start_pos]
        if valid_ends:
            end_pos = min(valid_ends) + end_len  # 修改这里：包含整个终止符
            if start_pos < end_pos:
                sections.append((start_pos, end_pos))
        else:
            # 没有结束标记，延续到序列末尾
            sections.append((start_pos, seq_len))
    
    return sections

def create_assistant_labels(
    input_ids: torch.Tensor,
    tokenizer,
    assistant_start_marker: str = "<|im_start|>assistant\n",
    im_end_marker: str = "<|im_end|>",
) -> torch.Tensor:
    """
    优化版本的标签创建函数
    
    Args:
        input_ids: Token IDs tensor of shape [batch_size, seq_len]
        tokenizer: The tokenizer used for encoding
        assistant_start_marker: Start marker for assistant responses
        im_end_marker: End marker for responses
        
    Returns:
        labels: Tensor of same shape as input_ids, with -100 for ignored positions
    """
    # 输入验证
    if input_ids.dim() != 2:
        raise ValueError(f"Expected 2D input_ids tensor, got {input_ids.dim()}D")
    
    batch_size, seq_len = input_ids.shape
    
    # 创建标签张量，全部初始化为-100
    labels = torch.full_like(input_ids, -100)
    assistant_start_marker = assistant_start_marker.replace("\\n", "\n")
    # 预编码特殊标记（只编码一次，提高效率）
    try:
        start_token_ids = tokenizer.encode(assistant_start_marker, add_special_tokens=False)
        end_token_ids = tokenizer.encode(im_end_marker, add_special_tokens=False)
    except Exception as e:
        raise ValueError(f"Failed to encode markers: {e}")

    

    if not start_token_ids or not end_token_ids:
        raise ValueError("Encoded markers cannot be empty")
    
    # 转换为tensor，移动到正确的设备
    start_tokens = torch.tensor(start_token_ids, device=input_ids.device, dtype=input_ids.dtype)
    end_tokens = torch.tensor(end_token_ids, device=input_ids.device, dtype=input_ids.dtype)

    # 批量处理每个序列
    for batch_idx in range(batch_size):
        sequence = input_ids[batch_idx]

        # 找到所有assistant sections
        sections = _find_assistant_sections_vectorized(sequence, start_tokens, end_tokens)
        
        # 为每个section设置标签
        for start_pos, end_pos in sections:
            # 边界检查
            start_pos = max(0, min(start_pos, seq_len))
            end_pos = max(start_pos, min(end_pos, seq_len))
            
            if start_pos < end_pos:
                labels[batch_idx, start_pos:end_pos] = sequence[start_pos:end_pos]
    
    # 处理padding tokens
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        padding_mask = (input_ids == tokenizer.pad_token_id)
        labels[padding_mask] = -100
    
    return labels


def mom_collate_fn(
    examples: List[Dict],
    processor: DLProcessor,
    max_length: int,
    response_template: str = "<|im_start|>assistant\n",
    response_end_marker: str = "<|im_end|>"
):
    auxiliary_inputs = [example["auxiliary_input"] for example in examples]
    main_texts = [example["text"] for example in examples]
    kl_texts = [example["kl_text"] for example in examples] if examples[0].get("kl_text",None) else None
    batch = processor(
        main_texts=main_texts,
        auxiliary_inputs=auxiliary_inputs,
        kl_texts=kl_texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length=max_length
    )
    if False:  # 生产环境可以改为 if self.global_rank == 0 之类的
        tokenizer = processor.main_tokenizer
        first_ids = batch["input_ids"][0]
        
        # 1. 获得每个 token 独立解码后的列表
        # convert_ids_to_tokens 会保留 TikToken 的特殊格式（如 Ġ 代表空格）
        tokens = tokenizer.convert_ids_to_tokens(first_ids)
        
        print("\n" + "="*50)
        print("【DRIFT 调试：Token 序列分析】")
        
        # 2. 查找并打印 Assistant Marker 附近的 Token
        # 尝试找到 "assistant" 所在的索引
        try:
            # 在 Llama 3 中，assistant 通常是一个独立 token
            # 如果找不到，可以遍历 tokens 打印前 50 个
            indices = [i for i, t in enumerate(tokens) if "assistant" in t.lower()]
            
            if indices:
                target_idx = indices[0]
                start_view = max(0, target_idx - 5)
                end_view = min(len(tokens), target_idx + 10)
                
                print(f"检测到 'assistant' 在索引 {target_idx} 附近：")
                print("-" * 30)
                for idx in range(start_view, end_view):
                    token_str = tokens[idx]
                    token_id = first_ids[idx].item()
                    # 标记出当前的 token，方便观察换行符
                    marker = " <--- START?" if idx == target_idx else ""
                    print(f"Index {idx:4d} | ID: {token_id:6d} | Token: [{token_str}] {marker}")
                print("-" * 30)
            else:
                print("警告：在 Token 序列中未搜索到包含 'assistant' 的标记。")
                print(f"前 30 个 Token 如下：{tokens[:30]}")
                
        except Exception as e:
            print(f"调试打印失败: {e}")
        print("="*50 + "\n")
    # --- 调试代码结束 ---
        # Create labels using the extracted function
    batch["labels"] = create_assistant_labels(
        input_ids=batch["input_ids"],
        tokenizer=processor.main_tokenizer,
        assistant_start_marker=response_template,
        im_end_marker=response_end_marker,
    )
    labels = batch["labels"]
    input_ids = batch["input_ids"]

    # 1️⃣ 看 label 是否全是 -100
    num_valid = (labels != -100).sum().item()
    print(f"[DEBUG] valid label tokens: {num_valid}")
    """
    # 2️⃣ 如果是第一个 batch，直接反解看看
    if num_valid > 0:
        idx = 0
        decoded_input = processor.main_tokenizer.decode(
            input_ids[idx].tolist(),
            skip_special_tokens=False
        )
        decoded_label = processor.main_tokenizer.decode(
            labels[idx][labels[idx] != -100].tolist(),
            skip_special_tokens=False
        )
        print("===== INPUT TEXT =====")
        print(decoded_input)
        print("===== LABEL TEXT =====")
        print(decoded_label)
    else:
        print("[DEBUG] No valid labels found in this batch")
    """
    if kl_texts:
        batch["kl_input"]["labels"] = create_assistant_labels(
            input_ids=batch["kl_input"]["input_ids"],
            tokenizer=processor.main_tokenizer,
            assistant_start_marker=response_template,
            im_end_marker=response_end_marker          
        )

    return batch


        


class MoMDataCollator(DataCollatorForCompletionOnlyLM):
    def __call__(self, features):
        # 暂时保存auxiliary_input字段
        auxiliary_inputs = []
        for feature in features:
            if "auxiliary_input" in feature:
                auxiliary_inputs.append(feature.pop("auxiliary_input"))
            else:
                auxiliary_inputs.append(None)
        
        # 调用父类的torch_call方法处理剩余字段
        batch = self.torch_call(features)
        
        # 将auxiliary_input字段添加回batch
        batch["auxiliary_input"] = auxiliary_inputs
        return batch


class MoMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        main_tokenizer: AutoTokenizer = None,
        aux_tokenizer: AutoTokenizer = None,
        chunker = None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        max_length: int = 8192,
        num_workers: int = 4,
        response_template: str = "<|im_start|>assistant\n",
        response_end_marker: str = "<|im_end|>",
        dataset_text_field: str = "text",
        packing: bool = False,
        dataset_num_proc: int = 4,
        remove_unused_columns: bool = True,  # 添加这个参数
        signature_columns: Optional[List[str]] = None,  # 添加这个参数
        phase: str = 'sft',
        compress_ratio: int = 8,
        compress_mode: str = 'fix'

    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.main_tokenizer = main_tokenizer
        self.aux_tokenizer = aux_tokenizer
        self.chunker = chunker
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.response_template = response_template
        self.response_end_marker = response_end_marker
        self.dataset_text_field = dataset_text_field
        self.packing = packing
        self.dataset_num_proc = dataset_num_proc
        self.remove_unused_columns = remove_unused_columns  # 保存参数
        self._signature_columns = signature_columns  # 初始化签名列
        self.phase = phase
        self.compress_ratio = compress_ratio
        self.compress_mode = compress_mode
        self.chunker = EnhancedRecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.main_tokenizer,
            chunk_size=1024,
            chunk_overlap=100
        )
    
            
    def _remove_unused_columns(self, dataset: Dataset) -> Dataset:
        """移除模型forward方法不需要的列"""
        if not self.remove_unused_columns:
            return dataset
            
        signature_columns = self._signature_columns
        
        # 找出需要忽略的列
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            print(f"以下列不在模型的forward方法签名中，将被忽略: {', '.join(ignored_columns)}")
        
        # 保留的列
        columns = [k for k in signature_columns if k in dataset.column_names]
        if len(columns) == 0:
            raise ValueError(
                "数据集中没有列与模型的forward方法签名匹配。"
                f"以下列已被忽略: [{', '.join(ignored_columns)}]。"
                "请检查数据集和模型。您可能需要设置`remove_unused_columns=False`。"
            )
        
        # 移除不需要的列
        return dataset.remove_columns(ignored_columns)
    
    def prepare_dataset(self, dataset, is_train=True):
        """使用TRL的数据处理函数准备数据集"""
        if self.phase == 'sft':
            # 处理上下文     
            dataset = dataset.map(
                process_auxiliary_input,
                fn_kwargs={"chunker": self.chunker},
                remove_columns=["context"] if "context" in dataset.column_names else None,
                num_proc=self.dataset_num_proc
            )
            
            # 应用消息模板
            dataset = dataset.map(
                convert_to_messages_and_apply_template_sft,
                fn_kwargs={"main_tokenizer": self.main_tokenizer, "aux_tokenizer": self.aux_tokenizer},
                remove_columns=["input", "auxiliary_input_list"],
                num_proc=self.dataset_num_proc
            )



            dataset = dataset.map(
                tokenize,
                fn_kwargs={"processing_class": self.main_tokenizer, "dataset_text_field": self.dataset_text_field},
                num_proc=self.dataset_num_proc
            )
            
            # 如果需要打包数据
            if self.packing:
                dataset = dataset.select_columns("input_ids")
                dataset = dataset.map(
                    pack_examples, 
                    batched=True, 
                    fn_kwargs={"seq_length": self.max_length},
                    num_proc=self.dataset_num_proc
                )
            elif self.max_length is not None:
                # 截断数据
                def truncate(example, max_length):
                    return {key: example[key][:max_length] for key in ["input_ids", "attention_mask"]}
                
                dataset = dataset.map(
                    truncate,
                    fn_kwargs={"max_length": self.max_length},
                    num_proc=self.dataset_num_proc
                )
            # 在移除未使用的列之前打印第一条数据的'text'字段
            if len(dataset) > 0:
                first_example = dataset[0]
                if 'text' in first_example:
                    print(list(first_example.keys()))
                    print(f"第一条数据的text字段内容: {first_example['text'][:5000]}..." if len(first_example['text']) > 5000 else first_example['text'])
                    print(f"第一条数据某一个辅助模型对应的内容:  {first_example['initial_aux'][0][:20000]}..." if len(first_example['initial_aux'][0]) > 20000 else first_example['initial_aux'][0])
                else:
                    print(f"数据中不包含'text'字段。可用字段有: {list(first_example.keys())}")
                    print(f"第一条数据的部分内容: {str(first_example)[:1000]}...")

            # 移除未使用的列
            # dataset = self._remove_unused_columns(dataset)
        elif self.phase == 'pretrain_1':
            # 应用消息模板
            dataset = dataset.map(
                convert_to_messages_and_apply_template_pretrain_first,
                fn_kwargs={"main_tokenizer": self.main_tokenizer, "aux_tokenizer": self.aux_tokenizer, "compress_ratio": self.compress_ratio, "compress_mode": self.compress_mode},
                remove_columns=["context"] if "context" in dataset.column_names else None,
                num_proc=self.dataset_num_proc
            )
            """
            dataset = dataset.map(
                tokenize,
                fn_kwargs={"processing_class": self.main_tokenizer, "dataset_text_field": self.dataset_text_field},
                num_proc=self.dataset_num_proc
            )
            """
            # 移除未使用的列
            # dataset = self._remove_unused_columns(dataset)
        elif self.phase == 'pretrain_2':
            # 应用消息模板
            
            dataset = dataset.map(
                convert_to_messages_and_apply_template_pretrain_second,
                fn_kwargs={"main_tokenizer": self.main_tokenizer, "aux_tokenizer": self.aux_tokenizer, "compress_ratio": self.compress_ratio, "compress_mode": self.compress_mode},
                remove_columns=["Document","Question","Evidence"],
                num_proc=self.dataset_num_proc
            )
            """
            dataset = dataset.map(
                tokenize,
                fn_kwargs={"processing_class": self.main_tokenizer, "dataset_text_field": self.dataset_text_field},
                num_proc=self.dataset_num_proc
            )
            """
            # 移除未使用的列
            # dataset = self._remove_unused_columns(dataset)

        elif self.phase == 'simple_sft':
            # 应用消息模板
            dataset = dataset.map(
                convert_to_messages_and_apply_template_simple_sft,
                fn_kwargs={"main_tokenizer": self.main_tokenizer, "aux_tokenizer": self.aux_tokenizer, "compress_ratio": self.compress_ratio, "compress_mode": self.compress_mode},
                remove_columns=["Document","Question","Answer","Evidence"],
                num_proc=self.dataset_num_proc
            )
            """
            dataset = dataset.map(
                tokenize,
                fn_kwargs={"processing_class": self.main_tokenizer, "dataset_text_field": self.dataset_text_field},
                num_proc=self.dataset_num_proc
            )
            """
            # 移除未使用的列
            # dataset = self._remove_unused_columns(dataset)
        
        elif self.phase == 'multi_sft':
            dataset = dataset.map(
                convert_to_messages_and_apply_template_multi_sft,
                fn_kwargs={"main_tokenizer": self.main_tokenizer, "aux_tokenizer": self.aux_tokenizer, "compress_ratio": self.compress_ratio, "compress_mode": self.compress_mode, "chunker": self.chunker},
                remove_columns=["Document","Question","Answer"],
                num_proc=self.dataset_num_proc
            )           

        elif self.phase == 'test_normal_sft':
            dataset = dataset.map(
                convert_to_messages_and_apply_template_test_normal_sft,
                fn_kwargs={"main_tokenizer": self.main_tokenizer, "aux_tokenizer": self.aux_tokenizer},
                remove_columns=["context"] if "context" in dataset.column_names else None,
                num_proc=self.dataset_num_proc
            )
            """
            dataset = dataset.map(
                tokenize,
                fn_kwargs={"processing_class": self.main_tokenizer, "dataset_text_field": self.dataset_text_field},
                num_proc=self.dataset_num_proc
            )
            """
            # 移除未使用的列
            # dataset = self._remove_unused_columns(dataset)
        return dataset
    
    def setup(self, stage=None):
        # 加载训练数据集
        if self.train_file.endswith('.parquet'):
            train_file_format = 'parquet'
        elif self.train_file.endswith(('.json', '.jsonl')):
            train_file_format = 'json'
        else:
            raise ValueError(f"Unsupported file format for {self.train_file}. Supported formats are: .parquet, .json, .jsonl")

        train_dataset = load_dataset(train_file_format, data_files=self.train_file, split="train")
        self.train_dataset = self.prepare_dataset(train_dataset, is_train=True)

        # 加载验证数据集（如果有）
        if self.val_file:
            if self.val_file.endswith('.parquet'):
                val_file_format = 'parquet'
            elif self.val_file.endswith(('.json', '.jsonl')):
                val_file_format = 'json'
            else:
                raise ValueError(f"Unsupported file format for {self.val_file}. Supported formats are: .parquet, .json, .jsonl")

            val_dataset = load_dataset(val_file_format, data_files=self.val_file, split="train")
            self.val_dataset = self.prepare_dataset(val_dataset, is_train=False)
        else:
            self.val_dataset = None
    
    def train_dataloader(self):
        processor = DLProcessor(
            aux_tokenizer=self.aux_tokenizer,
            main_tokenizer=self.main_tokenizer,
        )
        # Create partial function with all required arguments except the batch
        collate_fn = partial(
            mom_collate_fn,
            processor=processor,
            max_length=self.max_length,
            response_template=self.response_template,
            response_end_marker=self.response_end_marker,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        if self.val_dataset:
            processor = DLProcessor(
                aux_tokenizer=self.aux_tokenizer,
                main_tokenizer=self.main_tokenizer,
            )

            # Create partial function with all required arguments except the batch
            collate_fn = partial(
                mom_collate_fn,
                processor=processor,
                max_length=self.max_length,
                response_template=self.response_template,
                response_end_marker=self.response_end_marker,
            )
            return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=collate_fn
            )
        return None
