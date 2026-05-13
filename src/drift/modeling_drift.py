import os 
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.tokenization_utils_base import BatchEncoding
from transformers import Qwen2Tokenizer, Qwen2TokenizerFast, LlamaTokenizer, LlamaTokenizerFast
from typing import Any, List, Dict, Optional, Union, Tuple
from tqdm import tqdm
import concurrent
from concurrent.futures import ThreadPoolExecutor
from drift.data.processor import DRIFTProcessor
from drift.utils.constants import COMPRESSION_TOKEN
from drift.data.templates import (
    SYSTEM_PROMPT_AUXILIARY as system_prompt_auxiliary,
    USER_PROMPT_AUXILIARY as user_prompt_auxiliary,
    ASSISTANT_PROMPT_AUXILIARY as assistant_prompt_auxiliary,
    SYSTEM_PROMPT_MAIN as system_prompt_main,
    USER_PROMPT_MAIN as user_prompt_main,
    SYSTEM_PROMPT_AUXILIARY_PRETRAIN as system_prompt_auxiliary_pretrain,
    USER_PROMPT_AUXILIARY_PRETRAIN as user_prompt_auxiliary_pretrain,
    USER_PROMPT_MAIN_PRETRAIN as user_prompt_main_pretrain,
    ASSISTANT_PROMPT_AUXILIARY_PRETRAIN as assistant_prompt_auxiliary_pretrain,
    SYSTEM_PROMPT_AUXILIARY_PRETRAIN_SECOND as system_prompt_auxiliary_pretrain_second,
    USER_PROMPT_AUXILIARY_PRETRAIN_SECOND as user_prompt_auxiliary_pretrain_second,
    ASSISTANT_PROMPT_AUXILIARY_PRETRAIN_SECOND as assistant_prompt_auxiliary_pretrain_second,
    USER_PROMPT_MAIN_PRETRAIN_SECOND as user_prompt_main_pretrain_second,
    USER_PROMPT_MAIN_SFT as user_prompt_main_sft,
    USER_PROMPT_MAIN_MULTI_SFT as user_prompt_main_multi_sft,
)
from drift.utils.chunking import EnhancedRecursiveCharacterTextSplitter
import os
import gc
from typing import Any, Callable
import multiprocessing as mp
import time

def build_projector(input_size, output_size, use_layer_norm=False):
    """Build the three-layer MLP projector used by DRIFT."""
    if use_layer_norm:
        return nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size),
            nn.LayerNorm(output_size)
        )
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.GELU(),
        nn.Linear(output_size, output_size),
        nn.GELU(),
        nn.Linear(output_size, output_size)
    )


def _normalize_optional_batch_list(values, total_samples: int, name: str):
    """Expand optional per-sample arguments and reject length mismatches."""

    if values is None:
        return [None] * total_samples
    if isinstance(values, list):
        if len(values) != total_samples:
            raise ValueError(
                f"{name} must have length {total_samples}, got {len(values)}."
            )
        return values
    raise TypeError(f"{name} must be a list or None, got {type(values).__name__}.")



class DRIFTModel(nn.Module):
    def __init__(
        self,
        main_model_name: str,
        auxiliary_model_name: str,
        num_attention_heads: int = 8,
        device_map_main: str = "cuda:0",
        device_map_auxiliary: str = "balanced_low_0",
        device: str = "cuda:0",
        frozen_main: bool = False,  # 默认不冻结主模型
        frozen_auxiliary: bool = False,  # 默认不冻结辅助模型
        frozen_projector: bool = False,
        chunk_size: int = 4096,
        overlap: int = 200,
        attn_implementation: str = None,
        use_layer_norm: bool = False,
        projector_path: str = None
    ):
        super().__init__()
        self.name = "DRIFT"
        self.device = device
        self.compression_token = COMPRESSION_TOKEN
        self.available_gpus = []
        for i in range(torch.cuda.device_count()):
            if torch.cuda.is_available():
                self.available_gpus.append(f'cuda:{i}')
        if not self.available_gpus:  # 如果没有可用的GPU，使用CPU
            self.available_gpus = ['cpu']
        # 1. 加载主模型
        print(f"Loading main model on device: {device_map_main}")
        load_kwargs = {"output_hidden_states": True, "device_map": device_map_main, "torch_dtype": torch.bfloat16}
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
            
        self.main_model = AutoModelForCausalLM.from_pretrained(
            main_model_name,
            trust_remote_code=True,
            **load_kwargs
        )
        self.main_tokenizer = AutoTokenizer.from_pretrained(main_model_name,trust_remote_code=True)
        if self.main_tokenizer.pad_token is None:
            self.main_tokenizer.pad_token = self.main_tokenizer.eos_token
        self.main_hidden_size = self.main_model.config.hidden_size
        self.overlap = overlap
        
        
        # # 添加压缩token到主模型tokenizer
        # if compression_token not in self.main_tokenizer.get_vocab():
        #     print("添加压缩token")
        #     old_size = len(self.main_tokenizer)
        #     self.main_tokenizer.add_tokens([compression_token])
        #     # 如果词表大小变化，需要调整模型embedding大小
        #     if len(self.main_tokenizer) > old_size:
        #         self.main_model.resize_token_embeddings(len(self.main_tokenizer))
        
        self.compression_token_id = self.main_tokenizer.convert_tokens_to_ids(self.compression_token)
        self.frozen_main = frozen_main
        self.frozen_auxiliary = frozen_auxiliary
        self.frozen_projector = frozen_projector
        self.projector_path = projector_path
        self.chunk_size=chunk_size

        # 2. 加载辅助模型 - 使用balanced_low_0配置
        print(f"Loading auxiliary model on device: {device_map_auxiliary}")
        aux_load_kwargs = {"output_hidden_states": True, "device_map": device_map_auxiliary, "torch_dtype": torch.bfloat16}
        if attn_implementation:
            aux_load_kwargs["attn_implementation"] = attn_implementation
            
        self.auxiliary_model = AutoModelForCausalLM.from_pretrained(
            auxiliary_model_name,
            trust_remote_code=True,
            **aux_load_kwargs
        )
        
        # 设置辅助模型参数状态
        if self.frozen_auxiliary:
            # 仅当frozen_auxiliary为True时才冻结参数
            for param in self.auxiliary_model.parameters():
                param.requires_grad = False
        
        # 设置主模型参数状态
        if self.frozen_main:
            # 仅当frozen_main为True时才冻结参数
            for param in self.main_model.parameters():
                param.requires_grad = False
            
        

        self.auxiliary_tokenizer = AutoTokenizer.from_pretrained(auxiliary_model_name,trust_remote_code=True)
        if self.auxiliary_tokenizer.pad_token is None:
            self.auxiliary_tokenizer.pad_token = self.auxiliary_tokenizer.eos_token
        self.auxiliary_hidden_size = self.auxiliary_model.config.hidden_size
        
        # 3. 创建投影器 - 将辅助模型hidden states投影到主模型hidden states空间
        self.projector = self.create_projector(
            input_size=self.auxiliary_hidden_size,
            output_size=self.main_hidden_size,
            use_layer_norm=use_layer_norm
        ).to(self.device)
        # 确保投影器使用与主模型相同的数据类型
        dtype = next(self.main_model.parameters()).dtype
        self.projector = self.projector.to(self.device).to(dtype)
        if self.frozen_projector:
            # 仅当frozen_projector为True时才冻结参数
            for param in self.projector.parameters():
                param.requires_grad = False
        if projector_path:
            projector_state_dict = torch.load(projector_path, map_location=self.device)
            # 加载projector权重
            self.projector.load_state_dict(projector_state_dict)
        
        # 5. 创建分块器
        self.chunker = EnhancedRecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.auxiliary_tokenizer,
            chunk_size=self.chunk_size,          # 每个块的最大字符数
            chunk_overlap=self.overlap,        # 块之间的重叠字符数
            separators=["\n\n", "\n", ".", ",", "。", "，", " ", ""]  # 分割优先级顺序
        )
                # 获取辅助模型所在的所有设备
        self.aux_devices = []
        for param in self.auxiliary_model.parameters():
            device = param.device
            if device not in self.aux_devices:
                self.aux_devices.append(device)
        
        print(f"Found {len(self.aux_devices)} devices for auxiliary model: {self.aux_devices}")
        
        # 如果没有找到多个设备，则使用当前设备
        if not self.aux_devices:
            self.aux_devices = [next(self.auxiliary_model.parameters()).device]

        self.processor = DRIFTProcessor(aux_tokenizer=self.auxiliary_tokenizer, main_tokenizer=self.main_tokenizer)

        
    
    def get_trainable_parameters(self):
        """获取需要训练的参数"""
        trainable_params = []
        
        # 如果主模型未冻结，添加主模型参数
        if not self.frozen_main:
            trainable_params.extend(self.main_model.parameters())
            
        # 如果辅助模型未冻结，添加辅助模型参数
        if not self.frozen_auxiliary:
            trainable_params.extend(self.auxiliary_model.parameters())
            
        # 投影器参数
        if not self.frozen_projector:
            trainable_params.extend(self.projector.parameters())
        
        
        return trainable_params
    

    def create_projector(self, input_size, output_size, use_layer_norm=False):
        """创建三层投影器，将辅助模型hidden states投影到主模型hidden states空间"""
        return build_projector(input_size, output_size, use_layer_norm)
            
    def forward(
            self, 
            input_ids: torch.Tensor, 
            auxiliary_input: List = None,
            attention_mask: torch.Tensor = None, 
            labels: torch.Tensor = None,
        ) -> Dict[str, torch.Tensor]:
        """
        前向传播方法：辅助模型将context压缩到特殊token，主模型使用这些压缩信息
        所有主要逻辑都在main_device上完成
        """
        batch_size = input_ids.shape[0]
        main_device = self.device  # 主设备固定为第一张卡
        # 1. 确保主输入统一在main_device上
        input_ids = input_ids.to(main_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(main_device)
        # if labels is not None:
        #     labels = labels.to(main_device)
        # debug
        # first_input_ids = input_ids[0]
        # print((first_input_ids == self.main_tokenizer.convert_tokens_to_ids('<|CPS|>')).sum())
        # if hasattr(first_input_ids, 'tolist'):
        #     first_input_ids = first_input_ids.tolist()

        # # 1. 转成token字符串列表
        # tokens = self.main_tokenizer.convert_ids_to_tokens(first_input_ids, skip_special_tokens=True)

        # # 2. 用空格拼接
        # token_string = ';'.join(tokens)

        # print(token_string)
        
        # 2. 获取输入的原始embeddings
        inputs_embeds = self.main_model.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.to(main_device)
        if auxiliary_input:
            # 3. 从辅助模型获取压缩的隐藏状态（结果将确保回到main_device）
            compression_embeds = self.get_auxiliary_compression_embeds(auxiliary_input)

            # 4. 在main_device上处理compression token嵌入
            for i in range(batch_size):
                # 找出当前样本中所有压缩token的位置
                compression_positions = (input_ids[i] == self.compression_token_id).nonzero(as_tuple=True)[0]
                
                # 如果当前样本有压缩token且有对应的辅助模型输出
                if len(compression_positions) > 0 and i < len(compression_embeds) and compression_embeds[i] is not None:
                    # 获取当前样本的辅助模型输出（已确保在main_device上）
                    aux_embeds = compression_embeds[i]  # [num_tokens, hidden_size]
                    
                    # 检查压缩token数量与辅助模型输出的数量是否匹配
                    num_compression_tokens = len(compression_positions)
                    num_aux_embeds = aux_embeds.size(0)
                    
                    # 如果数量不匹配，记录警告并进行处理
                    if num_compression_tokens != num_aux_embeds:
                        print(f"警告：样本 {i} 中压缩token数量 ({num_compression_tokens}) 与辅助模型输出数量 ({num_aux_embeds}) 不匹配")
                        
                        # 如果压缩token数量更多，我们只使用可用的辅助模型输出
                        if num_compression_tokens > num_aux_embeds:
                            compression_positions = compression_positions[:num_aux_embeds]
                        # 如果辅助模型输出更多，我们只使用前面的输出
                        else:
                            aux_embeds = aux_embeds[:num_compression_tokens]
                    
                    # 将辅助模型输出映射到主模型空间（确保在main_device上）
                    # 投影器已经在初始化时固定在main_device上
                    # 检查并移动到正确设备
                    if aux_embeds.device != self.device:
                        aux_embeds = aux_embeds.to(self.device)
                    projected_embeds = self.projector(aux_embeds.to(inputs_embeds.dtype))  # [num_tokens, hidden_size]
                    
                    # 将映射后的embeddings分配给对应位置的压缩token
                    for j, pos in enumerate(compression_positions):
                        if j < projected_embeds.size(0):
                            inputs_embeds[i, pos] = projected_embeds[j]
            
            # 5. 最终确保所有输入都在main_device上
            # inputs_embeds = inputs_embeds.to(main_device)
        
        # 6. 使用主模型前向传播
        main_outputs = self.main_model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 7. 返回结果
        return main_outputs

    def visualize_masked_tokens(self, input_ids, labels):
        """
        Extracts the first item from input_ids and labels batches and returns:
        1. A list of text tokens corresponding to input_ids
        2. A list where labels with -100 remain as -100, but other label values are converted to text tokens
        
        Args:
            input_ids: Tensor containing token IDs [batch_size, sequence_length]
            labels: Tensor containing label IDs [batch_size, sequence_length]
            
        Returns:
            tuple: (input_tokens_list, label_tokens_list)
        """
        # Get the first item from the batch
        first_input_ids = input_ids[0].cpu().tolist()
        first_labels = labels[0].cpu().tolist()
        
        # Convert input_ids to text tokens
        input_tokens = []
        for token_id in first_input_ids:
            token_text = self.main_tokenizer.convert_ids_to_tokens(token_id)
            input_tokens.append(token_text)
        
        # Process labels: -100 remains as is, others are converted to text tokens
        label_tokens = []
        for label_id in first_labels:
            if label_id == -100:
                label_tokens.append(-100)  # Keep -100 as is
            else:
                token_text = self.main_tokenizer.convert_ids_to_tokens(label_id)
                label_tokens.append(token_text)
        
        # Return both lists for visualization
        return input_tokens, label_tokens

    def _process_batch_encoding(self, batch_encoding) -> List[torch.Tensor]:
        """处理直接的batch_encoding对象"""
        # 获取<|CPS|>对应的token_id
        cps_token_id = self.auxiliary_tokenizer.convert_tokens_to_ids(COMPRESSION_TOKEN)
        aux_device = self.aux_devices[0]

        # 将batch_encoding移动到模型设备
        batch_input = {
            'input_ids': batch_encoding['input_ids'].to(aux_device),
            'attention_mask': batch_encoding['attention_mask'].to(aux_device)
        }

        # 检查输入中是否包含<|CPS|> token
        contains_cps = (batch_input['input_ids'] == cps_token_id).any().item()

        # 使用辅助模型获取hidden states
        with torch.no_grad() if self.frozen_auxiliary else torch.enable_grad():
            outputs = self.auxiliary_model.model(
                **batch_input,
                output_hidden_states=True,
                use_cache=False
            )
        # 获取最后层的hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_length, hidden_dim]
        #hidden_states = outputs.last_hidden_state
        compression_embeds = []

        # 如果不包含压缩token，使用最后一个token作为替代
        if not contains_cps:
            # 对于每个序列，使用最后一个有效token作为压缩token
            for i in range(batch_input['input_ids'].size(0)):
                last_valid_pos = batch_input['attention_mask'][i].sum() - 1
                last_token_state = hidden_states[i, last_valid_pos].unsqueeze(0)
                compression_embeds.append(last_token_state)

        else:
            # 提取所有<|CPS|>的位置对应的隐藏状态，跳过第一个
            for i in range(batch_input['input_ids'].size(0)):
                # 找出所有<|CPS|>的位置
                cps_positions = (
                    (batch_input['input_ids'][i] == cps_token_id)
                    .nonzero(as_tuple=True)[0]
                )

                if len(cps_positions) > 1:  # 确保至少有两个压缩标记
                    # 跳过第一个压缩标记，从第二个开始收集
                    seq_cps_hidden_states = []
                    for pos in cps_positions[1:]:
                        seq_cps_hidden_states.append(hidden_states[i, pos])

                    # 堆叠该序列的所有<|CPS|>隐藏状态
                    if seq_cps_hidden_states:
                        seq_cps_tensor = torch.stack(
                            seq_cps_hidden_states,
                            dim=0
                        )  # [num_cps-1, hidden_dim]
                        compression_embeds.append(seq_cps_tensor)
                    else:
                        # 如果去掉第一个后没有剩余的压缩token
                        last_valid_pos = batch_input['attention_mask'][i].sum() - 1
                        compression_embeds.append(
                            hidden_states[i, last_valid_pos].unsqueeze(0)
                        )

                elif len(cps_positions) == 1:
                    # 如果只有一个压缩token，且我们要跳过它
                    last_valid_pos = batch_input['attention_mask'][i].sum() - 1
                    compression_embeds.append(
                        hidden_states[i, last_valid_pos].unsqueeze(0)
                    )

                else:
                    # 如果没有找到<|CPS|>
                    last_valid_pos = batch_input['attention_mask'][i].sum() - 1
                    compression_embeds.append(
                        hidden_states[i, last_valid_pos].unsqueeze(0)
                    )

        # 检查NaN或无穷大
        for i, result in enumerate(compression_embeds):
            if result is not None and (
                torch.isnan(result).any() or torch.isinf(result).any()
            ):
                print(f"警告：批次{i}中检测到NaN或无穷大，使用零向量替代")
                compression_embeds[i] = torch.zeros_like(result)

        # 清理内存
        del hidden_states, outputs, batch_input

        return compression_embeds

    def _process_auxiliary_input_list(self, auxiliary_input: List) -> List[torch.Tensor]:
        """处理列表格式的auxiliary_input（显存优化版）"""
        batch_size = len(auxiliary_input)
        compression_embeds = []
        cps_token_id = self.auxiliary_tokenizer.convert_tokens_to_ids(COMPRESSION_TOKEN)
        aux_device = self.aux_devices[0]

        for batch_idx in range(batch_size):
            # 1. 基础检查
            if auxiliary_input[batch_idx] is None or 'input_ids' not in auxiliary_input[batch_idx]:
                compression_embeds.append(None)
                continue
                
            batch_encoding = auxiliary_input[batch_idx]
            input_ids = batch_encoding['input_ids']
            attention_mask = batch_encoding.get('attention_mask', None)

            # 2. 格式标准化：确保是 Tensor 并移动到辅助设备
            if isinstance(input_ids, list):
                # 处理嵌套列表 [[...]]
                input_ids = torch.tensor(input_ids).to(aux_device)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                else:
                    attention_mask = torch.tensor(attention_mask).to(aux_device)
            else:
                input_ids = input_ids.to(aux_device)
                attention_mask = attention_mask.to(aux_device) if attention_mask is not None else torch.ones_like(input_ids)

            # 3. 统一调用辅助模型
            with torch.no_grad() if self.frozen_auxiliary else torch.enable_grad():
                outputs = self.auxiliary_model.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
                # 直接取最后一层输出 [sub_batch, seq_len, hidden_dim]
                #hidden_states = outputs.last_hidden_state
                hidden_states = outputs.hidden_states[-1]

            # 4. 提取 Embedding
            batch_results = []
            for i in range(input_ids.size(0)):
                cps_positions = (input_ids[i] == cps_token_id).nonzero(as_tuple=True)[0].to(hidden_states.device)
                cps_positions = cps_positions.to(hidden_states.device)
                # 如果有多个 CPS，跳过第一个（占位符），提取后续所有内容对应的 CPS
                if len(cps_positions) > 1:
                    seq_cps_embeds = hidden_states[i, cps_positions[1:]]
                    batch_results.append(seq_cps_embeds)
                else:
                    # 回退逻辑：无 CPS 或 仅有 1 个占位符 CPS
                    # 取序列最后一个有效位置（attention_mask 为 1 的最末尾）
                    last_valid_pos = attention_mask[i].sum() - 1
                    fallback_embed = hidden_states[i, last_valid_pos].unsqueeze(0)
                    batch_results.append(fallback_embed)

            # 5. 合并并检查异常
            if batch_results:
                result = torch.cat(batch_results, dim=0)
                if torch.isnan(result).any() or torch.isinf(result).any():
                    print(f"警告：批次 {batch_idx} 中检测到 NaN/Inf，已重置为零向量")
                    result = torch.zeros_like(result)
                compression_embeds.append(result)
            else:
                compression_embeds.append(None)

            # 显式清理显存
            del hidden_states, outputs

        return compression_embeds

    def get_auxiliary_compression_embeds(self, auxiliary_input: Union[List[Dict[str, Any]], Dict[str, torch.Tensor], BatchEncoding]) -> List[torch.Tensor]:
        """
        从辅助模型获取压缩的隐藏状态，提取所有<|CPS|>对应的隐藏状态
        跳过每个样本中的第一个压缩标记(仅作为示例用途)
        
        参数:
        auxiliary_input: 可以是两种格式：
            1. List: 辅助模型的输入列表，每个样本包含预处理的上下文
            2. batch_encoding对象: 直接的tokenize后的batch对象，可以直接给辅助模型
        
        返回:
        compression_embeds: 列表，包含每个样本所有压缩token的embeddings
        """
        # 检查输入格式
        if (isinstance(auxiliary_input, dict) or isinstance(auxiliary_input, BatchEncoding)) and 'input_ids' in auxiliary_input:
            # 情况2: 直接是batch_encoding对象
            return self._process_batch_encoding(auxiliary_input)
        elif isinstance(auxiliary_input, list):
            # 情况1: 是列表格式
            if not auxiliary_input or all(x is None for x in auxiliary_input):
                return [None] * len(auxiliary_input)
            return self._process_auxiliary_input_list(auxiliary_input)
        else:
            raise ValueError(f"Unsupported auxiliary_input format: {type(auxiliary_input)}")

    def generate_sft(
            self,
            input_ids: torch.Tensor,
            auxiliary_input: List,
            attention_mask: Optional[torch.Tensor] = None,
            max_new_tokens: int = 100,
            min_length: int = 0,
            do_sample: bool = False,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 1.0,
            num_beams: int = 1,
            no_repeat_ngram_size: int = 0,
        ):
        """
        支持批量处理的自回归生成文本，适用于新架构：
        辅助模型提供压缩信息，主模型直接回答问题
        """

        batch_size = input_ids.shape[0]
        main_device = self.device
        
        # 将输入张量移至主设备
        input_ids = input_ids.to(main_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(main_device)
        
        # 确保输入长度有效
        if isinstance(auxiliary_input, list):
            assert len(auxiliary_input) == batch_size, "auxiliary_input列表长度必须等于batch_size"
        
        # 如果未提供注意力掩码，则创建全1的掩码
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=main_device)
        
        # 获取辅助模型的压缩embeddings
        compression_embeds = self.get_auxiliary_compression_embeds(auxiliary_input)
        
        # 获取主模型的输入embeddings
        inputs_embeds = self.main_model.get_input_embeddings()(input_ids)
            # 添加压缩token到主模型tokenizer

        self.compression_token_id = self.main_tokenizer.convert_tokens_to_ids(self.compression_token)
        print(self.compression_token_id)
        # 将压缩token替换为辅助模型的压缩信息
        compression_token_positions = (input_ids == self.compression_token_id).nonzero(as_tuple=True)
        for i in range(batch_size):
            batch_positions = (compression_token_positions[0] == i)
            if torch.any(batch_positions):
                seq_positions = compression_token_positions[1][batch_positions]
                if i < len(compression_embeds) and compression_embeds[i] is not None:
                    # 将压缩embeddings映射到主模型的embedding空间
                    projected_embeds = self.projector(compression_embeds[i].to(inputs_embeds.dtype))
                    final_embed = projected_embeds[0]
                    
                    # 替换所有压缩token
                    for pos in seq_positions:
                        inputs_embeds[i, pos] = final_embed.to(main_device)
        
        # 准备生成参数
        generate_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "min_length": min_length,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "num_beams": num_beams,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }
        
        # 直接调用主模型的generate函数
        with torch.no_grad():
            generated_tokens = self.main_model.generate(**generate_kwargs)
        
        return generated_tokens
            

    def chat_sft(
        self, 
        questions: List[str],
        contexts: List[str],
        max_new_tokens: int = 500,
        do_sample: bool = True
    ) -> List[str]:
        """
        处理对话请求 - 适用于新架构，辅助模型提供压缩信息给主模型
        
        questions: 问题列表
        contexts: 与问题对应的上下文列表
        max_new_tokens: 最大生成token数
        """
        batch_size = len(questions)
        assert len(questions) == len(contexts), "question数量和context数量需要相等"
        
        # 对每个上下文进行切分
        chunk_context = []
        for context in contexts:
            temp_contexts = self.chunker.split_text(context)
            print(f"文本块数量: {len(temp_contexts)}")
            chunk_context.append(temp_contexts)
        
        # 设置tokenizer的填充方向
        self.main_tokenizer.padding_side = 'left'
        self.auxiliary_tokenizer.padding_side = 'left'
        if self.compression_token not in self.main_tokenizer.get_vocab():
            print("添加压缩token")
            old_size = len(self.main_tokenizer)
            self.main_tokenizer.add_tokens([self.compression_token])
            # 如果词表大小变化，需要调整模型embedding大小
            if len(self.main_tokenizer) > old_size:
                self.main_model.resize_token_embeddings(len(self.main_tokenizer))
        
        if self.compression_token not in self.auxiliary_tokenizer.get_vocab():
            print("添加压缩token")
            old_size = len(self.auxiliary_tokenizer)
            self.auxiliary_tokenizer.add_tokens([self.compression_token])
            # 如果词表大小变化，需要调整模型embedding大小
            if len(self.main_tokenizer) > old_size:
                self.auxiliary_model.resize_token_embeddings(len(self.auxiliary_tokenizer))
        # 创建主模型输入 - 添加压缩token
        num_compression_tokens = 1
        background = " ".join([self.compression_token] * num_compression_tokens)
        
        # 格式化问题
        formatted_questions = [
            self.main_tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt_main}, 
                {"role": "user", "content": user_prompt_main.format(background=background, question=question)}],
                tokenize=False,
                add_generation_prompt=True
            )
            for question in questions
        ]
        
        # 格式化上下文 - 为辅助模型准备输入
        auxiliary_inputs = []
        for i, doc_chunks in enumerate(chunk_context):
            batch_aux_inputs = []
            for context in doc_chunks:
                # 创建辅助模型的输入格式
                messages_aux = [
                    {"role": "system", "content": system_prompt_auxiliary}, 
                    {"role": "user", "content": user_prompt_auxiliary.format(context=context, question=questions[i])}, 
                    {"role": "assistant", "content": assistant_prompt_auxiliary.format(judgement='Yes')}
                ]
                temp_aux = self.auxiliary_tokenizer.apply_chat_template(messages_aux, tokenize=False)
                batch_aux_inputs.append(temp_aux)
                print(temp_aux)
            # 对整批上下文进行tokenize
            tokenized_context = self.auxiliary_tokenizer(
                batch_aux_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            auxiliary_inputs.append(tokenized_context)
        
        # Tokenize主模型的输入
        tokenized_questions = self.main_tokenizer(
            formatted_questions, 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # 存储原始输入长度
        input_lengths = [len(tokenized_questions.input_ids[i]) for i in range(batch_size)]
        
        # 运行生成
        print("开始生成回答...")
        generated_ids = self.generate(
            input_ids=tokenized_questions.input_ids,
            auxiliary_input=auxiliary_inputs,
            attention_mask=tokenized_questions.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1
        )
        
        # 仅解码新生成的部分
        generated_texts = []
        for i in range(batch_size):
            # 仅获取新生成的token（不包括输入）
            new_tokens = generated_ids[i]
            # 仅解码这些新token
            generated_text = self.main_tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(f"问题 {i+1} 生成的回答: {generated_text}")
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def generate_pretrain(
            self,
            input_ids: torch.Tensor,
            auxiliary_input: Union[List[Dict[str, Any]], Dict[str, torch.Tensor], BatchEncoding],
            attention_mask: Optional[torch.Tensor] = None,
            max_new_tokens: int = 100,
            min_length: int = 0,
            do_sample: bool = False,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 1.0,
            num_beams: int = 1,
            no_repeat_ngram_size: int = 0,
        ):
            """
            支持批量处理的自回归生成文本，专门用于预训练阶段：
            辅助模型提供压缩信息，主模型根据压缩信息重建原文本
            """
            batch_size = input_ids.shape[0]
            main_device = self.device
            
            # 将输入张量移至主设备
            input_ids = input_ids.to(main_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(main_device)
            
            # 确保输入长度有效
            if isinstance(auxiliary_input, list):
                assert len(auxiliary_input) == batch_size, "auxiliary_input列表长度必须等于batch_size"
            
            # 如果未提供注意力掩码，则创建全1的掩码
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=main_device)
            
            # 获取辅助模型的压缩embeddings
            compression_embeds = self.get_auxiliary_compression_embeds(auxiliary_input)
            # 获取主模型的输入embeddings
            inputs_embeds = self.main_model.get_input_embeddings()(input_ids)
            self.compression_token_id = self.main_tokenizer.convert_tokens_to_ids(self.compression_token)
            # ------------------- 新增核对功能开始 -------------------
            for i in range(batch_size):
                # 找主模型输入的压缩token位置
                compression_positions = (input_ids[i] == self.compression_token_id).nonzero(as_tuple=True)[0]
                num_compression_tokens = len(compression_positions)
                # 获取辅助模型输出
                num_aux_embeds = 0
                if i < len(compression_embeds) and compression_embeds[i] is not None:
                    num_aux_embeds = compression_embeds[i].size(0)
                print(f"[核对] 样本 {i}: 主模型输入压缩token数: {num_compression_tokens}，辅助模型输出压缩embedding数: {num_aux_embeds}")
                if num_compression_tokens != num_aux_embeds:
                    print(f"[警告] 样本 {i} 的主模型输入压缩token数 ({num_compression_tokens}) 与辅助模型输出embedding数 ({num_aux_embeds}) 不一致！")
            # 将压缩token替换为辅助模型的压缩信息
            compression_token_positions = (input_ids == self.compression_token_id).nonzero(as_tuple=True)
            for i in range(batch_size):
                batch_positions = (compression_token_positions[0] == i)
                if torch.any(batch_positions):
                    seq_positions = compression_token_positions[1][batch_positions]
                    if i < len(compression_embeds) and compression_embeds[i] is not None:
                        # 将压缩embeddings映射到主模型的embedding空间
                        if compression_embeds[i].device != main_device:
                            compression_embeds[i] = compression_embeds[i].to(main_device)
                        projected_embeds = self.projector(compression_embeds[i].to(inputs_embeds.dtype))
                        # 对每个压缩token位置替换对应的embedding
                        for j, pos in enumerate(seq_positions):
                            if j < projected_embeds.size(0):
                                inputs_embeds[i, pos] = projected_embeds[j].to(main_device)
                            else:
                                # 如果压缩token数量多于提供的embedding，使用最后一个
                                inputs_embeds[i, pos] = projected_embeds[-1].to(main_device)
            # 准备生成参数
            generate_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "min_length": min_length,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "num_beams": num_beams,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "pad_token_id": self.main_tokenizer.pad_token_id, 
                "eos_token_id": self.main_tokenizer.eos_token_id,  
                "use_cache": True,  # 推荐添加
            }
            # 直接调用主模型的generate函数
            with torch.no_grad():
                generated_tokens = self.main_model.generate(**generate_kwargs)
            
            return generated_tokens

    def chat_pretrain(
        self, 
        contexts: List[str],
        max_new_tokens: int = 500,
        do_sample: bool = True,
        temperature: float = 0.1,
        compress_ratio: int = 8,  # 压缩比例，用于计算压缩token数量
        compress_mode: str = 'fix',
        batch_size: int = 4  # 新增批处理大小参数
    ) -> List[str]:
        """
        处理预训练对话请求 - 辅助模型提供压缩信息，主模型重建原文本
        优化版本：使用批处理提高效率
        
        contexts: 上下文文本列表
        max_new_tokens: 最大生成token数
        do_sample: 是否进行采样
        temperature: 温度参数
        compress_ratio: 压缩比例
        compress_mode: 压缩模式
        batch_size: 批处理大小，如果为None则使用全部数据的长度
        """
        total_samples = len(contexts)
        batch_size = min(batch_size, total_samples)
        
        # 设置tokenizer的填充方向
        self.main_tokenizer.padding_side = 'left'
        self.auxiliary_tokenizer.padding_side = 'left'
        
        # 确保tokenizer中有压缩token
        if self.compression_token not in self.main_tokenizer.get_vocab():
            print("添加压缩token到主模型tokenizer")
            old_size = len(self.main_tokenizer)
            self.main_tokenizer.add_tokens([self.compression_token])
            if len(self.main_tokenizer) > old_size:
                self.main_model.resize_token_embeddings(len(self.main_tokenizer))
        
        if self.compression_token not in self.auxiliary_tokenizer.get_vocab():
            print("添加压缩token到辅助模型tokenizer")
            old_size = len(self.auxiliary_tokenizer)
            self.auxiliary_tokenizer.add_tokens([self.compression_token])
            if len(self.auxiliary_tokenizer) > old_size:
                self.auxiliary_model.resize_token_embeddings(len(self.auxiliary_tokenizer))
        
        all_generated_texts = []
        
        # 按批次处理数据
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_contexts = contexts[batch_start:batch_end]
            current_batch_size = len(batch_contexts)
            
            print(f"处理批次 {batch_start//batch_size + 1}: 样本 {batch_start+1}-{batch_end}")
            
            # 准备当前批次的辅助模型输入和主模型输入
            auxiliary_texts = []
            main_texts = []
            
            for text in batch_contexts:
                # 计算文本token长度用于确定压缩token数量
                tokens = self.auxiliary_tokenizer.encode(text)
                text_length = len(tokens)
                if compress_mode == 'fix':
                    num_compression_tokens = max(text_length // compress_ratio, 1)
                elif compress_mode == 'small_threshold':
                    if text_length <= 128:
                        num_compression_tokens = max(128 // compress_ratio, 1)
                    elif text_length <= 256:
                        num_compression_tokens = max(256 // compress_ratio, 1)
                    elif text_length <= 512:
                        num_compression_tokens = max(512 // compress_ratio, 1)
                    elif text_length <= 1024:
                        num_compression_tokens = max(1024 // compress_ratio, 1)
                    elif text_length <= 2048:  
                        num_compression_tokens = max(2048 // compress_ratio, 1)
                    elif text_length <= 4096:
                        num_compression_tokens = max(4096 // compress_ratio, 1)
                    elif text_length <= 8192:
                        num_compression_tokens = max(8192 // compress_ratio, 1)
                    else:
                        print(f"Warning: length > 8192 ({text_length}), using default compress ratio")
                        num_compression_tokens = max(text_length // compress_ratio, 1)
                else:
                    raise ValueError(f"未知的compress_mode: {compress_mode}")
                
                # 构造压缩token字符串
                compression_tokens_str = " ".join([self.compression_token] * num_compression_tokens)
                
                # 为辅助模型准备输入 - 将原文本压缩成特殊token
                messages_aux = [
                    {"role": "system", "content": system_prompt_auxiliary_pretrain.format(num=num_compression_tokens)}, 
                    {"role": "user", "content": user_prompt_auxiliary_pretrain.format(context=text)}, 
                    {"role": "assistant", "content": assistant_prompt_auxiliary_pretrain.format(CPS_tokens=compression_tokens_str)}
                ]
                
                aux_input = self.auxiliary_tokenizer.apply_chat_template(
                    messages_aux, tokenize=False, enable_thinking=False
                )
                auxiliary_texts.append(aux_input)
                
                # 为主模型准备输入 - 使用压缩token重建原文本
                messages_main = [
                    {"role": "user", "content": user_prompt_main_pretrain.format(compressed_information=compression_tokens_str)}
                ]
                
                main_input = self.main_tokenizer.apply_chat_template(
                    messages_main,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                main_texts.append(main_input)

            # 批量tokenize主模型输入
            tokenized_main_inputs = self.main_tokenizer(
                main_texts, 
                padding=True, 
                return_tensors="pt",
                max_length=8192,
                truncation=True
            ).to(self.device)
            
            # 批量tokenize辅助模型输入
            tokenized_auxiliary_inputs = self.auxiliary_tokenizer(
                auxiliary_texts,
                padding=True,
                return_tensors="pt",
                max_length=8192,
                truncation=True
            )
            
            # 运行当前批次的生成
            print(f"开始生成批次 {batch_start//batch_size + 1} 的预训练回答...")
            generated_ids = self.generate_pretrain(
                input_ids=tokenized_main_inputs['input_ids'],
                auxiliary_input=tokenized_auxiliary_inputs,
                attention_mask=tokenized_main_inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature
            )
            
            # 解码当前批次的生成结果
            batch_generated_texts = []
            for i in range(current_batch_size):
                
                # 获取新生成的token（排除输入部分）
                new_tokens = generated_ids[i]
                
                # 解码这些新token
                generated_text = self.main_tokenizer.decode(new_tokens, skip_special_tokens=True)
                print(f"批次 {batch_start//batch_size + 1} 样本 {i+1} 生成的文本: {generated_text[:100]}...")  # 只打印前100个字符
                print(f"总共{len(new_tokens)}个token")
                batch_generated_texts.append(generated_text)
            
            # 将当前批次的结果添加到总结果中
            all_generated_texts.extend(batch_generated_texts)
        
        return all_generated_texts



    # def chat_pretrain_second(
    #     self, 
    #     contexts: List[str],
    #     questions: List[str],
    #     max_new_tokens: int = 500,
    #     do_sample: bool = True,
    #     temperature: float = 0.1,
    #     compress_ratio: int = 8,  # 压缩比例，用于计算压缩token数量
    #     compress_mode: str = 'fix'
    # ) -> List[str]:
    #     """
    #     处理预训练对话请求 V2 - 辅助模型提供压缩信息，主模型重建知识文本（基于文档和问题）
    #     contexts: 文档文本列表
    #     questions: 问题列表
    #     max_new_tokens: 最大生成token数
    #     do_sample: 是否进行采样
    #     temperature: 温度参数
    #     compress_ratio: 压缩比例
    #     compress_mode: 压缩模式
    #     """
    #     assert len(contexts) == len(questions), "contexts 和 questions 长度必须一致"
    #     batch_size = len(contexts)
        
    #     # 设置tokenizer的填充方向
    #     self.main_tokenizer.padding_side = 'left'
    #     self.auxiliary_tokenizer.padding_side = 'left'
        
    #     # 确保tokenizer中有压缩token
    #     if self.compression_token not in self.main_tokenizer.get_vocab():
    #         print("添加压缩token到主模型tokenizer")
    #         old_size = len(self.main_tokenizer)
    #         self.main_tokenizer.add_tokens([self.compression_token])
    #         if len(self.main_tokenizer) > old_size:
    #             self.main_model.resize_token_embeddings(len(self.main_tokenizer))

    #     if self.compression_token not in self.auxiliary_tokenizer.get_vocab():
    #         print("添加压缩token到辅助模型tokenizer")
    #         old_size = len(self.auxiliary_tokenizer)
    #         self.auxiliary_tokenizer.add_tokens([self.compression_token])
    #         if len(self.auxiliary_tokenizer) > old_size:
    #             self.auxiliary_model.resize_token_embeddings(len(self.auxiliary_tokenizer))

    #     auxiliary_texts = []
    #     formatted_main_inputs = []

    #     for i, (text, question) in enumerate(zip(contexts, questions)):
    #         # 计算文本token长度用于确定压缩token数量
    #         tokens = self.auxiliary_tokenizer.encode(text)
    #         text_length = len(tokens)
    #         if compress_mode == 'fix':
    #             num_compression_tokens = max(text_length // compress_ratio, 1)
    #         elif compress_mode == 'small_threshold':
    #             if text_length <= 256:
    #                 num_compression_tokens = max(256 // compress_ratio, 1)
    #             elif text_length <= 512:
    #                 num_compression_tokens = max(512 // compress_ratio, 1)
    #             elif text_length <= 1024:
    #                 num_compression_tokens = max(1024 // compress_ratio, 1)
    #             elif text_length <= 2048:
    #                 num_compression_tokens = max(2048 // compress_ratio, 1)
    #             elif text_length <= 4096:
    #                 num_compression_tokens = max(4096 // compress_ratio, 1)
    #             elif text_length <= 8192:
    #                 num_compression_tokens = max(8192 // compress_ratio, 1)
    #             else:
    #                 print(f"Warning: length = {text_length} > 1024, using default compress ratio")
    #                 num_compression_tokens = max(text_length // compress_ratio, 1)
    #         else:
    #             raise ValueError(f"未知的compress_mode: {compress_mode}")

    #         # 构造压缩token字符串
    #         compression_tokens_str = " ".join([self.compression_token] * num_compression_tokens)

    #         # 为辅助模型准备输入 - 文档+问题
    #         messages_aux = [
    #             {
    #                 "role": "system",
    #                 "content": system_prompt_auxiliary_pretrain_second.format(num=num_compression_tokens),
    #             },
    #             {
    #                 "role": "user",
    #                 "content": user_prompt_auxiliary_pretrain_second.format(document=text, question=question),
    #             },
    #             {
    #                 "role": "assistant",
    #                 "content": assistant_prompt_auxiliary_pretrain_second.format(CPS_tokens=compression_tokens_str),
    #             },
    #         ]
    #         aux_input = self.auxiliary_tokenizer.apply_chat_template(
    #             messages_aux, tokenize=False
    #         )
    #         auxiliary_texts.append(aux_input)

    #         # 主模型输入：只用压缩token，输出为目标知识内容（推理/答案）
    #         messages_main = [
    #             {
    #                 "role": "user",
    #                 "content": user_prompt_main_pretrain_second.format(compressed_information=compression_tokens_str),
    #             }
    #         ]
    #         main_input = self.main_tokenizer.apply_chat_template(
    #             messages_main,
    #             tokenize=False,
    #             add_generation_prompt=True,
    #             enable_thinking=False
    #         )
    #         formatted_main_inputs.append(main_input)

    #     # 批量tokenize辅助模型输入
    #     tokenized_auxiliary_inputs = self.auxiliary_tokenizer(
    #         auxiliary_texts,
    #         padding=True,
    #         return_tensors="pt",
    #         max_length=8192,
    #         truncation=True
    #     )

    #     # 批量tokenize主模型输入
    #     tokenized_main_inputs = self.main_tokenizer(
    #         formatted_main_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #         max_length=8192,
    #         truncation=True
    #     ).to(self.device)

    #     print("开始生成预训练回答(second)...")
    #     generated_ids = self.generate_pretrain(
    #         input_ids=tokenized_main_inputs['input_ids'],
    #         auxiliary_input=tokenized_auxiliary_inputs,
    #         attention_mask=tokenized_main_inputs['attention_mask'],
    #         max_new_tokens=max_new_tokens,
    #         do_sample=do_sample,
    #         temperature=temperature
    #     )

    #     generated_texts = []
    #     for i in range(batch_size):
    #         new_tokens = generated_ids[i]
    #         generated_text = self.main_tokenizer.decode(new_tokens, skip_special_tokens=True)
    #         print(f"[Second]样本 {i+1} 生成的文本: {generated_text[:100]}...")
    #         generated_texts.append(generated_text)

    #     return generated_texts

    def chat_pretrain_second(
            self, 
            contexts: List[str],
            questions: List[str],
            max_new_tokens: int = 500,
            do_sample: bool = True,
            temperature: float = 0.1,
            compress_ratio: int = 8,  # 压缩比例
            compress_mode: str = 'fix',
            batch_size: int = 4  # 新增批处理大小参数
        ) -> List[str]:
        """
        处理预训练对话请求 V2 - 优化后的批处理版本
        辅助模型提供压缩信息，主模型基于压缩信息重建知识文本（文档+问题）
        """
        assert len(contexts) == len(questions), "contexts 和 questions 长度必须一致"
        
        total_samples = len(contexts)
        batch_size = min(batch_size, total_samples)
        
        # 1. 初始化设置
        self.main_tokenizer.padding_side = 'left'
        self.auxiliary_tokenizer.padding_side = 'left'
        
        # 确保压缩token存在（逻辑与第一个函数对齐）
        for tokenizer, model, name in [
            (self.main_tokenizer, self.main_model, "主模型"),
            (self.auxiliary_tokenizer, self.auxiliary_model, "辅助模型")
        ]:
            if self.compression_token not in tokenizer.get_vocab():
                print(f"添加压缩token到{name}tokenizer")
                old_size = len(tokenizer)
                tokenizer.add_tokens([self.compression_token])
                if len(tokenizer) > old_size:
                    model.resize_token_embeddings(len(tokenizer))

        all_generated_texts = []

        # 2. 按批次处理数据
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            curr_contexts = contexts[batch_start:batch_end]
            curr_questions = questions[batch_start:batch_end]
            current_batch_size = len(curr_contexts)

            print(f"[Second] 处理批次 {batch_start//batch_size + 1}: 样本 {batch_start+1}-{batch_end}")

            auxiliary_texts = []
            main_texts = []

            # 3. 构造当前批次的 Prompt
            for text, question in zip(curr_contexts, curr_questions):
                # 计算长度以确定压缩 token 数量
                tokens = self.auxiliary_tokenizer.encode(text)
                text_length = len(tokens)
                
                if compress_mode == 'fix':
                    num_compression_tokens = max(text_length // compress_ratio, 1)
                elif compress_mode == 'small_threshold':
                    # 补齐了 128 阈值，保持与第一个函数逻辑一致
                    if text_length <= 128:
                        num_compression_tokens = max(128 // compress_ratio, 1)
                    elif text_length <= 256:
                        num_compression_tokens = max(256 // compress_ratio, 1)
                    elif text_length <= 512:
                        num_compression_tokens = max(512 // compress_ratio, 1)
                    elif text_length <= 1024:
                        num_compression_tokens = max(1024 // compress_ratio, 1)
                    elif text_length <= 2048:
                        num_compression_tokens = max(2048 // compress_ratio, 1)
                    elif text_length <= 4096:
                        num_compression_tokens = max(4096 // compress_ratio, 1)
                    elif text_length <= 8192:
                        num_compression_tokens = max(8192 // compress_ratio, 1)
                    else:
                        print(f"Warning: length = {text_length} > 8192, using default compress ratio")
                        num_compression_tokens = max(text_length // compress_ratio, 1)
                else:
                    raise ValueError(f"未知的compress_mode: {compress_mode}")

                compression_tokens_str = " ".join([self.compression_token] * num_compression_tokens)

                # 辅助模型输入模板
                messages_aux = [
                    {"role": "system", "content": system_prompt_auxiliary_pretrain_second.format(num=num_compression_tokens)},
                    {"role": "user", "content": user_prompt_auxiliary_pretrain_second.format(document=text, question=question)},
                    {"role": "assistant", "content": assistant_prompt_auxiliary_pretrain_second.format(CPS_tokens=compression_tokens_str)}
                ]
                # 这里加入了 enable_thinking=False 保持一致性
                aux_input = self.auxiliary_tokenizer.apply_chat_template(messages_aux, tokenize=False, enable_thinking=False)
                auxiliary_texts.append(aux_input)

                # 主模型输入模板
                messages_main = [
                    {"role": "user", "content": user_prompt_main_pretrain_second.format(compressed_information=compression_tokens_str)}
                ]
                main_input = self.main_tokenizer.apply_chat_template(
                    messages_main, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                main_texts.append(main_input)

            # 4. 批量 Tokenize
            tokenized_aux_inputs = self.auxiliary_tokenizer(
                auxiliary_texts, padding=True, return_tensors="pt", max_length=10240, truncation=True
            )
            tokenized_main_inputs = self.main_tokenizer(
                main_texts, padding=True, return_tensors="pt", max_length=8192, truncation=True
            ).to(self.device)

            # 5. 生成结果
            print(f"[Second] 开始生成批次 {batch_start//batch_size + 1} 的回答...")
            batch_generated_ids = self.generate_pretrain(
                input_ids=tokenized_main_inputs['input_ids'],
                auxiliary_input=tokenized_aux_inputs,
                attention_mask=tokenized_main_inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature
            )

            # 6. 解码并收集
            for i in range(current_batch_size):
                new_tokens = batch_generated_ids[i]
                generated_text = self.main_tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_generated_texts.append(generated_text)
                
        return all_generated_texts

    def chat_simple_sft(
        self, 
        contexts: List[str],
        questions: List[str],
        max_new_tokens: int = 500,
        do_sample: bool = True,
        temperature: float = 0.1,
        compress_ratio: int = 8,
        compress_mode: str = 'fix',
        instruction_aux: str = None,  # 可选的辅助模型指令
        batch_size: int = 4  # 新增批处理大小参数
    ) -> List[str]:
        """
        处理简单SFT对话请求 - 辅助模型提供压缩信息，主模型根据问题生成答案
        优化版本：使用批处理提高效率
        
        contexts: 文档文本列表
        questions: 问题列表
        max_new_tokens: 最大生成token数
        do_sample: 是否进行采样
        temperature: 温度参数
        compress_ratio: 压缩比例
        compress_mode: 压缩模式
        instruction_aux: 自定义辅助模型指令
        batch_size: 批处理大小
        """
        assert len(contexts) == len(questions), "contexts 和 questions 长度必须一致"
        total_samples = len(contexts)
        batch_size = min(batch_size, total_samples)
        
        # 设置tokenizer的填充方向
        self.main_tokenizer.padding_side = 'left'
        self.auxiliary_tokenizer.padding_side = 'left'
        
        # 确保tokenizer中有压缩token
        if self.compression_token not in self.main_tokenizer.get_vocab():
            print("添加压缩token到主模型tokenizer")
            old_size = len(self.main_tokenizer)
            self.main_tokenizer.add_tokens([self.compression_token])
            if len(self.main_tokenizer) > old_size:
                self.main_model.resize_token_embeddings(len(self.main_tokenizer))

        if self.compression_token not in self.auxiliary_tokenizer.get_vocab():
            print("添加压缩token到辅助模型tokenizer")
            old_size = len(self.auxiliary_tokenizer)
            self.auxiliary_tokenizer.add_tokens([self.compression_token])
            if len(self.auxiliary_tokenizer) > old_size:
                self.auxiliary_model.resize_token_embeddings(len(self.auxiliary_tokenizer))

        all_generated_texts = []
        
        # 按批次处理数据
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_contexts = contexts[batch_start:batch_end]
            batch_questions = questions[batch_start:batch_end]
            current_batch_size = len(batch_contexts)
            
            print(f"处理批次 {batch_start//batch_size + 1}: 样本 {batch_start+1}-{batch_end}")
            
            # 准备当前批次的辅助模型输入和主模型输入
            auxiliary_texts = []
            main_texts = []

            for text, question in zip(batch_contexts, batch_questions):
                # 使用辅助模型系统提示或自定义指令
                if not instruction_aux:
                    instruction_aux_current = system_prompt_auxiliary_pretrain_second
                else:
                    instruction_aux_current = instruction_aux
                
                # 计算文本token长度用于确定压缩token数量
                tokens = self.auxiliary_tokenizer.encode(text)
                text_length = len(tokens)
                
                if compress_mode == 'fix':
                    num_compression_tokens = max(text_length // compress_ratio, 1)
                elif compress_mode == 'small_threshold':
                    if text_length <= 128:
                        num_compression_tokens = max(128 // compress_ratio, 1)
                    elif text_length <= 256:
                        num_compression_tokens = max(256 // compress_ratio, 1)
                    elif text_length <= 512:
                        num_compression_tokens = max(512 // compress_ratio, 1)
                    elif text_length <= 1024:
                        num_compression_tokens = max(1024 // compress_ratio, 1)
                    elif text_length <= 2048: 
                        num_compression_tokens = max(2048 // compress_ratio, 1)
                    elif text_length <= 4096: 
                        num_compression_tokens = max(4096 // compress_ratio, 1)
                    elif text_length <= 8192: 
                        num_compression_tokens = max(8192 // compress_ratio, 1)
                    else:
                        print(f"Warning: length > 8192 ({text_length}), using default compress ratio")
                        num_compression_tokens = max(text_length // compress_ratio, 1)
                else:
                    raise ValueError(f"未知的compress_mode: {compress_mode}")

                # 构造压缩token字符串
                compression_tokens_str = " ".join([self.compression_token] * num_compression_tokens)

                # 为辅助模型准备输入 - 按照原来的SFT模板
                messages_aux = [
                    {"role": "system", "content": instruction_aux_current.format(num=num_compression_tokens)}, 
                    {"role": "user", "content": user_prompt_auxiliary_pretrain_second.format(document=text, question=question)}, 
                    {"role": "assistant", "content": assistant_prompt_auxiliary_pretrain_second.format(CPS_tokens=compression_tokens_str)}
                ]
                
                aux_input = self.auxiliary_tokenizer.apply_chat_template(
                    messages_aux, tokenize=False, enable_thinking=False
                )
                auxiliary_texts.append(aux_input)

                # 主模型输入：推理时只需要用户部分，不包含assistant回答
                messages_main = [   
                    {"role": "user", "content": user_prompt_main_sft.format(compressed_information=compression_tokens_str, question=question)}
                ]
                
                main_input = self.main_tokenizer.apply_chat_template(
                    messages_main,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                main_texts.append(main_input)

            # 批量tokenize主模型输入 - 使用和chat_pretrain相同的逻辑
            tokenized_main_inputs = self.main_tokenizer(
                main_texts, 
                padding=True, 
                return_tensors="pt",
                max_length=20480,
                truncation=True
            ).to(self.device)
            
            # 批量tokenize辅助模型输入 - 使用和chat_pretrain相同的逻辑
            tokenized_auxiliary_inputs = self.auxiliary_tokenizer(
                auxiliary_texts,
                padding=True,
                return_tensors="pt",
                max_length=10240,
                truncation=True
            )
            
            
            # 运行当前批次的生成
            print(f"开始生成批次 {batch_start//batch_size + 1} 的SFT回答...")
            generated_ids = self.generate_pretrain(
                input_ids=tokenized_main_inputs['input_ids'],
                auxiliary_input=tokenized_auxiliary_inputs,
                attention_mask=tokenized_main_inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature
            )
            
            # 解码当前批次的生成结果
            batch_generated_texts = []
            for i in range(current_batch_size):
                # 获取新生成的token（排除输入部分）
                new_tokens = generated_ids[i]
                
                # 解码这些新token
                generated_text = self.main_tokenizer.decode(new_tokens, skip_special_tokens=True)
                print(f"批次 {batch_start//batch_size + 1} 样本 {i+1} 生成的回答: {generated_text[:100]}...")  # 只打印前100个字符
                print(f"总共{len(new_tokens)}个token")
                batch_generated_texts.append(generated_text)
            
            # 将当前批次的结果添加到总结果中
            all_generated_texts.extend(batch_generated_texts)
            
        
        return all_generated_texts

    def chat_multi_sft(
        self, 
        contexts: List[str] or List[List],
        questions: List[str],
        instruction_users: Optional[List] = None,
        answer_prefixs: Optional[List] = None,
        max_new_tokens: int = 500,
        do_sample: bool = True,
        temperature: float = 0.1,
        compress_ratio: int = 32,
        compress_mode: str = 'fix',
        instruction_aux: str = None,  # 可选的辅助模型指令
        batch_size: int = 4  # 新增批处理大小参数
    ) -> List[str]:
        """
        处理简单SFT对话请求 - 辅助模型提供压缩信息，主模型根据问题生成答案
        优化版本：使用批处理提高效率
        
        contexts: 文档文本列表
        questions: 问题列表
        max_new_tokens: 最大生成token数
        do_sample: 是否进行采样
        temperature: 温度参数
        compress_ratio: 压缩比例
        compress_mode: 压缩模式
        instruction_aux: 自定义辅助模型指令
        batch_size: 批处理大小
        """
        assert len(contexts) == len(questions), "contexts 和 questions 长度必须一致"
        total_samples = len(contexts)
        if total_samples == 0:
            return []
        instruction_users = _normalize_optional_batch_list(
            instruction_users,
            total_samples,
            "instruction_users",
        )
        answer_prefixs = _normalize_optional_batch_list(
            answer_prefixs,
            total_samples,
            "answer_prefixs",
        )
        batch_size = min(batch_size, total_samples)
        
        # 设置tokenizer的填充方向
        self.main_tokenizer.padding_side = 'left'
        self.auxiliary_tokenizer.padding_side = 'left'
        
        # 确保tokenizer中有压缩token
        if self.compression_token not in self.main_tokenizer.get_vocab():
            print("添加压缩token到主模型tokenizer")
            old_size = len(self.main_tokenizer)
            self.main_tokenizer.add_tokens([self.compression_token])
            if len(self.main_tokenizer) > old_size:
                self.main_model.resize_token_embeddings(len(self.main_tokenizer))

        if self.compression_token not in self.auxiliary_tokenizer.get_vocab():
            print("添加压缩token到辅助模型tokenizer")
            old_size = len(self.auxiliary_tokenizer)
            self.auxiliary_tokenizer.add_tokens([self.compression_token])
            if len(self.auxiliary_tokenizer) > old_size:
                self.auxiliary_model.resize_token_embeddings(len(self.auxiliary_tokenizer))

        all_generated_texts = []
        
        # 按批次处理数据
        for batch_start in range(0, total_samples, batch_size):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_contexts = contexts[batch_start:batch_end]
            batch_questions = questions[batch_start:batch_end]
            user_instructions = instruction_users[batch_start:batch_end]
            batch_answer_prefixs = answer_prefixs[batch_start:batch_end]
            current_batch_size = len(batch_contexts)
            
            print(f"处理批次 {batch_start//batch_size + 1}: 样本 {batch_start+1}-{batch_end}")
            
            # 准备当前批次的辅助模型输入和主模型输入
            tokenized_auxiliary_inputs_list = []
            main_texts = []

            for text, question, instruction, answer_prefix in zip(batch_contexts, batch_questions, user_instructions, batch_answer_prefixs):
                # 使用辅助模型系统提示或自定义指令
                if not instruction_aux:
                    instruction_aux_current = system_prompt_auxiliary_pretrain_second
                else:
                    instruction_aux_current = instruction_aux
                
                # 计算文本token长度用于确定压缩token数量
                if isinstance(text, list):
                    text_list = text
                else:
                    text_list = self.chunker.split_text(text)
                num_main_compression_tokens = []
                auxiliary_input_list = []
                current_limit = (131072 - max_new_tokens) // self.chunk_size
                text_list = text_list[:current_limit]
                for single_text in text_list:
                    tokens = self.auxiliary_tokenizer.encode(single_text)
                    text_length = len(tokens)
                    
                    if compress_mode == 'fix':
                        num_compression_tokens = max(text_length // compress_ratio, 1)
                    elif compress_mode == 'small_threshold':
                        if text_length <= 128:
                            num_compression_tokens = max(128 // compress_ratio, 1)
                        elif text_length <= 256:
                            num_compression_tokens = max(256 // compress_ratio, 1)
                        elif text_length <= 512:
                            num_compression_tokens = max(512 // compress_ratio, 1)
                        elif text_length <= 1024:
                            num_compression_tokens = max(1024 // compress_ratio, 1)
                        elif text_length <= 2048: 
                            num_compression_tokens = max(2048 // compress_ratio, 1)
                        elif text_length <= 4096: 
                            num_compression_tokens = max(4096 // compress_ratio, 1)
                        elif text_length <= 8192: 
                            num_compression_tokens = max(8192 // compress_ratio, 1)
                        else:
                            print(f"Warning: length > 8192 ({text_length}), using default compress ratio")
                            num_compression_tokens = max(text_length // compress_ratio, 1)
                    else:
                        raise ValueError(f"未知的compress_mode: {compress_mode}")
                    # 构造压缩token字符串
                    compression_tokens_str = " ".join([self.compression_token] * num_compression_tokens)
                    num_main_compression_tokens.append(num_compression_tokens)
                    # 为辅助模型准备输入 - 按照原来的SFT模板
                    messages_aux = [
                        {"role": "system", "content": instruction_aux_current.format(num=num_compression_tokens)}, 
                        {"role": "user", "content": user_prompt_auxiliary_pretrain_second.format(document=single_text, question=question)}, 
                        {"role": "assistant", "content": assistant_prompt_auxiliary_pretrain_second.format(CPS_tokens=compression_tokens_str)}
                    ]
                    
                    aux_input = self.auxiliary_tokenizer.apply_chat_template(
                        messages_aux, tokenize=False, enable_thinking=False
                    )
                    auxiliary_input_list.append(aux_input)
                # 批量tokenize辅助模型输入
                tokenized_auxiliary_inputs = self.auxiliary_tokenizer(
                    auxiliary_input_list,
                    padding=True,
                    return_tensors="pt",
                    max_length=10240,
                    truncation=True
                )
                tokenized_auxiliary_inputs_list.append(tokenized_auxiliary_inputs)
                print(f"对应的压缩token数：{str(num_main_compression_tokens)}")
                for i, aux_encoding in enumerate(tokenized_auxiliary_inputs['input_ids']):
                    cps_count = (aux_encoding == self.auxiliary_tokenizer.convert_tokens_to_ids(COMPRESSION_TOKEN)).sum().item()
                    print(f"Chunk {i} 实际压缩token数：{cps_count}，预期：{num_main_compression_tokens[i]}")
                temp_main_CPS_tokens = []
                for num_single_compression_tokens in num_main_compression_tokens:
                    temp_main_CPS_tokens.append(" ".join([COMPRESSION_TOKEN] * num_single_compression_tokens))
                main_CPS_tokens = "".join(
                    f" {seg} \n" for i, seg in enumerate(temp_main_CPS_tokens)
                )

                # 主模型输入：推理时只需要用户部分，不包含assistant回答
                if not answer_prefix:
                    answer_prefix = "Your answer of this question is: "
                if not instruction:
                    messages_main = [
                        {"role": "user", "content": user_prompt_main_multi_sft.format(num=len(auxiliary_input_list), compressed_information=main_CPS_tokens, question=question, answer_prefix=answer_prefix)}
                    ]
                else:
                    if isinstance(self.main_tokenizer, LlamaTokenizer) or isinstance(self.main_tokenizer, LlamaTokenizerFast):
                        messages_main = [
                            {"role": "user", "content": f"{instruction}\n\n" + user_prompt_main_multi_sft.format(num=len(auxiliary_input_list), compressed_information=main_CPS_tokens, question=question, answer_prefix=answer_prefix)}
                        ]
                    else:
                        messages_main = [
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": user_prompt_main_multi_sft.format(num=len(auxiliary_input_list), compressed_information=main_CPS_tokens, question=question, answer_prefix=answer_prefix)}
                        ]
                
                main_input = self.main_tokenizer.apply_chat_template(
                    messages_main,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                main_texts.append(main_input)

            # 批量tokenize主模型输入 - 使用和chat_pretrain相同的逻辑
            tokenized_main_inputs = self.main_tokenizer(
                main_texts, 
                padding=True, 
                return_tensors="pt",
                max_length=20480,
                truncation=True
            )
            
            # 运行当前批次的生成
            print(f"开始生成批次 {batch_start//batch_size + 1} 的SFT回答...")
            generated_ids = self.generate_pretrain(
                input_ids=tokenized_main_inputs['input_ids'],
                auxiliary_input=tokenized_auxiliary_inputs_list,
                attention_mask=tokenized_main_inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature
            )
            
            # 解码当前批次的生成结果
            batch_generated_texts = []
            for i in range(current_batch_size):
                # 获取新生成的token（排除输入部分）
                new_tokens = generated_ids[i]
                
                # 解码这些新token
                generated_text = self.main_tokenizer.decode(new_tokens, skip_special_tokens=True)
                print(f"批次 {batch_start//batch_size + 1} 样本 {i+1} 生成的回答: {generated_text[:100]}...")  # 只打印前100个字符
                print(f"总共{len(new_tokens)}个token")
                batch_generated_texts.append(generated_text)
            
            # 将当前批次的结果添加到总结果中
            all_generated_texts.extend(batch_generated_texts)
            
        
        return all_generated_texts

    def generate_pretrain_special(
            self,
            input_ids: torch.Tensor,
            auxiliary_input: Union[List[Dict[str, Any]], Dict[str, torch.Tensor], BatchEncoding],
            attention_mask: Optional[torch.Tensor] = None,
            max_new_tokens: int = 100,
            min_length: int = 0,
            do_sample: bool = False,
            temperature: float = 1.0,
            top_k: int = 50,
            top_p: float = 1.0,
            num_beams: int = 1,
            no_repeat_ngram_size: int = 0,
        ):
            """
            支持批量处理的自回归生成文本，专门用于预训练阶段：
            辅助模型提供压缩信息，主模型根据压缩信息重建原文本
            """
            batch_size = input_ids.shape[0]
            main_device = self.device
            
            # 将输入张量移至主设备
            input_ids = input_ids.to(main_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(main_device)
            
            # 确保输入长度有效
            if isinstance(auxiliary_input, list):
                assert len(auxiliary_input) == batch_size, "auxiliary_input列表长度必须等于batch_size"
            
            # 如果未提供注意力掩码，则创建全1的掩码
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=main_device)
            self.print_gpu_utilization("辅助模型开始压缩前")

            # --- [新增：计时开始] ---
            if torch.cuda.is_available():
                torch.cuda.synchronize() # 同步以确保之前的操作已完成
            start_time = time.perf_counter()

            # 获取辅助模型的压缩embeddings
            compression_embeds = self.get_auxiliary_compression_embeds_special(auxiliary_input)

            if torch.cuda.is_available():
                torch.cuda.synchronize() # 等待 GPU 压缩任务完成
            compress_time = time.perf_counter() - start_time
            # ------------------------

            self.print_gpu_utilization(f"辅助模型压缩完成 (耗时: {compress_time:.4f}s)")
            # 获取主模型的输入embeddings
            inputs_embeds = self.main_model.get_input_embeddings()(input_ids)
            self.compression_token_id = self.main_tokenizer.convert_tokens_to_ids(self.compression_token)
            # ------------------- 新增核对功能开始 -------------------
            for i in range(batch_size):
                # 找主模型输入的压缩token位置
                compression_positions = (input_ids[i] == self.compression_token_id).nonzero(as_tuple=True)[0]
                num_compression_tokens = len(compression_positions)
                # 获取辅助模型输出
                num_aux_embeds = 0
                if i < len(compression_embeds) and compression_embeds[i] is not None:
                    num_aux_embeds = compression_embeds[i].size(0)
                print(f"[核对] 样本 {i}: 主模型输入压缩token数: {num_compression_tokens}，辅助模型输出压缩embedding数: {num_aux_embeds}")
                if num_compression_tokens != num_aux_embeds:
                    print(f"[警告] 样本 {i} 的主模型输入压缩token数 ({num_compression_tokens}) 与辅助模型输出embedding数 ({num_aux_embeds}) 不一致！")
            # 将压缩token替换为辅助模型的压缩信息
            compression_token_positions = (input_ids == self.compression_token_id).nonzero(as_tuple=True)
            for i in range(batch_size):
                batch_positions = (compression_token_positions[0] == i)
                if torch.any(batch_positions):
                    seq_positions = compression_token_positions[1][batch_positions]
                    if i < len(compression_embeds) and compression_embeds[i] is not None:
                        # 将压缩embeddings映射到主模型的embedding空间
                        if compression_embeds[i].device != main_device:
                            compression_embeds[i] = compression_embeds[i].to(main_device)
                        projected_embeds = self.projector(compression_embeds[i].to(inputs_embeds.dtype))
                        # 对每个压缩token位置替换对应的embedding
                        for j, pos in enumerate(seq_positions):
                            if j < projected_embeds.size(0):
                                inputs_embeds[i, pos] = projected_embeds[j].to(main_device)
                            else:
                                # 如果压缩token数量多于提供的embedding，使用最后一个
                                inputs_embeds[i, pos] = projected_embeds[-1].to(main_device)
            self.print_gpu_utilization("映射完成后，给主模型推理前")
            # --- [新增：显存清理点] ---
            # 1. 显式删除辅助模型相关的张量
            # 这里的 auxiliary_input 通常包含大量的 input_ids, attention_mask
            del auxiliary_input
            
            # 2. 删除压缩 embeddings。
            # 重点：如果 compression_embeds 是从辅助模型 output 直接切片来的，
            # 它可能还带着整个辅助模型的计算图 (Grad Fn)，必须确保它彻底消失。
            if 'compression_embeds' in locals():
                del compression_embeds
            
            # 3. 删除投影后的临时变量
            if 'projected_embeds' in locals():
                del projected_embeds

            # 4. 强制触发 Python 垃圾回收并清理 CUDA 缓存
            # 注意：这会带来几十毫秒的延迟，但能确保主模型 generate 时有最大的连续显存空间
            import gc
            gc.collect()
            torch.cuda.empty_cache() 
            # --------------------------
            # 准备生成参数
            generate_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "min_length": min_length,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "num_beams": num_beams,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "pad_token_id": self.main_tokenizer.pad_token_id, 
                "eos_token_id": self.main_tokenizer.eos_token_id,  
                "use_cache": True,  # 推荐添加
            }
            # 直接调用主模型的generate函数
            with torch.no_grad():
                generated_tokens = self.main_model.generate(**generate_kwargs)
            self.print_gpu_utilization("主模型推理后")
            return generated_tokens, compress_time
    
    def chat_multi_sft_ttft(
        self, 
        context: str, # 假设测试单条
        question: str,
        max_new_tokens: int = 50, # 保持与原逻辑一致的 limit 计算
        compress_ratio: int = 32,
        compress_mode: str = 'small_threshold',
        instruction_aux: str = None,
        answer_prefix: str = "Your answer of this question is: ",
    ) -> float:
        """
        严格对齐 chat_multi_sft 逻辑的 TTFT 测量
        """
        
        # --- [初始化阶段] 不计入 TTFT ---
        self.main_tokenizer.padding_side = 'left'
        self.auxiliary_tokenizer.padding_side = 'left'
        print(f"辅助模型实现方式: {self.auxiliary_model.config._attn_implementation}")
        # 辅助函数：封装所有推理逻辑，以便多次调用（Warm-up + 正式测试）
        def run_single_inference():
            # 1. 辅助模型指令准备 (完全对齐原逻辑)
            if not instruction_aux:
                instruction_aux_current = system_prompt_auxiliary_pretrain_second
            else:
                instruction_aux_current = instruction_aux
            
            # 2. 文本切分与 Limit 限制
            if isinstance(context, list):
                text_list = context
            else:
                text_list = self.chunker.split_text(context)
            
            
            auxiliary_input_list = []
            num_main_compression_tokens = []
            
            # 3. 逐 Chunk 处理 (对齐原逻辑中的 for single_text in text_list)
            for single_text in text_list:
                tokens = self.auxiliary_tokenizer.encode(single_text)
                text_length = len(tokens)
                
                # 严格复现你的 compress_mode 逻辑
                if compress_mode == 'fix':
                    num_tokens = max(text_length // compress_ratio, 1)
                elif compress_mode == 'small_threshold':
                    if text_length <= 128: num_tokens = max(128 // compress_ratio, 1)
                    elif text_length <= 256: num_tokens = max(256 // compress_ratio, 1)
                    elif text_length <= 512: num_tokens = max(512 // compress_ratio, 1)
                    elif text_length <= 1024: num_tokens = max(1024 // compress_ratio, 1)
                    elif text_length <= 2048: num_tokens = max(2048 // compress_ratio, 1)
                    elif text_length <= 4096: num_tokens = max(4096 // compress_ratio, 1)
                    elif text_length <= 8192: num_tokens = max(8192 // compress_ratio, 1)
                    else: num_tokens = max(text_length // compress_ratio, 1)
                
                num_main_compression_tokens.append(num_tokens)
                compression_tokens_str = " ".join([self.compression_token] * num_tokens)
                
                # 辅助模型消息模板
                messages_aux = [
                    {"role": "system", "content": instruction_aux_current.format(num=num_tokens)}, 
                    {"role": "user", "content": user_prompt_auxiliary_pretrain_second.format(document=single_text, question=question)}, 
                    {"role": "assistant", "content": assistant_prompt_auxiliary_pretrain_second.format(CPS_tokens=compression_tokens_str)}
                ]
                aux_input = self.auxiliary_tokenizer.apply_chat_template(messages_aux, tokenize=False, enable_thinking=False)
                auxiliary_input_list.append(aux_input)

            # 4. 辅助模型批量 Tokenize
            tokenized_auxiliary_inputs = self.auxiliary_tokenizer(
                auxiliary_input_list, padding=True, return_tensors="pt", max_length=12800, truncation=True
            ).to(self.device)
            
            # 5. 主模型输入构造 (完全对齐原逻辑的 main_CPS_tokens 拼接)
            temp_main_CPS_tokens = [" ".join([self.compression_token] * n) for n in num_main_compression_tokens]
            main_CPS_tokens = "".join(f" {seg} \n" for seg in temp_main_CPS_tokens)

            messages_main = [
                {"role": "user", "content": user_prompt_main_multi_sft.format(
                    num=len(auxiliary_input_list), 
                    compressed_information=main_CPS_tokens, 
                    question=question, 
                    answer_prefix=answer_prefix
                )}
            ]
            
            main_input = self.main_tokenizer.apply_chat_template(
                messages_main, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            
            tokenized_main_inputs = self.main_tokenizer(
                [main_input], padding=True, return_tensors="pt", max_length=25600, truncation=True
            ).to(self.device)

            # 6. 调用 generate_pretrain (只生成 1 个 token 来测 TTFT)
            with torch.no_grad():
                _, compress_time = self.generate_pretrain_special(
                    input_ids=tokenized_main_inputs['input_ids'],
                    auxiliary_input=[tokenized_auxiliary_inputs], # 包装成 List 匹配原逻辑
                    attention_mask=tokenized_main_inputs['attention_mask'],
                    max_new_tokens=1, # 核心：只跑 Prefill + 1 token
                    do_sample=False
                )
            return compress_time

        # --- [执行测量] ---
        
        # 1. 环境清理：确保测量的是当前长度的增量峰值
        torch.cuda.empty_cache() 
        torch.cuda.reset_peak_memory_stats(self.device) # 关键：重置水位线
        torch.cuda.synchronize()
        
        # 2. 正式计时与显存监控
        start_time = time.perf_counter()
        
        compress_time = run_single_inference()
        
        torch.cuda.synchronize() # 确保 GPU 计算全部完成
        ttft_latency = time.perf_counter() - start_time
        
        # 3. 获取显存峰值 (单位: GB)
        peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        
        return ttft_latency, peak_memory, compress_time
    
    def print_gpu_utilization(self, stage: str):
        torch.cuda.synchronize() # 确保之前的计算已完成
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        print(f"[{stage}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {peak_memory:.2f}GB")

    def _process_batch_encoding_special(self, batch_encoding) -> List[torch.Tensor]:
        """处理直接的batch_encoding对象"""
        cps_token_id = self.auxiliary_tokenizer.convert_tokens_to_ids(COMPRESSION_TOKEN)
        aux_device = self.aux_devices[0]
        
        batch_input = {
            'input_ids': batch_encoding['input_ids'].to(aux_device),
            'attention_mask': batch_encoding['attention_mask'].to(aux_device)
        }
        
        contains_cps = (batch_input['input_ids'] == cps_token_id).any().item()
        
        # --- 优化点 1: 去掉 output_hidden_states=True ---
        with torch.no_grad() if self.frozen_auxiliary else torch.enable_grad():
            outputs = self.auxiliary_model.model(**batch_input)
        # --- 优化点 2: 直接取 last_hidden_state ---
        # outputs.last_hidden_state 的内容等价于之前的 outputs.hidden_states[-1]
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_length, hidden_dim]

        compression_embeds = []
        
        if not contains_cps:
            for i in range(batch_input['input_ids'].size(0)):
                last_valid_pos = batch_input['attention_mask'][i].sum() - 1
                # 使用 .clone() 确保即使后面清理了 hidden_states，提取出的向量依然有效
                last_token_state = hidden_states[i, last_valid_pos].unsqueeze(0).clone()
                compression_embeds.append(last_token_state)
        else:
            for i in range(batch_input['input_ids'].size(0)):
                cps_positions = (batch_input['input_ids'][i] == cps_token_id).nonzero(as_tuple=True)[0]
                
                if len(cps_positions) > 1:
                    seq_cps_hidden_states = []
                    for pos in cps_positions[1:]:
                        # 使用 .clone() 彻底切断与大张量及计算图的引用关系
                        seq_cps_hidden_states.append(hidden_states[i, pos].clone())
                    
                    if seq_cps_hidden_states:
                        seq_cps_tensor = torch.stack(seq_cps_hidden_states, dim=0)
                        compression_embeds.append(seq_cps_tensor)
                    else:
                        last_valid_pos = batch_input['attention_mask'][i].sum() - 1
                        compression_embeds.append(hidden_states[i, last_valid_pos].unsqueeze(0).clone())
                elif len(cps_positions) == 1:
                    last_valid_pos = batch_input['attention_mask'][i].sum() - 1
                    compression_embeds.append(hidden_states[i, last_valid_pos].unsqueeze(0).clone())
                else:
                    last_valid_pos = batch_input['attention_mask'][i].sum() - 1
                    compression_embeds.append(hidden_states[i, last_valid_pos].unsqueeze(0).clone())
        
        for i, result in enumerate(compression_embeds):
            if result is not None and (torch.isnan(result).any() or torch.isinf(result).any()):
                print(f"警告：批次{i}中检测到NaN或无穷大，使用零向量替代")
                compression_embeds[i] = torch.zeros_like(result)
        
        # 清理大对象
        del hidden_states, outputs, batch_input
        
        return compression_embeds

    def get_auxiliary_compression_embeds_special(self, auxiliary_input: Union[List[Dict[str, Any]], Dict[str, torch.Tensor], BatchEncoding]) -> List[torch.Tensor]:
        """
        从辅助模型获取压缩的隐藏状态，提取所有<|CPS|>对应的隐藏状态
        跳过每个样本中的第一个压缩标记(仅作为示例用途)
        
        参数:
        auxiliary_input: 可以是两种格式：
            1. List: 辅助模型的输入列表，每个样本包含预处理的上下文
            2. batch_encoding对象: 直接的tokenize后的batch对象，可以直接给辅助模型
        
        返回:
        compression_embeds: 列表，包含每个样本所有压缩token的embeddings
        """
        # 检查输入格式
        if (isinstance(auxiliary_input, dict) or isinstance(auxiliary_input, BatchEncoding)) and 'input_ids' in auxiliary_input:
            # 情况2: 直接是batch_encoding对象
            return self._process_batch_encoding_special(auxiliary_input)
        elif isinstance(auxiliary_input, list):
            # 情况1: 是列表格式
            if not auxiliary_input or all(x is None for x in auxiliary_input):
                return [None] * len(auxiliary_input)
            return self._process_auxiliary_input_list_special(auxiliary_input)
        else:
            raise ValueError(f"Unsupported auxiliary_input format: {type(auxiliary_input)}")
        
    def _process_auxiliary_input_list_special(self, auxiliary_input: List) -> List[torch.Tensor]:
        """处理列表格式的auxiliary_input（显存优化版）"""
        batch_size = len(auxiliary_input)
        compression_embeds = []
        cps_token_id = self.auxiliary_tokenizer.convert_tokens_to_ids(COMPRESSION_TOKEN)
        aux_device = self.aux_devices[0]

        for batch_idx in range(batch_size):
            # 1. 基础检查
            if auxiliary_input[batch_idx] is None or 'input_ids' not in auxiliary_input[batch_idx]:
                compression_embeds.append(None)
                continue
                
            batch_encoding = auxiliary_input[batch_idx]
            input_ids = batch_encoding['input_ids']
            attention_mask = batch_encoding.get('attention_mask', None)

            # 2. 格式标准化：确保是 Tensor 并移动到辅助设备
            if isinstance(input_ids, list):
                # 处理嵌套列表 [[...]]
                input_ids = torch.tensor(input_ids).to(aux_device)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                else:
                    attention_mask = torch.tensor(attention_mask).to(aux_device)
            else:
                input_ids = input_ids.to(aux_device)
                attention_mask = attention_mask.to(aux_device) if attention_mask is not None else torch.ones_like(input_ids)
            self.print_gpu_utilization("获得了hidden states之前")
            # 3. 统一调用辅助模型（优化：移除 output_hidden_states）
 
            with torch.no_grad() if self.frozen_auxiliary else torch.enable_grad():
                outputs = self.auxiliary_model.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    use_cache=False
                )
                # 直接取最后一层输出 [sub_batch, seq_len, hidden_dim]
                hidden_states = outputs.last_hidden_state
            del outputs
            self.print_gpu_utilization("获得了hidden states之后")
            # 4. 提取 Embedding
            batch_results = []
            for i in range(input_ids.size(0)):
                # 找出当前样本的所有 CPS 位置
                cps_positions = (input_ids[i] == cps_token_id).nonzero(as_tuple=True)[0]
                
                if len(cps_positions) > 1:
                    # 【优化点 1】：使用 .detach().clone() 
                    # 这会申请一块极小的独立内存（仅包含几千个 token），
                    # 从而允许后续 del hidden_states 时释放那个 128k 的大矩阵。
                    seq_cps_embeds = hidden_states[i, cps_positions[1:]].detach().clone()
                    batch_results.append(seq_cps_embeds)
                else:
                    # 回退逻辑
                    last_valid_pos = attention_mask[i].sum() - 1
                    # 【优化点 2】：同样使用 .clone()，并确保维度正确
                    # 使用切片 [last_valid_pos:last_valid_pos+1] 代替 unsqueeze(0) 在 clone 时更直观
                    fallback_embed = hidden_states[i, last_valid_pos:last_valid_pos+1].detach().clone()
                    batch_results.append(fallback_embed)
            del hidden_states
            # 5. 合并并检查异常
            if batch_results:
                result = torch.cat(batch_results, dim=0)
                if torch.isnan(result).any() or torch.isinf(result).any():
                    print(f"警告：批次 {batch_idx} 中检测到 NaN/Inf，已重置为零向量")
                    result = torch.zeros_like(result)
                compression_embeds.append(result)
            else:
                compression_embeds.append(None)

            # 显式清理显存


        return compression_embeds
