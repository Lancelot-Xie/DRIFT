import os
import torch
import time
import datetime
import numpy as np
from tqdm import tqdm
import gc
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother
import torch.nn as nn 
from drift.training.losses import calculate_shifted_loss, get_kl_loss, calculate_combined_loss
from drift.utils.stages import DRIFTStage, normalize_stage
import csv  # 用于 save_metrics_to_csv
import json  # 用于保存配置
import math

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

def print_optimizer_parameters(optimizer):
    """打印优化器中所有参数的名称、形状和是否需要梯度"""
    print("\n=== 优化器参数信息 ===")
    total_params = 0
    trainable_params = 0
    
    for i, param_group in enumerate(optimizer.param_groups):
        group_params = 0
        print(f"\n参数组 {i}:")
        print(f"  学习率: {param_group['lr']}")
        print(f"  权重衰减: {param_group.get('weight_decay', 0.0)}")
        
        for j, p in enumerate(param_group['params']):
            if p.requires_grad:
                trainable_params += p.numel()
                group_params += p.numel()
            total_params += p.numel()
            
            # 只打印前10个参数，避免输出过多
            if j < 10:
                print(f"  参数 {j}: 形状={p.shape}, 需要梯度={p.requires_grad}, 元素数={p.numel()}")
        
        print(f"  参数组 {i} 可训练参数总数: {group_params:,}")
    
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"可训练参数占比: {trainable_params/total_params*100:.2f}%")


def create_optimizer(model, learning_rate, weight_decay):
    """创建优化器"""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer

def create_scheduler(optimizer, scheduler_type, num_warmup_steps, num_training_steps):
    """创建学习率调度器"""
    if scheduler_type.lower() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    return scheduler

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, output_dir, frozen_main, frozen_auxiliary, frozen_projector, compress_ratio, compress_mode, small_compress_model, is_final=False):
    """保存模型检查点，如果是最终模型则额外保存合并后的完整模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存主模型和tokenizer (如果未冻结)
    if not frozen_main:
        main_model_dir = os.path.join(output_dir, "main_model")
        os.makedirs(main_model_dir, exist_ok=True)
        model.main_model.save_pretrained(main_model_dir)
        model.main_tokenizer.save_pretrained(main_model_dir)
        
        # 如果是最终模型，额外保存合并后的主模型
        if is_final:
            merged_main_dir = os.path.join(main_model_dir, "merged_model")
            save_merged_model(model.main_model, model.main_tokenizer, merged_main_dir, "main")
    
    # 2. 保存辅助模型和tokenizer (如果未冻结)
    if not frozen_auxiliary:
        aux_model_dir = os.path.join(output_dir, "auxiliary_model")
        os.makedirs(aux_model_dir, exist_ok=True)
        model.auxiliary_model.save_pretrained(aux_model_dir)
        model.auxiliary_tokenizer.save_pretrained(aux_model_dir)
        
        # 如果是最终模型，额外保存合并后的辅助模型
        if is_final:
            merged_aux_dir = os.path.join(aux_model_dir, "merged_model")
            save_merged_model(model.auxiliary_model, model.auxiliary_tokenizer, merged_aux_dir, "auxiliary")
    
    # 3. 始终保存投影器，保证checkpoint自包含。冻结时保存的是加载后的当前权重。
    projector_path = os.path.join(output_dir, "projector.pt")
    torch.save(model.projector.state_dict(), projector_path)
    
    # 4. 保存训练状态
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "loss": loss,
        "frozen_main": frozen_main,
        "frozen_auxiliary": frozen_auxiliary,
        "frozen_projector": frozen_projector,
        "compression_token": model.compression_token
    }
    training_state_path = os.path.join(output_dir, "training_state.pt")
    torch.save(training_state, training_state_path)
    
    # 5. 保存配置信息
    config = {
        "main_model_hidden_size": model.main_hidden_size,
        "auxiliary_hidden_size": model.auxiliary_hidden_size,
        "compression_token": model.compression_token,
        "frozen_main": frozen_main,
        "frozen_auxiliary": frozen_auxiliary,
        "frozen_projector":frozen_projector,
        "compress_ratio": compress_ratio,
        "compress_mode": compress_mode,
        "small_compress_model": small_compress_model,
        "reasoning_model_name_or_path": getattr(model, "reasoning_model_name_or_path", None),
        "knowledge_model_name_or_path": getattr(model, "knowledge_model_name_or_path", None),
    }
    
    # 将配置保存为JSON文件
    with open(os.path.join(output_dir, "drift_config.json"), "w") as f:
        import json
        json.dump(config, f, indent=2)
    
    print(f"模型已保存到 {output_dir}")
    if is_final:
        print("已额外保存合并后的完整模型")


def save_merged_model(model, tokenizer, merged_dir, model_type):
    """
    保存合并LoRA权重后的完整模型
    
    Args:
        model: 带有LoRA适配器的模型
        tokenizer: 对应的tokenizer
        merged_dir: 保存合并模型的目录
        model_type: 模型类型标识("main"或"auxiliary")
    """
    try:
        print(f"正在合并并保存{model_type}模型的完整权重...")
        
        # 创建保存目录
        os.makedirs(merged_dir, exist_ok=True)
        
        # 检查模型是否有LoRA适配器
        if hasattr(model, 'merge_and_unload'):
            # 如果是PEFT模型，合并LoRA权重
            print(f"检测到{model_type}模型使用了LoRA，正在合并权重...")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            print(f"{model_type}模型LoRA权重已合并并保存")
        elif hasattr(model, 'peft_config') or hasattr(model, 'base_model'):
            # 另一种PEFT模型的处理方式
            print(f"检测到{model_type}模型使用了PEFT适配器，正在合并权重...")
            try:
                # 尝试使用merge_and_unload方法
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(merged_dir)
            except AttributeError:
                # 如果没有merge_and_unload方法，尝试手动合并
                print(f"使用手动方式合并{model_type}模型权重...")
                from peft import get_peft_model_state_dict
                
                # 获取基础模型
                base_model = model.base_model if hasattr(model, 'base_model') else model.model
                
                # 手动合并权重 (这部分需要根据具体的PEFT类型调整)
                merged_model = base_model
                merged_model.save_pretrained(merged_dir)
        else:
            # 如果不是LoRA模型，直接保存
            print(f"{model_type}模型未使用LoRA适配器，直接保存完整模型...")
            model.save_pretrained(merged_dir)
        
        # 保存tokenizer
        tokenizer.save_pretrained(merged_dir)
        
        print(f"完整的{model_type}模型已保存到: {merged_dir}")
        
    except Exception as e:
        print(f"保存合并后的{model_type}模型时出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果合并失败，至少尝试保存当前状态
        try:
            print(f"尝试保存{model_type}模型的当前状态...")
            model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"{model_type}模型当前状态已保存")
        except Exception as e2:
            print(f"保存{model_type}模型当前状态也失败: {e2}")



def get_kl_weight(step, total_steps, init_weight=1.0, strategy="sqrt"):
    """
    计算KL散度权重的衰减策略
    
    Args:
        step: 当前训练步数 (0-based)
        total_steps: 总训练步数
        init_weight: 初始权重
        strategy: 衰减策略 ("linear", "exp", "cosine", "maintain_decay", "sqrt", "log")
    
    Returns:
        float: 当前步数对应的权重
    """
    # 计算进度 (0到1)
    progress = step / total_steps
    
    if strategy == "linear":
        return init_weight * (1 - progress)
    
    elif strategy in {"exp", "exponential"}:
        # 改进：使用progress而不是固定的1000步间隔
        decay_rate = 0.001  # 可调参数
        return init_weight * math.exp(-decay_rate * step)
    
    elif strategy == "cosine":
        return init_weight * 0.5 * (1 + math.cos(math.pi * progress))
    
    elif strategy == "maintain_decay":
        # 前50%保持固定，后50%线性衰减
        if progress < 0.5:
            return init_weight
        else:
            decay_progress = (progress - 0.5) / 0.5
            return init_weight * (1 - decay_progress)
    
    elif strategy == "sqrt":
        # 新增：平方根衰减 (前期快，后期慢)
        return init_weight * (1 - math.sqrt(progress))
    
    elif strategy == "log":
        # 新增：对数衰减 (前期快，后期慢)
        if progress == 0:
            return init_weight
        return init_weight * (1 - math.log(1 + progress) / math.log(2))
    
    elif strategy == "inverse":
        # 新增：反比衰减
        return init_weight / (1 + progress * 10)  # 10是调节参数
    
    else:
        return init_weight  # 保持原始写法，返回原始权重

def train_model(
    model,
    train_dataloader,
    val_dataloader=None,
    main_device=None,
    config=None
):
    """
    训练DRIFT模型，仅每N步使用验证批次评估性能，不在epoch结束时进行完整验证
    
    参数:
        model: DRIFT模型实例
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器 (可选)
        main_device: 主计算设备
        config: 配置字典，包含训练参数
    
    返回:
        训练统计信息字典
    """
    # 导入绘图库 - 确保matplotlib已安装
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
    import matplotlib.pyplot as plt
    
    # 提取配置
    learning_rate = config.get('learning_rate', 3e-5)
    weight_decay = config.get('weight_decay', 0.01)
    max_epochs = config.get('max_epochs', 2)
    accumulate_grad_batches = config.get('accumulate_grad_batches', 16)
    warmup_ratio = config.get('warmup_ratio', 0.1)
    scheduler_type = config.get('scheduler_type', 'cosine')
    gradient_clip_val = config.get('gradient_clip_val', 1.0)
    label_smoothing_factor = config.get('label_smoothing_factor', 0.1)
    save_steps = config.get('save_steps', 500)
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints/DRIFT')
    log_steps = config.get('log_steps', 10)
    use_wandb = config.get('use_wandb', True)
    if use_wandb and wandb is None:
        raise ImportError("wandb is required when use_wandb=True.")
    frozen_main = config.get('frozen_main', True)
    frozen_auxiliary = config.get('frozen_auxiliary', True)  # 默认冻结辅助模型
    frozen_projector = config.get('frozen_projector', True)
    compress_ratio = config.get('compress_ratio', 8)
    compress_mode = config.get('compress_mode', "fix")
    small_compress_model = config.get("small_compress_model", False)
    phase = normalize_stage(config.get("phase", DRIFTStage.LFRP)).value
    
    # 添加KL损失相关配置
    temperature = config.get('temperature', 1.0)
    distill_topk = config.get('distill_topk', None)
    kl_weight_initial = config.get('kl_weight_initial', 0.5)  # 初始权重，默认使用kl_weight
    enable_kl_weight_decay = config.get('enable_kl_weight_decay', False)  # 是否启用衰减
    kl_weight_decay_strategy = config.get('kl_weight_decay_strategy', 'sqrt')  # 衰减策略
    
    # 添加验证频率配置 - 每N步验证一次，默认为100步
    validation_steps = config.get('validation_steps', 10)
    
    # 添加日志记录相关配置
    log_to_file = config.get('log_to_file', False)
    log_file_path = config.get('log_file_path', os.path.join(checkpoint_dir, 'training_log.txt'))
    
    # 添加绘图相关配置
    plot_metrics = config.get('plot_metrics', True)
    plots_dir = os.path.join(checkpoint_dir, 'plots')
    if not use_wandb and plot_metrics:
        os.makedirs(plots_dir, exist_ok=True)
    
    # 如果需要日志文件，则创建或打开文件
    log_file = None
    if not use_wandb and log_to_file:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        log_file = open(log_file_path, 'a')
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f"\n\n=== 训练开始于 {log_time} ===\n")
        log_file.write(f"配置参数:\n")
        for key, value in config.items():
            log_file.write(f"  {key}: {value}\n")
        log_file.write(f"训练阶段: {phase}\n")
        if phase == DRIFTStage.QAFT_QA.value:
            if enable_kl_weight_decay:
                log_file.write(f"动态KL权重: 初始={kl_weight_initial}, 策略={kl_weight_decay_strategy}\n")
            else:
                log_file.write(f"固定KL权重: {kl_weight_initial}\n")
            log_file.write(f"KL损失温度: {temperature}\n")
            log_file.write(f"蒸馏TopK: {distill_topk}\n")
        log_file.flush()
    
    # 创建保存目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 计算总训练步数
    total_steps = max(1, int(len(train_dataloader) * max_epochs / accumulate_grad_batches))
    print(f"总训练步数: {total_steps}")
    print(f"训练阶段: {phase}")
    if phase == DRIFTStage.QAFT_QA.value:
        if enable_kl_weight_decay:
            print(f"使用动态KL权重衰减: 初始权重={kl_weight_initial}, 策略={kl_weight_decay_strategy}")
        else:
            print(f"使用固定KL权重: {kl_weight_initial}")
        print(f"KL损失温度: {temperature}")
        print(f"蒸馏TopK: {distill_topk if distill_topk else '全部'}")
    else:
        print(f"使用CE损失")
    
    # 创建优化器
    optimizer = create_optimizer(model, learning_rate, weight_decay)

    # 创建学习率调度器
    num_warmup_steps = int(total_steps * warmup_ratio)
    scheduler = create_scheduler(optimizer, scheduler_type, num_warmup_steps, total_steps)
    
    # 创建Label Smoother
    label_smoother = LabelSmoother(epsilon=label_smoothing_factor)
    
    # 验证数据迭代器 - 用于单批次验证
    val_iterator = None
    has_validation_data = False
    if val_dataloader is not None:
        try:
            val_iterator = iter(val_dataloader)
            # 检查验证数据集是否为空
            _ = next(val_iterator)
            val_iterator = iter(val_dataloader)  # 重置迭代器
            has_validation_data = True
        except (StopIteration, TypeError):
            print("警告: 验证数据集为空，将跳过验证")
            has_validation_data = False
    
    # 训练循环
    print("开始训练...")
    global_step = 0
    best_train_loss = float('inf')
    best_val_loss = float('inf')  # 添加最佳验证损失追踪
    training_stats = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'val_losses': [],          # 验证损失列表(按步骤)
        'val_accuracies': [],      # 验证准确率列表(按步骤)
        'val_steps': [],           # 对应的步骤数
        'best_loss': float('inf'),
        'best_val_loss': float('inf'), 
        'total_training_time': 0
    }

    # 创建一个简单的日志记录字典，保存每个step的metrics
    step_metrics = []
    
    # 存储用于绘图的数据（修改版本，包含分离的CE和KL Loss）
    plot_data = {
        'steps': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_steps': [],          
        'val_loss': [],           
        'val_accuracy': [],       
        'learning_rate': [],
        'epoch_boundaries': [],
        # 修改KL Loss相关数据结构
        'train_ce_loss': [],      # 训练CE损失
        'train_kl_loss': [],      # 训练KL损失
        'val_ce_loss': [],        # 验证CE损失
        'val_kl_loss': [],        # 验证KL损失
        'phase': phase            # 记录训练阶段
    }

    # 定义一个执行批次验证的辅助函数（修改版本，返回分离的CE和KL损失）
    def run_batch_validation():
        nonlocal val_iterator, best_val_loss, epoch, global_step  
        
        if not has_validation_data:
            return None, None, None, None  # 修改：返回四个值
            
        # 尝试获取下一个验证批次，如果没有则重置迭代器
        try:
            val_batch = next(val_iterator)
        except (StopIteration, TypeError):
            val_iterator = iter(val_dataloader)
            try:
                val_batch = next(val_iterator)
            except StopIteration:
                print("警告: 验证数据集为空")
                return None, None, None, None
        
        # 保存当前模型状态
        model_was_training = {}
        if not frozen_main:
            model_was_training['main_model'] = model.main_model.training
            model.main_model.eval()
        if not frozen_auxiliary:
            model_was_training['auxiliary_model'] = model.auxiliary_model.training
            model.auxiliary_model.eval()
        if not frozen_projector:
            model_was_training['projector'] = model.projector.training
            model.projector.eval()
        
        val_loss = None
        val_accuracy = None
        val_kl_loss = 0.0
        val_ce_loss = 0.0  # 添加CE损失变量
        
        try:
            with torch.no_grad():  # 不计算梯度
                input_ids = val_batch["input_ids"]
                attention_mask = val_batch.get("attention_mask", None)
                labels = val_batch.get("labels", None)
                auxiliary_input = val_batch.get("auxiliary_input", None)
                
                if labels is not None:
                    labels = labels.to(main_device)
                
                outputs = model(
                    input_ids=input_ids,
                    auxiliary_input=auxiliary_input,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 根据阶段计算kl_logits
                val_kl_logits = None
                val_kl_labels = None
                if phase == DRIFTStage.QAFT_QA.value:
                    kl_input = val_batch.get("kl_input", None)
                    if kl_input is not None:
                        kl_input_ids = kl_input["input_ids"]
                        kl_attention_mask = kl_input.get("attention_mask", None)
                        val_kl_labels = kl_input.get("labels", None).to(main_device)
                        
                        # 使用模型计算kl_logits，不传入auxiliary_input
                        kl_outputs = model(
                            input_ids=kl_input_ids,
                            attention_mask=kl_attention_mask,
                            labels=val_kl_labels
                        )
                        val_kl_logits = kl_outputs.logits
                
                # 计算验证时的动态KL权重
                if phase == DRIFTStage.QAFT_QA.value and enable_kl_weight_decay:
                    current_kl_weight = get_kl_weight(
                        step=global_step, 
                        total_steps=int(total_steps), 
                        init_weight=kl_weight_initial, 
                        strategy=kl_weight_decay_strategy
                    )
                else:
                    current_kl_weight = kl_weight_initial
                
                # 使用组合损失计算函数
                val_loss_result = calculate_combined_loss(
                    outputs=outputs,
                    labels=labels,
                    stage=phase,
                    kl_logits=val_kl_logits,
                    kl_labels=val_kl_labels,
                    kl_weight=current_kl_weight,  # 使用动态权重
                    temperature=temperature,
                    distill_topk=distill_topk
                )
                
                if len(val_loss_result) == 3:
                    batch_val_loss, val_ce_loss_tensor, val_kl_loss_tensor = val_loss_result
                    val_ce_loss = val_ce_loss_tensor.item()  # 直接使用返回的CE损失
                    val_kl_loss = val_kl_loss_tensor.item() if val_kl_loss_tensor != 0 else 0.0
                else:
                    batch_val_loss = val_loss_result
                    val_ce_loss = batch_val_loss.item()  # 对于非QAFT-QA阶段，总损失就是CE损失
                    val_kl_loss = 0.0
                
                val_loss = batch_val_loss.item()
                
                # 计算验证集的token准确率
                logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits", None)
                if logits is not None and labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    predictions = shift_logits.argmax(dim=-1)
                    mask = shift_labels != -100
                    correct_predictions = (predictions == shift_labels) & mask
                    total_tokens = mask.sum()
                    correct_tokens = correct_predictions.sum()
                    val_accuracy = (correct_tokens / total_tokens).item() if total_tokens > 0 else 0.0
                    
                # 如果是最佳验证损失，保存为最佳模型
                improvement_threshold = 0.05  # 0.5%

                if (best_val_loss - val_loss > best_val_loss * improvement_threshold) or best_val_loss == float('inf'):
                    best_val_loss = val_loss
                    print(f"验证损失显著下降至: {best_val_loss:.6f}, 保存最佳模型...")
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        val_loss,
                        f"{checkpoint_dir}/best_model",
                        frozen_main,
                        frozen_auxiliary,
                        frozen_projector,
                        compress_ratio,
                        compress_mode,
                        small_compress_model
                    )

        except Exception as e:
            print(f"批次验证过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 恢复模型的训练状态
            if not frozen_main and 'main_model' in model_was_training and model_was_training['main_model']:
                model.main_model.train()
            if not frozen_auxiliary and 'auxiliary_model' in model_was_training and model_was_training['auxiliary_model']:
                model.auxiliary_model.train()
            if not frozen_projector and 'projector' in model_was_training and model_was_training['projector']:
                model.projector.train()
        
        return val_loss, val_accuracy, val_kl_loss, val_ce_loss  # 修改：返回分离的CE损失

    training_start_time = time.time()
    epoch_boundaries = []  # 用于跟踪每个epoch的结束步骤

    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        epoch_loss = 0.0
        epoch_token_accuracy = 0.0
        epoch_kl_loss = 0.0  # 添加epoch级别的KL损失统计
        num_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        if frozen_main:
            model.main_model.eval()
        else:
            model.main_model.train()

        if frozen_auxiliary:
            model.auxiliary_model.eval()
        else:
            model.auxiliary_model.train()

        if frozen_projector: 
            model.projector.eval()
        else:
            model.projector.train()

        optimizer.zero_grad()
        accumulated_loss = 0.0
        accumulated_token_accuracy = 0.0
        accumulated_kl_loss = 0.0  # 添加累积KL损失
        accumulated_ce_loss = 0.0  # 添加累积CE损失
        cur_accumulate = 0

        for batch_idx, batch in enumerate(progress_bar):
            try:
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", None)
                labels = batch.get("labels", None)
                auxiliary_input = batch.get("auxiliary_input", None)
                
                if labels is not None:
                    labels = labels.to(main_device)
                    
                outputs = model(
                    input_ids=input_ids,
                    auxiliary_input=auxiliary_input,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # 根据阶段计算kl_logits
                train_kl_logits = None
                train_kl_labels = None
                if phase == DRIFTStage.QAFT_QA.value:
                    kl_input = batch.get("kl_input", None)
                    if kl_input is not None:
                        kl_input_ids = kl_input["input_ids"]
                        kl_attention_mask = kl_input.get("attention_mask", None)
                        train_kl_labels = kl_input.get("labels", None).to(main_device)
                        
                        # 使用模型计算kl_logits，不传入auxiliary_input
                        kl_outputs = model(
                            input_ids=kl_input_ids,
                            attention_mask=kl_attention_mask,
                            labels=train_kl_labels
                        )
                        train_kl_logits = kl_outputs.logits
                
                # 计算当前步骤的动态KL权重
                if phase == DRIFTStage.QAFT_QA.value and enable_kl_weight_decay:
                    current_kl_weight = get_kl_weight(
                        step=global_step, 
                        total_steps=int(total_steps), 
                        init_weight=kl_weight_initial, 
                        strategy=kl_weight_decay_strategy
                    )
                else:
                    current_kl_weight = kl_weight_initial

                # 使用组合损失计算函数
                loss_result = calculate_combined_loss(
                    outputs=outputs,
                    labels=labels,
                    stage=phase,
                    kl_logits=train_kl_logits,
                    kl_labels=train_kl_labels,
                    kl_weight=current_kl_weight,  # 使用动态权重
                    temperature=temperature,
                    distill_topk=distill_topk
                )
                
                if len(loss_result) == 3:  # 返回了分解损失
                    loss, ce_loss_tensor, kl_loss_tensor = loss_result
                    ce_loss_val = ce_loss_tensor.item()  # 直接使用返回的CE损失值
                    kl_loss_val = kl_loss_tensor.item() if kl_loss_tensor != 0 else 0.0
                else:  # 只返回总损失（向后兼容）
                    loss = loss_result
                    ce_loss_val = loss.item()  # 对于非QAFT-QA阶段，总损失就是CE损失
                    kl_loss_val = 0.0

                # 计算token准确率
                logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits", None)
                if logits is not None and labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    predictions = shift_logits.argmax(dim=-1)
                    mask = shift_labels != -100
                    correct_predictions = (predictions == shift_labels) & mask
                    total_tokens = mask.sum()
                    correct_tokens = correct_predictions.sum()
                    accuracy = (correct_tokens / total_tokens).item() if total_tokens > 0 else 0.0
                    accumulated_token_accuracy += accuracy

                cur_accumulate += 1
                loss = loss / accumulate_grad_batches
                loss.backward()
                accumulated_loss += loss.item()
                accumulated_kl_loss += kl_loss_val  # 累积KL损失
                accumulated_ce_loss += ce_loss_val  # 累积CE损失

                is_grad_accum_end = (cur_accumulate == accumulate_grad_batches)
                is_last_batch = (batch_idx + 1) == len(train_dataloader)

                # 累积步数达到或最后一批时更新
                if is_grad_accum_end or is_last_batch:
                    # 对于最后一组不足累积步数，loss已按标准缩放，无需再缩放
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        gradient_clip_val
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    # 统计平均loss与准确率，注意用实际cur_accumulate
                    avg_loss = accumulated_loss * accumulate_grad_batches / cur_accumulate
                    avg_accuracy = accumulated_token_accuracy / cur_accumulate
                    avg_kl_loss = accumulated_kl_loss / cur_accumulate  # 平均KL损失
                    avg_ce_loss = accumulated_ce_loss / cur_accumulate  # 平均CE损失
                    
                    # 更新全局步数
                    global_step += 1
                    
                    # 记录当前步骤的metrics（修改版本，包含分离的CE和KL Loss）
                    current_metrics = {
                        "train_loss": avg_loss,
                        "train_token_accuracy": avg_accuracy,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step,
                        "actual_accumulated_batches": cur_accumulate,
                        "phase": phase
                    }
                    
                    # 根据阶段添加CE和KL Loss信息
                    if phase == DRIFTStage.QAFT_QA.value:
                        current_metrics["train_ce_loss"] = avg_ce_loss  # 使用实际返回的CE损失值
                        current_metrics["train_kl_loss"] = avg_kl_loss
                        current_metrics["current_kl_weight"] = current_kl_weight  # 添加当前KL权重
                    else:
                        current_metrics["train_ce_loss"] = avg_loss  # 非QAFT-QA阶段，总损失就是CE损失
                    
                    # 每N步执行一次批次验证（修改版本）
                    if has_validation_data and global_step % validation_steps == 0:
                        val_result = run_batch_validation()
                        if val_result[0] is not None:
                            val_loss, val_accuracy, val_kl_loss, val_ce_loss = val_result  # 修改：接收分离的损失
                            
                            current_metrics["val_loss"] = val_loss
                            current_metrics["val_token_accuracy"] = val_accuracy
                            current_metrics["val_ce_loss"] = val_ce_loss  # 添加验证CE损失
                            
                            if phase == DRIFTStage.QAFT_QA.value:
                                current_metrics["val_kl_loss"] = val_kl_loss
                            
                            # 记录验证指标
                            training_stats['val_losses'].append(val_loss)
                            training_stats['val_accuracies'].append(val_accuracy)
                            training_stats['val_steps'].append(global_step)
                            
                            # 记录用于绘图的验证数据
                            if not use_wandb and plot_metrics:
                                plot_data['val_steps'].append(global_step)
                                plot_data['val_loss'].append(val_loss)
                                plot_data['val_accuracy'].append(val_accuracy)
                                plot_data['val_ce_loss'].append(val_ce_loss)  # 添加验证CE损失
                                if phase == DRIFTStage.QAFT_QA.value:
                                    plot_data['val_kl_loss'].append(val_kl_loss)
                    
                    step_metrics.append(current_metrics)
                    
                    # 保存用于绘图的数据（修改版本）
                    if not use_wandb and plot_metrics:
                        plot_data['steps'].append(global_step)
                        plot_data['train_loss'].append(avg_loss)
                        plot_data['train_accuracy'].append(avg_accuracy)
                        plot_data['learning_rate'].append(scheduler.get_last_lr()[0])
                        
                        # 添加CE和KL Loss数据
                        if phase == DRIFTStage.QAFT_QA.value:
                            plot_data['train_ce_loss'].append(avg_ce_loss)  # 使用实际CE损失
                            plot_data['train_kl_loss'].append(avg_kl_loss)
                        else:
                            plot_data['train_ce_loss'].append(avg_loss)  # 非QAFT-QA阶段CE损失就是总损失
                        
                        # 每100步或配置的步数绘制一次图表
                        if global_step % 100 == 0 or global_step % save_steps == 0:
                            plot_training_metrics(plot_data, plots_dir, global_step, phase)

                    if use_wandb:
                        wandb.log(current_metrics)
                    else:
                        # 如果不使用wandb，则打印metrics
                        log_message = (f"Step {global_step}: Loss={avg_loss:.6f}, "
                                      f"Accuracy={avg_accuracy:.4f}, "
                                      f"LR={scheduler.get_last_lr()[0]:.8f}")
                        
                        # 添加训练阶段信息
                        if phase == DRIFTStage.QAFT_QA.value:
                            log_message += f" (Phase: {phase}, CE Loss: {avg_ce_loss:.6f}, KL Loss: {avg_kl_loss:.6f}"
                            if enable_kl_weight_decay:
                                log_message += f", KL Weight: {current_kl_weight:.6f})"
                            else:
                                log_message += ")"
                        else:
                            log_message += f" (Phase: {phase}, CE Loss: {avg_ce_loss:.6f})"
                        
                        # 如果有验证结果，添加到日志信息中
                        if "val_loss" in current_metrics:
                            log_message += f", Val Loss={current_metrics['val_loss']:.6f}, Val Acc={current_metrics['val_token_accuracy']:.4f}"
                            log_message += f", Val CE Loss={current_metrics['val_ce_loss']:.6f}"
                            if phase == DRIFTStage.QAFT_QA.value and "val_kl_loss" in current_metrics:
                                log_message += f", Val KL Loss={current_metrics['val_kl_loss']:.6f}"
                        
                        print(log_message)
                        
                        # 如果需要保存到文件
                        if log_file:
                            time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            log_file.write(f"[{time_str}] {log_message}\n")
                            log_file.flush()

                    epoch_loss += avg_loss
                    epoch_token_accuracy += avg_accuracy
                    epoch_kl_loss += avg_kl_loss  # 累积epoch级别KL损失
                    num_batches += 1
                    if avg_loss < best_train_loss:
                        best_train_loss = avg_loss
                        training_stats["best_loss"] = best_train_loss

                    accumulated_loss = 0.0
                    accumulated_token_accuracy = 0.0
                    accumulated_kl_loss = 0.0  # 重置累积KL损失
                    accumulated_ce_loss = 0.0  # 重置累积CE损失
                    cur_accumulate = 0

                    if global_step % save_steps == 0:
                        save_checkpoint(
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            global_step,
                            avg_loss,
                            f"{checkpoint_dir}/checkpoint-{global_step}",
                            frozen_main,
                            frozen_auxiliary,
                            frozen_projector,
                            compress_ratio,
                            compress_mode,
                            small_compress_model
                        )
                        # 不使用wandb时，保存中间结果到CSV文件
                        if not use_wandb:
                            metrics_file = os.path.join(checkpoint_dir, f"metrics-step-{global_step}.csv")
                            save_metrics_to_csv(step_metrics, metrics_file)
                            print(f"指标已保存到: {metrics_file}")
                            
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                if (batch_idx + 1) % log_steps == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if num_batches > 0:
            epoch_loss /= num_batches
            epoch_token_accuracy /= num_batches
            epoch_kl_loss /= num_batches  # 平均epoch KL损失

        training_stats['epoch_losses'].append(epoch_loss)
        training_stats['epoch_accuracies'].append(epoch_token_accuracy)
        training_stats['best_val_loss'] = best_val_loss
        
        # 记录epoch边界，用于绘图和统计
        epoch_boundaries.append(global_step)
        if not use_wandb and plot_metrics:
            plot_data['epoch_boundaries'].append(global_step)
        
        # 打印每个epoch的汇总信息
        epoch_summary = f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_token_accuracy:.4f}"
        
        # 添加阶段信息
        if phase == DRIFTStage.QAFT_QA.value:
            epoch_summary += f" (Phase: {phase}, Combined Loss, Avg KL Loss: {epoch_kl_loss:.4f})"
        else:
            epoch_summary += f" (Phase: {phase}, CE Loss)"
        
        # 找出该epoch中最新的验证结果(如果有)
        if has_validation_data and epoch > 0 and len(training_stats['val_steps']) > 0 and len(epoch_boundaries) > 0:
            # 安全获取当前和上一个epoch的边界
            current_boundary = epoch_boundaries[epoch]
            previous_boundary = epoch_boundaries[epoch-1] if epoch > 0 else 0
            
            # 获取当前epoch内的验证结果
            current_epoch_val_data = [(step, loss, acc) 
                                      for step, loss, acc in zip(training_stats['val_steps'], 
                                                               training_stats['val_losses'], 
                                                               training_stats['val_accuracies'])
                                      if previous_boundary < step <= current_boundary]
            
            if current_epoch_val_data:
                # 获取最新的验证结果
                _, latest_val_loss, latest_val_acc = current_epoch_val_data[-1]
                epoch_summary += f", Latest Val Loss: {latest_val_loss:.4f}, Latest Val Acc: {latest_val_acc:.4f}"
        
        print(epoch_summary)
        
        # 保存到日志文件
        if not use_wandb and log_file:
            log_file.write(f"\n=== {epoch_summary} ===\n\n")
            log_file.flush()

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            epoch_loss,
            f"{checkpoint_dir}/epoch-{epoch+1}",
            frozen_main,
            frozen_auxiliary,
            frozen_projector,
            compress_ratio,
            compress_mode,
            small_compress_model
        )
        
        # 不使用wandb时，保存每个epoch的metrics到CSV文件
        if not use_wandb:
            metrics_file = os.path.join(checkpoint_dir, f"metrics-epoch-{epoch+1}.csv")
            save_metrics_to_csv(step_metrics, metrics_file)
            print(f"Epoch {epoch+1} 指标已保存到: {metrics_file}")
            
            # 绘制每个epoch结束时的metrics图表
            if plot_metrics:
                plot_training_metrics(plot_data, plots_dir, f"epoch-{epoch+1}", phase)

        gc.collect()
        torch.cuda.empty_cache()

    save_checkpoint(
        model,
        optimizer,
        scheduler,
        max_epochs-1,
        global_step,
        epoch_loss,
        f"{checkpoint_dir}/final_model",
        frozen_main,
        frozen_auxiliary,
        frozen_projector,
        compress_ratio,
        compress_mode,
        small_compress_model,
        is_final=True
    )

    training_stats['total_training_time'] = time.time() - training_start_time
    
    # 最终保存所有metrics到CSV文件
    if not use_wandb:
        final_metrics_file = os.path.join(checkpoint_dir, "all_metrics.csv")
        save_metrics_to_csv(step_metrics, final_metrics_file)
        print(f"所有训练指标已保存到: {final_metrics_file}")
        
        # 绘制最终的训练指标图表
        if plot_metrics:
            plot_training_metrics(plot_data, plots_dir, "final", phase)
    
    # 关闭日志文件
    if log_file:
        log_file.write(f"\n训练完成! 总用时: {str(datetime.timedelta(seconds=int(training_stats['total_training_time'])))}\n")
        log_file.close()

    print(f"训练完成! 总用时: {str(datetime.timedelta(seconds=int(training_stats['total_training_time'])))}")
    return training_stats


def plot_training_metrics(plot_data, plots_dir, step_or_name, phase=DRIFTStage.LFRP.value):
    """
    Draw training metrics charts with enhanced loss type distinction and KL Loss charts for QAFT-QA stage
    
    Args:
        plot_data: Dictionary containing training and validation metrics
        plots_dir: Directory to save plots
        step_or_name: Current step or name, used for filename
        phase: Training phase (used for loss type labeling)
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
    if not plot_data['steps']:
        return  # No data, don't plot
    
    # Ensure directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # 确定损失类型标题
    if phase == DRIFTStage.QAFT_QA.value:
        loss_type_label = "(CE Loss + KL Loss)"
    else:
        loss_type_label = "(CE Loss)"
    
    # ====================================
    # 1. 绘制组合指标图 (Combined train+val metrics)
    # ====================================
    fig, ax_array = plt.subplots(2, 1, figsize=(12, 10), dpi=120)
    
    # Plot loss curves (both train and val)
    ax1 = ax_array[0]
    ax1.plot(plot_data['steps'], plot_data['train_loss'], 'b-', label='Training Loss')
    
    # If validation loss data exists, plot it
    if 'val_steps' in plot_data and plot_data['val_steps'] and 'val_loss' in plot_data and plot_data['val_loss']:
        ax1.plot(plot_data['val_steps'], plot_data['val_loss'], 'r-', marker='o', markersize=4, label='Validation Loss')
    
    ax1.set_title(f'Loss Curves {loss_type_label}', fontsize=14, pad=10)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Draw vertical lines at epoch boundaries
    for idx, boundary in enumerate(plot_data['epoch_boundaries']):
        ax1.axvline(x=boundary, color='g', linestyle='--', alpha=0.5, label=f'Epoch {idx+1} End' if idx == 0 else "")
    
    # Use integer ticks on x-axis
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Add legend
    ax1.legend(loc='upper right')
    
    # Plot accuracy curves (both train and val)
    ax2 = ax_array[1]
    ax2.plot(plot_data['steps'], plot_data['train_accuracy'], 'g-', label='Training Accuracy')
    
    # If validation accuracy data exists, plot it
    if 'val_steps' in plot_data and plot_data['val_steps'] and 'val_accuracy' in plot_data and plot_data['val_accuracy']:
        ax2.plot(plot_data['val_steps'], plot_data['val_accuracy'], 'm-', marker='o', markersize=4, label='Validation Accuracy')
    
    ax2.set_title('Accuracy Curves', fontsize=14, pad=10)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Draw vertical lines at epoch boundaries
    for idx, boundary in enumerate(plot_data['epoch_boundaries']):
        ax2.axvline(x=boundary, color='g', linestyle='--', alpha=0.5, label=f'Epoch {idx+1} End' if idx == 0 else "")
    
    # Use integer ticks on x-axis
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Add legend
    ax2.legend(loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    combined_plot_filename = f"combined_metrics_{step_or_name}.png"
    plt.savefig(os.path.join(plots_dir, combined_plot_filename))
    plt.close(fig)
    
    # ====================================
    # 2. 绘制训练指标图 (Train metrics only)
    # ====================================
    fig, ax_array = plt.subplots(2, 1, figsize=(12, 10), dpi=120)
    
    # Plot loss curve (train only)
    ax1 = ax_array[0]
    ax1.plot(plot_data['steps'], plot_data['train_loss'], 'b-', label='Training Loss')
    
    ax1.set_title(f'Training Loss {loss_type_label}', fontsize=14, pad=10)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Draw vertical lines at epoch boundaries
    for idx, boundary in enumerate(plot_data['epoch_boundaries']):
        ax1.axvline(x=boundary, color='g', linestyle='--', alpha=0.5, label=f'Epoch {idx+1} End' if idx == 0 else "")
    
    # Use integer ticks on x-axis
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Add legend
    ax1.legend(loc='upper right')
    
    # Plot accuracy curve (train only)
    ax2 = ax_array[1]
    ax2.plot(plot_data['steps'], plot_data['train_accuracy'], 'g-', label='Training Accuracy')
    
    ax2.set_title('Training Accuracy', fontsize=14, pad=10)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Draw vertical lines at epoch boundaries
    for idx, boundary in enumerate(plot_data['epoch_boundaries']):
        ax2.axvline(x=boundary, color='g', linestyle='--', alpha=0.5, label=f'Epoch {idx+1} End' if idx == 0 else "")
    
    # Use integer ticks on x-axis
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Add legend
    ax2.legend(loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    train_plot_filename = f"training_metrics_{step_or_name}.png"
    plt.savefig(os.path.join(plots_dir, train_plot_filename))
    plt.close(fig)
    
    # ====================================
    # 3. 绘制验证指标图 (Validation metrics only, if available)
    # ====================================
    if 'val_steps' in plot_data and plot_data['val_steps'] and len(plot_data['val_steps']) > 0:
        fig, ax_array = plt.subplots(2, 1, figsize=(12, 10), dpi=120)
        
        # Plot validation loss curve
        ax1 = ax_array[0]
        ax1.plot(plot_data['val_steps'], plot_data['val_loss'], 'r-', marker='o', markersize=4, label='Validation Loss')
        
        ax1.set_title(f'Validation Loss {loss_type_label}', fontsize=14, pad=10)
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Draw vertical lines at epoch boundaries
        for idx, boundary in enumerate(plot_data['epoch_boundaries']):
            ax1.axvline(x=boundary, color='g', linestyle='--', alpha=0.5, label=f'Epoch {idx+1} End' if idx == 0 else "")
        
        # Use integer ticks on x-axis
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Add legend
        ax1.legend(loc='upper right')
        
        # Plot validation accuracy curve
        ax2 = ax_array[1]
        ax2.plot(plot_data['val_steps'], plot_data['val_accuracy'], 'm-', marker='o', markersize=4, label='Validation Accuracy')
        
        ax2.set_title('Validation Accuracy', fontsize=14, pad=10)
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Draw vertical lines at epoch boundaries
        for idx, boundary in enumerate(plot_data['epoch_boundaries']):
            ax2.axvline(x=boundary, color='g', linestyle='--', alpha=0.5, label=f'Epoch {idx+1} End' if idx == 0 else "")
        
        # Use integer ticks on x-axis
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Add legend
        ax2.legend(loc='lower right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        val_plot_filename = f"validation_metrics_{step_or_name}.png"
        plt.savefig(os.path.join(plots_dir, val_plot_filename))
        plt.close(fig)
    
    # ====================================
    # ====================================
    # 4. 绘制损失详细信息图表 (仅在QAFT-QA阶段) - 修改后的版本
    # ====================================
    if phase == DRIFTStage.QAFT_QA.value and 'train_ce_loss' in plot_data and plot_data['train_ce_loss']:
        fig, ax_array = plt.subplots(2, 1, figsize=(12, 10), dpi=120)
        
        # 上子图：CE Loss (训练和验证)
        ax1 = ax_array[0]
        ax1.plot(plot_data['steps'], plot_data['train_ce_loss'], 'blue', label='Training CE Loss')
        
        # 如果有验证CE损失数据，绘制它
        if 'val_steps' in plot_data and plot_data['val_steps'] and 'val_ce_loss' in plot_data and plot_data['val_ce_loss']:
            ax1.plot(plot_data['val_steps'], plot_data['val_ce_loss'], 'lightblue', marker='o', markersize=4, label='Validation CE Loss')
        
        ax1.set_title('Cross Entropy Loss', fontsize=14, pad=10)
        ax1.set_xlabel('Training Steps', fontsize=12)
        ax1.set_ylabel('CE Loss', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制epoch边界线
        for idx, boundary in enumerate(plot_data['epoch_boundaries']):
            ax1.axvline(x=boundary, color='g', linestyle='--', alpha=0.5, label=f'Epoch {idx+1} End' if idx == 0 else "")
        
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.legend(loc='upper right')
        
        # 下子图：KL Loss (训练和验证)
        ax2 = ax_array[1]
        if 'train_kl_loss' in plot_data and plot_data['train_kl_loss']:
            ax2.plot(plot_data['steps'], plot_data['train_kl_loss'], 'red', label='Training KL Loss')
        
        # 如果有验证KL损失数据，绘制它
        if 'val_steps' in plot_data and plot_data['val_steps'] and 'val_kl_loss' in plot_data and plot_data['val_kl_loss']:
            ax2.plot(plot_data['val_steps'], plot_data['val_kl_loss'], 'orange', marker='o', markersize=4, label='Validation KL Loss')
        
        ax2.set_title('KL Divergence Loss', fontsize=14, pad=10)
        ax2.set_xlabel('Training Steps', fontsize=12)
        ax2.set_ylabel('KL Loss', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制epoch边界线
        for idx, boundary in enumerate(plot_data['epoch_boundaries']):
            ax2.axvline(x=boundary, color='g', linestyle='--', alpha=0.5, label=f'Epoch {idx+1} End' if idx == 0 else "")
        
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.legend(loc='upper right')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形 - 修改文件名
        loss_details_filename = f"loss_details_{step_or_name}.png"
        plt.savefig(os.path.join(plots_dir, loss_details_filename))
        plt.close(fig)
    
    # ====================================
    # 5. 绘制学习率曲线 (Learning Rate)
    # ====================================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.plot(plot_data['steps'], plot_data['learning_rate'], 'r-', label='Learning Rate')
    ax.set_title('Learning Rate Curve', fontsize=14, pad=10)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Draw vertical lines at epoch boundaries
    for idx, boundary in enumerate(plot_data['epoch_boundaries']):
        ax.axvline(x=boundary, color='b', linestyle='--', alpha=0.5, label=f'Epoch {idx+1} End' if idx == 0 else "")
    
    # Use integer ticks on x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Add legend
    ax.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    lr_plot_filename = f"learning_rate_{step_or_name}.png"
    plt.savefig(os.path.join(plots_dir, lr_plot_filename))
    plt.close(fig)
    
    # 打印保存的文件信息
    saved_files = [
        f"组合图({os.path.join(plots_dir, combined_plot_filename)})",
        f"训练图({os.path.join(plots_dir, train_plot_filename)})",
        f"学习率图({os.path.join(plots_dir, lr_plot_filename)})"
    ]
    
    if 'val_steps' in plot_data and plot_data['val_steps'] and len(plot_data['val_steps']) > 0:
        saved_files.append(f"验证图({os.path.join(plots_dir, val_plot_filename)})")
    
    if phase == DRIFTStage.QAFT_QA.value and 'train_ce_loss' in plot_data and plot_data['train_ce_loss']:
        saved_files.append(f"损失详情图({os.path.join(plots_dir, loss_details_filename)})")  # 修改名称
    
    print(f"指标已保存: {', '.join(saved_files)}")

def save_metrics_to_csv(metrics_list, filepath):
    """
    将指标保存为CSV文件，处理不同记录中可能包含不同字段的情况
    
    参数:
        metrics_list: 包含指标的字典列表
        filepath: CSV文件保存路径
    """
    import csv
    
    if not metrics_list:
        return
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 收集所有可能出现的字段名
    all_fieldnames = set()
    for metrics in metrics_list:
        all_fieldnames.update(metrics.keys())
    
    # 转换为有序列表，确保关键字段排在前面
    important_fields = ["global_step", "epoch", "train_loss", "train_token_accuracy", 
                        "val_loss", "val_token_accuracy", "learning_rate", "phase",
                        "train_ce_loss", "train_kl_loss", "val_kl_loss"]  # 添加KL Loss字段
    
    # 确保重要字段首先出现（如果它们存在于数据中）
    fieldnames = [field for field in important_fields if field in all_fieldnames]
    
    # 添加其他字段
    remaining_fields = sorted(field for field in all_fieldnames if field not in important_fields)
    fieldnames.extend(remaining_fields)
    
    # 写入CSV文件
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='', extrasaction='ignore')
        writer.writeheader()
        
        for metrics in metrics_list:
            # 确保每一行的字典包含所有字段，缺少的字段设为空字符串
            row = {field: metrics.get(field, '') for field in fieldnames}
            writer.writerow(row)
            
    print(f"指标已保存到: {filepath}")
