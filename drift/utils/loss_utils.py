import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_shifted_loss(logits, labels, ignore_index=-100):
    """
    Calculate the shifted loss for language modeling.
    
    Args:
        logits (torch.Tensor): The model's output logits of shape [batch_size, sequence_length, vocab_size]
        labels (torch.Tensor): The target labels of shape [batch_size, sequence_length]
        ignore_index (int): Index to ignore in loss computation (default: -100)
        
    Returns:
        torch.Tensor: The calculated loss
    """
    # 1. Shift predictions and labels (predict next token)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()  # Move labels one step forward
    
    # 2. Flatten the tokens
    vocab_size = logits.size(-1)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    # 3. Calculate loss with CrossEntropy (ignoring padding tokens)
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fct(shift_logits, shift_labels)
    
    return loss

def get_kl_loss(kl_logits, main_logits, main_labels, kl_labels, temperature=1.0, distill_topk=None):
    """
    计算KL散度损失
    
    参数:
        kl_logits: KL输入得到的logits
        main_logits: 主模型的logits
        main_labels: 主模型的标签
        kl_labels: KL输入的标签
        temperature: 温度参数，默认为1.0
        distill_topk: 是否只使用topk个token进行蒸馏，默认为None（使用全部）
    
    返回:
        KL损失张量
    """
    ## make sure the kl_logits and main_logits have the same shape
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    _, _, vocab_size = main_logits.shape

    ## only compute loss in the completion part, not prompt
    
    main_mask = (main_labels != -100).unsqueeze(-1).expand_as(main_logits)  ## batch_size,num_tokens,vocab_size
    main_logits_selected = torch.masked_select(main_logits, main_mask).view(-1, vocab_size)

    kl_mask = (kl_labels != -100).unsqueeze(-1).expand_as(kl_logits)
    kl_logits_selected = torch.masked_select(kl_logits, kl_mask).view(-1, vocab_size)

    if distill_topk is not None:
        _, topk_kl_indices = torch.topk(kl_logits_selected, k=distill_topk, dim=-1)  
        
        kl_logits_selected = torch.gather(kl_logits_selected, 1, topk_kl_indices)  
        main_logits_selected = torch.gather(main_logits_selected, 1, topk_kl_indices) 

    assert kl_logits_selected.shape == main_logits_selected.shape, (
        f"The shape of kl logits is {kl_logits_selected.shape}, while that of main is {main_logits_selected.shape}"
    )

    kl_loss = loss_fct(
        F.log_softmax(main_logits_selected / temperature, dim=-1),
        F.softmax(kl_logits_selected / temperature, dim=-1),
    ) * temperature ** 2
    
    return kl_loss


def calculate_combined_loss(outputs, labels, phase, kl_logits=None, kl_labels=None, kl_weight=0.5, temperature=1.0, distill_topk=None):
    """
    根据训练阶段计算组合损失，并返回分解的损失
    
    返回:
        total_loss: 总损失 (tensor)
        ce_loss: CE损失 (tensor) 
        kl_loss: KL损失 (tensor或0)
    """
    # 计算CE损失
    ce_loss = calculate_shifted_loss(outputs.logits, labels)
    
    if (phase == "simple_sft" or phase == "multi_sft") and kl_weight > 0:
        if kl_logits is not None and kl_labels is not None:
            kl_loss = get_kl_loss(
                kl_logits=kl_logits,
                main_logits=outputs.logits,
                main_labels=labels,
                kl_labels=kl_labels,
                temperature=temperature,
                distill_topk=distill_topk
            )
            
            total_loss = ce_loss + kl_weight * kl_loss
            return total_loss, ce_loss, kl_loss  # 全部返回tensor
        else:
            print("警告: phase为simple_sft但没有提供kl_logits或kl_labels，只使用CE损失")
            return ce_loss, ce_loss, torch.tensor(0.0, device=ce_loss.device)
    elif (phase == "simple_sft" or phase == "multi_sft") and kl_weight == 0:
        if kl_logits is not None and kl_labels is not None:
            kl_loss = get_kl_loss(
                kl_logits=kl_logits,
                main_logits=outputs.logits,
                main_labels=labels,
                kl_labels=kl_labels,
                temperature=temperature,
                distill_topk=distill_topk
            )
            
            total_loss = ce_loss
            return total_loss, ce_loss, kl_loss  # 全部返回tensor
        else:
            print("警告: phase为simple_sft但没有提供kl_logits或kl_labels，只使用CE损失")
            return ce_loss, ce_loss, torch.tensor(0.0, device=ce_loss.device)
    else:
        return ce_loss, ce_loss, torch.tensor(0.0, device=ce_loss.device)