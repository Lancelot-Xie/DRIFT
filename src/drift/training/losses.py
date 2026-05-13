"""Loss functions for DRIFT training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from drift.utils.stages import DRIFTStage, normalize_stage


def calculate_shifted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Calculate next-token language modeling loss."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = logits.size(-1)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return loss_fct(shift_logits, shift_labels)


def get_kl_loss(
    kl_logits: torch.Tensor,
    main_logits: torch.Tensor,
    main_labels: torch.Tensor,
    kl_labels: torch.Tensor,
    temperature: float = 1.0,
    distill_topk: int | None = None,
) -> torch.Tensor:
    """Calculate KL distillation loss over completion tokens."""

    loss_fct = nn.KLDivLoss(reduction="batchmean")
    _, _, vocab_size = main_logits.shape

    main_mask = (main_labels != -100).unsqueeze(-1).expand_as(main_logits)
    main_logits_selected = torch.masked_select(main_logits, main_mask).view(
        -1, vocab_size
    )

    kl_mask = (kl_labels != -100).unsqueeze(-1).expand_as(kl_logits)
    kl_logits_selected = torch.masked_select(kl_logits, kl_mask).view(-1, vocab_size)

    if distill_topk is not None:
        _, topk_kl_indices = torch.topk(kl_logits_selected, k=distill_topk, dim=-1)
        kl_logits_selected = torch.gather(kl_logits_selected, 1, topk_kl_indices)
        main_logits_selected = torch.gather(main_logits_selected, 1, topk_kl_indices)

    if kl_logits_selected.shape != main_logits_selected.shape:
        raise ValueError(
            "KL logits and main logits must have the same selected shape. "
            f"Got {kl_logits_selected.shape} and {main_logits_selected.shape}."
        )

    return loss_fct(
        F.log_softmax(main_logits_selected / temperature, dim=-1),
        F.softmax(kl_logits_selected / temperature, dim=-1),
    ) * temperature**2


def calculate_combined_loss(
    outputs,
    labels: torch.Tensor,
    stage: str | DRIFTStage,
    kl_logits: torch.Tensor | None = None,
    kl_labels: torch.Tensor | None = None,
    kl_weight: float = 0.0,
    temperature: float = 1.0,
    distill_topk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate CE loss and QAFT-QA KL metric.

    KL is primarily used as a diagnostic metric comparing the distribution from
    compressed implicit facts with the distribution from annotated evidence. By
    default it is not added to the training objective.
    """

    ce_loss = calculate_shifted_loss(outputs.logits, labels)
    normalized_stage = normalize_stage(stage)

    if normalized_stage != DRIFTStage.QAFT_QA:
        return ce_loss, ce_loss, torch.tensor(0.0, device=ce_loss.device)

    if kl_logits is None or kl_labels is None:
        return ce_loss, ce_loss, torch.tensor(0.0, device=ce_loss.device)

    kl_loss = get_kl_loss(
        kl_logits=kl_logits,
        main_logits=outputs.logits,
        main_labels=labels,
        kl_labels=kl_labels,
        temperature=temperature,
        distill_topk=distill_topk,
    )

    return ce_loss + kl_weight * kl_loss, ce_loss, kl_loss
