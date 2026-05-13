"""Training stage names and legacy aliases."""

from __future__ import annotations

from enum import Enum


class DRIFTStage(str, Enum):
    LFRP = "lfrp"
    QAFT_DC = "qaft_dc"
    QAFT_QA = "qaft_qa"


_STAGE_ALIASES = {
    "lfrp": DRIFTStage.LFRP,
    "pretrain_1": DRIFTStage.LFRP,
    "qaft_dc": DRIFTStage.QAFT_DC,
    "pretrain_2": DRIFTStage.QAFT_DC,
    "qaft_qa": DRIFTStage.QAFT_QA,
    "simple_sft": DRIFTStage.QAFT_QA,
    "multi_sft": DRIFTStage.QAFT_QA,
}


def normalize_stage(stage: str | DRIFTStage) -> DRIFTStage:
    """Normalize paper stage names and legacy phase names."""

    if isinstance(stage, DRIFTStage):
        return stage
    try:
        return _STAGE_ALIASES[stage]
    except KeyError as exc:
        valid = ", ".join(sorted(_STAGE_ALIASES))
        raise ValueError(f"Unknown DRIFT stage '{stage}'. Valid stages: {valid}") from exc


def to_legacy_phase(stage: str | DRIFTStage) -> str:
    """Return the legacy phase string used by the original training code."""

    normalized = normalize_stage(stage)
    return {
        DRIFTStage.LFRP: "pretrain_1",
        DRIFTStage.QAFT_DC: "pretrain_2",
        DRIFTStage.QAFT_QA: "simple_sft",
    }[normalized]
