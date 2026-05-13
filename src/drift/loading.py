"""Model loading helpers for DRIFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drift.modeling_drift import DRIFTModel


def resolve_drift_checkpoint_paths(
    checkpoint_path: str | Path,
    *,
    reasoning_model_name_or_path: str | Path | None = None,
    knowledge_model_name_or_path: str | Path | None = None,
) -> dict[str, Path | str]:
    """Resolve standard DRIFT checkpoint subpaths.

    This keeps the legacy checkpoint contract explicit while the loader is
    migrated from `load_mom`.
    """

    checkpoint = Path(checkpoint_path)
    config_path = checkpoint / "drift_config.json"
    checkpoint_config = {}
    if config_path.exists():
        checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))

    reasoning_candidates = [
        checkpoint / "reasoning_model" / "merged_model",
        checkpoint / "main_model" / "merged_model",
    ]
    knowledge_candidates = [
        checkpoint / "knowledge_model" / "merged_model",
        checkpoint / "auxiliary_model" / "merged_model",
    ]

    if reasoning_model_name_or_path is not None:
        reasoning_path: Path | str | None = str(reasoning_model_name_or_path)
    else:
        reasoning_path = next((p for p in reasoning_candidates if p.exists()), None)
        if reasoning_path is None and checkpoint_config.get("reasoning_model_name_or_path"):
            reasoning_path = str(checkpoint_config["reasoning_model_name_or_path"])

    if knowledge_model_name_or_path is not None:
        knowledge_path: Path | str | None = str(knowledge_model_name_or_path)
    else:
        knowledge_path = next((p for p in knowledge_candidates if p.exists()), None)
        if knowledge_path is None and checkpoint_config.get("knowledge_model_name_or_path"):
            knowledge_path = str(checkpoint_config["knowledge_model_name_or_path"])
    projector_path = checkpoint / "projector.pt"

    missing = []
    if reasoning_path is None:
        missing.append("reasoning model")
    if knowledge_path is None:
        missing.append("knowledge model")
    if not projector_path.exists():
        missing.append("projector.pt")
    if missing:
        raise FileNotFoundError(
            f"Could not resolve {', '.join(missing)} from checkpoint: {checkpoint}"
        )

    return {
        "checkpoint": checkpoint,
        "reasoning_model": reasoning_path,
        "knowledge_model": knowledge_path,
        "projector": projector_path,
    }


def load_drift_model(
    checkpoint_path: str | Path,
    reasoning_model_name_or_path: str | Path | None = None,
    knowledge_model_name_or_path: str | Path | None = None,
    num_attention_heads: int = 8,
    device_map_reasoning: str | dict = "auto",
    device_map_knowledge: str | dict = "auto",
    device: str = "cuda:0",
    frozen_reasoning: bool = True,
    frozen_knowledge: bool = True,
    frozen_projector: bool = True,
    chunk_size: int = 4096,
    overlap: int = 200,
    attn_implementation: str | None = None,
    use_layer_norm: bool = False,
) -> DRIFTModel:
    """Load a DRIFT model from a standard or legacy checkpoint directory.

    Supports both public checkpoint names:

    ```text
    reasoning_model/merged_model
    knowledge_model/merged_model
    projector.pt
    ```

    and legacy names:

    ```text
    main_model/merged_model
    auxiliary_model/merged_model
    projector.pt
    ```
    """

    from drift.modeling_drift import DRIFTModel

    paths = resolve_drift_checkpoint_paths(
        checkpoint_path,
        reasoning_model_name_or_path=reasoning_model_name_or_path,
        knowledge_model_name_or_path=knowledge_model_name_or_path,
    )

    model = DRIFTModel(
        main_model_name=str(paths["reasoning_model"]),
        auxiliary_model_name=str(paths["knowledge_model"]),
        num_attention_heads=num_attention_heads,
        device_map_main=device_map_reasoning,
        device_map_auxiliary=device_map_knowledge,
        device=device,
        frozen_main=frozen_reasoning,
        frozen_auxiliary=frozen_knowledge,
        frozen_projector=frozen_projector,
        chunk_size=chunk_size,
        overlap=overlap,
        attn_implementation=attn_implementation,
        use_layer_norm=use_layer_norm,
        projector_path=str(paths["projector"]),
    )
    model.eval()
    return model
