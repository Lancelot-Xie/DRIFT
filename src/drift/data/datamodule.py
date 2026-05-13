"""Dataset loading and dataloader construction for DRIFT training."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from drift.data.collator import drift_collate_fn
from drift.data.preprocessing import (
    convert_to_messages_and_apply_template_lfrp,
    convert_to_messages_and_apply_template_qaft_dc,
    convert_to_messages_and_apply_template_qaft_qa,
)
from drift.data.processor import DRIFTProcessor
from drift.utils.stages import DRIFTStage, normalize_stage


def dataset_format_from_path(path: str | Path) -> str:
    """Return a Hugging Face datasets format name from a local file path."""

    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".json", ".jsonl"}:
        return "json"
    raise ValueError(
        f"Unsupported file format for {path}. Supported formats: .parquet, .json, .jsonl"
    )


class DRIFTDataModule:
    """Prepare DRIFT datasets for LFRP, QAFT-DC, and QAFT-QA training."""

    def __init__(
        self,
        train_file: str,
        val_file: Optional[str] = None,
        main_tokenizer=None,
        aux_tokenizer=None,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        max_length: int = 8192,
        num_workers: int = 4,
        response_template: str = "<|im_start|>assistant\n",
        response_end_marker: str = "<|im_end|>",
        dataset_num_proc: int = 4,
        stage: str | DRIFTStage = DRIFTStage.LFRP,
        compress_ratio: int = 8,
        compress_mode: str = "fix",
        dataloader_shuffle: bool = False,
        pin_memory: bool = True,
        debug_collator: bool = False,
    ):
        self.train_file = train_file
        self.val_file = val_file
        self.main_tokenizer = main_tokenizer
        self.aux_tokenizer = aux_tokenizer
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.response_template = response_template
        self.response_end_marker = response_end_marker
        self.dataset_num_proc = dataset_num_proc
        self.stage = normalize_stage(stage)
        self.compress_ratio = compress_ratio
        self.compress_mode = compress_mode
        self.dataloader_shuffle = dataloader_shuffle
        self.pin_memory = pin_memory
        self.debug_collator = debug_collator
        self.train_dataset = None
        self.val_dataset = None

    def load_file(self, file_path: str) -> Dataset:
        dataset_format = dataset_format_from_path(file_path)
        return load_dataset(dataset_format, data_files=file_path, split="train")

    def prepare_dataset(self, dataset: Dataset, is_train: bool = True) -> Dataset:
        """Apply the stage-specific prompt conversion."""

        _ = is_train
        map_kwargs = {
            "main_tokenizer": self.main_tokenizer,
            "aux_tokenizer": self.aux_tokenizer,
            "compress_ratio": self.compress_ratio,
            "compress_mode": self.compress_mode,
        }

        if self.stage == DRIFTStage.LFRP:
            return dataset.map(
                convert_to_messages_and_apply_template_lfrp,
                fn_kwargs=map_kwargs,
                remove_columns=["context"] if "context" in dataset.column_names else None,
                num_proc=self.dataset_num_proc,
            )

        if self.stage == DRIFTStage.QAFT_DC:
            return dataset.map(
                convert_to_messages_and_apply_template_qaft_dc,
                fn_kwargs=map_kwargs,
                remove_columns=[
                    col
                    for col in ["Document", "Question", "Evidence"]
                    if col in dataset.column_names
                ],
                num_proc=self.dataset_num_proc,
            )

        if self.stage == DRIFTStage.QAFT_QA:
            return dataset.map(
                convert_to_messages_and_apply_template_qaft_qa,
                fn_kwargs=map_kwargs,
                remove_columns=[
                    col
                    for col in ["Document", "Question", "Answer", "Evidence"]
                    if col in dataset.column_names
                ],
                num_proc=self.dataset_num_proc,
            )

        raise ValueError(f"Unsupported DRIFT stage: {self.stage}")

    def setup(self, stage=None) -> None:
        _ = stage
        self.train_dataset = self.prepare_dataset(self.load_file(self.train_file), is_train=True)
        if self.val_file:
            self.val_dataset = self.prepare_dataset(self.load_file(self.val_file), is_train=False)
        else:
            self.val_dataset = None

    def _collate_fn(self):
        processor = DRIFTProcessor(
            aux_tokenizer=self.aux_tokenizer,
            main_tokenizer=self.main_tokenizer,
        )
        return partial(
            drift_collate_fn,
            processor=processor,
            max_length=self.max_length,
            response_template=self.response_template,
            response_end_marker=self.response_end_marker,
            debug=self.debug_collator,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DRIFTDataModule.setup() must be called before train_dataloader().")
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.dataloader_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn(),
        )
