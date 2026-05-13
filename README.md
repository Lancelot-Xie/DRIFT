# DRIFT

Official implementation of the paper:
**[Decoupled Reasoning with Implicit Fact Tokens (DRIFT): A Dual-Model
Framework for Efficient Long-Context Inference](https://arxiv.org/abs/2602.10021)**
(arXiv 2026).

DRIFT decouples reading from reasoning by preventing the reasoning model from
directly processing raw long-context inputs. Instead, a knowledge model
compresses long documents into implicit fact token embeddings that are then used
by the reasoning model for answer generation.

Across multiple long-context benchmarks, DRIFT improves inference efficiency
through aggressive context compression while preserving strong reasoning
performance.

<p align="center">
  <img src="assets/DRIFT.png" width="90%"><br>
  <em>Figure 1: The overall architecture of the DRIFT framework.</em>
</p>

## News

- **2026-02-12**: Project repository initialized. We are cleaning up the code
  for public release.
- **2026-03-05**: Released the **LFRP** and **QAFT** datasets on Hugging Face:
  [LFRP data](https://huggingface.co/datasets/SII-LancelotXie/DRIFT_LFRP) and
  [QAFT data](https://huggingface.co/datasets/SII-LancelotXie/DRIFT_QAFT).
  Released the data synthesis pipeline at `data_generation/generate_qa.py` for
  generating QA-Evidence triplets.

## Open Source Roadmap

To ensure reproducibility and code quality, we are releasing the project in
stages:

- [ ] **Phase 1: Inference and Data**
  - [x] Processed training datasets and data synthesis pipeline.
  - [x] Core model architecture.
  - [x] Single-example inference entrypoint.
  - [x] Multi-context JSONL evaluation entrypoint.
- [ ] **Phase 2: Training Pipeline**
  - [x] Three-stage training CLI.
  - [x] Public shell templates for LFRP, QAFT-DC, and QAFT-QA.

If you find this work useful, please consider starring the repository to follow
future checkpoint and training updates.

## Installation

From this directory:

```bash
pip install -e .
```

For training utilities:

```bash
pip install -e ".[train]"
```

For evaluation metrics:

```bash
pip install -e ".[eval]"
```

For the QA-Evidence data generation pipeline:

```bash
pip install -e ".[data-generation]"
```

## Datasets

Processed training datasets are available on Hugging Face:

- [DRIFT_LFRP](https://huggingface.co/datasets/SII-LancelotXie/DRIFT_LFRP)
- [DRIFT_QAFT](https://huggingface.co/datasets/SII-LancelotXie/DRIFT_QAFT)

Training files can also be local JSON, JSONL, or Parquet files.

The preprocessing code is intentionally permissive about legacy field names, but
the expected semantic fields are:

- `Document`: long context or evidence text.
- `Question`: user question.
- `Answer`: target answer. Samples without an answer raise a clear
  preprocessing error with sample identifiers when available.
- `Instruction` or `UserInstruction`: optional task instruction.
- `answer_prefix`: optional dataset-level answer prefix. This has higher
  priority than a CLI/function-level prefix.

## Data Generation

The QA-Evidence synthesis pipeline is included in:

```text
data_generation/generate_qa.py
data_generation/construct_data.sh
```

`generate_qa.py` takes a Parquet file with a `context` column and produces JSONL
records containing generated `question`, `answer`, and `evidence` fields. It
expects an OpenAI-compatible chat completion server, such as a local vLLM server.

Example:

```bash
cd data_generation
python generate_qa.py /path/to/input.parquet /path/to/output.jsonl \
  --sample_size 1.0 \
  --num_workers 32
```

The current public script preserves the original data generation behavior. The
default endpoint/model settings are still the original local vLLM settings and
should be adjusted before running in a different environment.

## Training Pipeline

DRIFT uses three training stages:

```text
LFRP -> QAFT-DC -> QAFT-QA
```

The public stage names are:

- `lfrp`: latent fact reconstruction, called `pretrain_1` in the legacy code.
- `qaft_dc`: query-aware dynamic compression, called `pretrain_2` in the legacy
  code.
- `qaft_qa`: answer generation from implicit fact embeddings, called
  `simple_sft` in the legacy code.

Legacy stage names are still accepted as CLI aliases for migration convenience,
but new scripts and documentation use the paper names.

The main training CLI is:

```bash
drift-train --stage lfrp --help
```

The recommended sequence is to use the provided shell templates:

```bash
bash scripts/train_lfrp.sh
bash scripts/train_qaft_dc.sh
bash scripts/train_qaft_qa.sh
```

A minimal LFRP run:

```bash
export REASONING_MODEL=Qwen/Qwen2.5-3B-Instruct
export KNOWLEDGE_MODEL=Qwen/Qwen2.5-3B-Instruct
export TRAIN_FILE=/path/to/lfrp_train.jsonl
export CHECKPOINT_DIR=/path/to/lfrp_output
bash scripts/train_lfrp.sh
```

QAFT-DC and QAFT-QA usually start from the previous stage:

```bash
export REASONING_MODEL=/path/or/hf/id
export KNOWLEDGE_MODEL=/path/to/previous/final_model/knowledge_model/merged_model
export PROJECTOR_PATH=/path/to/previous/final_model/projector.pt
export TRAIN_FILE=/path/to/train.jsonl
export CHECKPOINT_DIR=/path/to/output
bash scripts/train_qaft_dc.sh
```

Then run QAFT-QA with the QAFT-DC knowledge model and projector:

```bash
export KNOWLEDGE_MODEL=/path/to/qaft_dc/final_model/knowledge_model/merged_model
export PROJECTOR_PATH=/path/to/qaft_dc/final_model/projector.pt
export TRAIN_FILE=/path/to/qaft_qa_train.jsonl
export CHECKPOINT_DIR=/path/to/qaft_qa_output
bash scripts/train_qaft_qa.sh
```

`REASONING_MODEL` and `KNOWLEDGE_MODEL` may come from different model families,
as long as the resulting hidden sizes and projector configuration are compatible
with the checkpoint being trained or loaded.

## Checkpoints

Evaluation is expected to use the QAFT-QA `final_model` directory. Intermediate
`checkpoint-*` directories are training-state checkpoints and are not the default
evaluation target.

`load_drift_model()` resolves model sources in this order:

1. User-provided `reasoning_model_name_or_path` and
   `knowledge_model_name_or_path`.
2. Merged model weights saved inside the checkpoint.
3. `drift_config.json` model paths as a fallback.

The checkpoint must contain `projector.pt`. If the projector was frozen during a
stage, training still saves a copy into the final checkpoint so the checkpoint is
self-contained.

## Inference

For one document/question pair:

```bash
drift-infer \
  --checkpoint /path/to/qaft_qa/final_model \
  --question "What is the answer?" \
  --document-file /path/to/document.txt
```

Use `--json` to print a JSON object instead of raw answer text.

For benchmark-style multi-context evaluation:

```bash
export CHECKPOINT=/path/to/qaft_qa/final_model
export INPUT_FILE=/path/to/eval.jsonl
export OUTPUT_FILE=/path/to/predictions.jsonl
export NUM_GPUS=1
bash scripts/eval_multi_context.sh
```

Single-GPU evaluation supports `--num-gpus 1` and optional `--device`. Multi-GPU
evaluation keeps one worker per GPU and restores output order after sharding.

## Public Entrypoints

- `drift-train`: three-stage training.
- `drift-infer`: single document/question inference.
- `drift-eval-multi`: JSONL multi-context evaluation.

## Development Checks

Lightweight prompt/data conversion regression:

```bash
PYTHONPATH=src python scripts/regression_prompt_conversion.py
```

Syntax checks used during the refactor:

```bash
PYTHONPATH=src python -m compileall -q src examples
bash -n scripts/train_lfrp.sh scripts/train_qaft_dc.sh scripts/train_qaft_qa.sh scripts/eval_multi_context.sh
```

## Project Layout

```text
DRIFT/
  src/drift/          # public package
  data_generation/    # QA-Evidence synthesis pipeline
  docs/               # architecture and development notes
  examples/           # minimal runnable examples
  scripts/            # shell templates for training and evaluation
  confused.md         # strange details found during refactor
  question.md         # unresolved design questions
```

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@misc{xie2026decoupledreasoningimplicitfact,
      title={Decoupled Reasoning with Implicit Fact Tokens (DRIFT): A Dual-Model Framework for Efficient Long-Context Inference},
      author={Wenxuan Xie and Yujia Wang and Xin Tan and Chaochao Lu and Xia Hu and Xuhong Wang},
      year={2026},
      eprint={2602.10021},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.10021},
}
```

## Contact

For questions or collaborations, please open an issue or contact:

- **Wenxuan Xie**: [wxxie25@m.fudan.edu.cn](mailto:wxxie25@m.fudan.edu.cn)

## Refactor Notes

The refactor aims to preserve original behavior by default. Surprising but
manageable implementation details are recorded in `confused.md`; ambiguous
design questions are recorded in `question.md`.
