"""Minimal single-example multi-context DRIFT inference.

Run from the repository root, for example:

PYTHONPATH=drift/src python drift/examples/inference_multi_context.py \
  --checkpoint /path/to/final_model \
  --document-file /path/to/document.txt \
  --question "What is the answer?"
"""

from __future__ import annotations

from drift.inference.infer import main


if __name__ == "__main__":
    main()
