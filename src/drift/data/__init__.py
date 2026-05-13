"""Data utilities for DRIFT.

Heavy dependencies such as torch and transformers are imported by submodules on
demand. This keeps lightweight imports like `drift.data.templates` usable in
minimal environments.
"""
