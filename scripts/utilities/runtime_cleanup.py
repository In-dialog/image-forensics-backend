"""
Release stage-owned models and CUDA allocations for API and CLI entrypoints.
"""
from __future__ import annotations

import sys

from scripts.runtime.gpu import release_cuda_memory
from scripts.runtime.ollama import offload_all_models


def cleanup_runtime(*, strict: bool = False, report: bool = False) -> None:
    """Release stage-owned weights and allocations without resetting CUDA."""
    sam = sys.modules.get("scripts.utilities.sam3_utils")
    if sam is not None:
        sam.unload_all_sam3()
    try:
        offload_all_models(strict=strict)
    finally:
        release_cuda_memory(report=report, strict=strict)
