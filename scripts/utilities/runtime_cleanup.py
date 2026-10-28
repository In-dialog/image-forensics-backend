"""
Runtime cleanup helpers to aggressively release model/process GPU memory.
Used by UI and orchestration entrypoints.
"""
from __future__ import annotations

import gc


def cleanup_runtime(
    *,
    offload_vision: bool = True,
    unload_sam_from_gc: bool = True,
    cuda_reset: bool = False,
) -> None:
    """
    Best-effort cleanup of runtime resources.

    Args:
        offload_vision: Ask Ollama to unload the vision model.
        unload_sam_from_gc: Find leaked Sam3Loaded objects in gc and unload them.
        cuda_reset: Hard-reset CUDA context for this process.
    """
    gc.collect()

    if unload_sam_from_gc:
        try:
            from scripts.utilities.sam3_utils import unload_all_sam3
            unload_all_sam3()
        except Exception:
            pass

    if offload_vision:
        try:
            from scripts.utilities.ollama_vision import offload_vision_model
            offload_vision_model()
        except Exception:
            pass

    gc.collect()

    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

    if cuda_reset:
        try:
            cudart = torch.cuda.cudart()
            if cudart is not None:
                cudart.cudaDeviceReset()
        except Exception:
            pass
