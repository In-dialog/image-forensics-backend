"""Bounded, verified Ollama unloads for the local development pipeline."""
from __future__ import annotations

import time


def _client():
    try:
        import ollama
    except ImportError:
        return None
    # The SDK honors OLLAMA_HOST, including a custom local installation.
    factory = getattr(ollama, "Client", None)
    return factory(timeout=10.0) if factory else ollama


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _loaded(client):
    if not hasattr(client, "ps"):
        return None
    return [name for item in _field(client.ps(), "models", [])
            if (name := _field(item, "model") or _field(item, "name"))]


def _same_model(left, right):
    def normalized(name):
        return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"
    return normalized(left) == normalized(right)


def _unload(client, model):
    client.generate(model=model, prompt="", keep_alive=0)
    deadline = time.monotonic() + 10
    while True:
        models = _loaded(client)
        if models is None or not any(_same_model(name, model) for name in models):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Ollama did not release {model}; the next model was not started.")
        time.sleep(0.1)


def offload_model(model: str, *, strict: bool = True) -> None:
    try:
        client = _client()
        if client is None:
            return
        models = _loaded(client)
        if models is None or any(_same_model(name, model) for name in models):
            _unload(client, model)
    except ConnectionError:
        return  # No reachable Ollama server/model to unload.
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Could not unload Ollama model {model}: {exc}") from exc
        print(f"Ollama cleanup warning: {exc}")


def offload_all_models(*, except_model: str | None = None, strict: bool = True) -> None:
    """Release every resident tag, including models selected outside defaults."""
    try:
        client = _client()
        if client is None:
            return
        for model in _loaded(client) or []:
            if except_model is None or not _same_model(model, except_model):
                _unload(client, model)
    except ConnectionError:
        return
    except Exception as exc:
        if strict:
            raise RuntimeError(f"Could not release Ollama VRAM: {exc}") from exc
        print(f"Ollama cleanup warning: {exc}")


def prepare_torch_model() -> None:
    """Hand the GPU from Ollama to a local inference model before loading it."""
    from .gpu import release_cuda_memory
    offload_all_models()
    release_cuda_memory()
