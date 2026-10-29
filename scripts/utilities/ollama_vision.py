"""
Ollama vision: send a prompt and an image to an Ollama vision model, get text response.
Uses fixed options to reduce hallucination (internal; no other component needs to know).

Options are passed at the request level (ollama.chat(..., options=options)), not inside
messages — this is required by the Ollama API (POST /api/chat).

Format: a flat dict sent as the request body's top-level "options" field. Keys are
snake_case; values are scalars (int, float, bool, or str). Example JSON sent to API:
  {"model": "...", "messages": [...], "options": {"num_ctx": 16384, "temperature": 0.15, ...}}
Official option names: num_ctx, num_predict, temperature, top_k, top_p, repeat_penalty,
repeat_last_n, presence_penalty, seed (see ollama-python _types.Options). Thinking uses the top-level think field.
"""
import logging
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

from scripts.runtime.gpu import release_cuda_memory
from scripts.runtime.ollama import offload_model, offload_all_models

_SESSION_MODEL: ContextVar[str | None] = ContextVar("vision_session_model", default=None)


def _prepare_vision_model(model: str) -> None:
    release_cuda_memory()
    offload_all_models(except_model=model)

# PhotoHolmes initializes root logging at INFO, which makes httpx print every Ollama POST.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Fixed generation options (flat dict, snake_case keys, scalar values per API).
_OPTIONS = {
    "num_ctx": 8192,
    "temperature": 0,
    "top_p": 0.75,
    "top_k": 30,
    "repeat_penalty": 1.25,
    "presence_penalty": 0,
    "repeat_last_n": 256,
    "seed": 42,
}


def _response_field(response: Any, key: str, default: Any = None) -> Any:
    """Read a field from dict-like or object-like Ollama responses."""
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def _response_message_content(response: Any) -> str:
    """Extract assistant content from dict-like or object-like Ollama responses."""
    message = _response_field(response, "message", None)
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if content is None:
        return ""
    return str(content).strip()


def _response_message_thinking(response: Any) -> str:
    """Extract assistant thinking text when available (debug use only)."""
    message = _response_field(response, "message", None)
    if isinstance(message, dict):
        thinking = message.get("thinking", "")
    else:
        thinking = getattr(message, "thinking", "")
    if thinking is None:
        return ""
    return str(thinking).strip()


def _image_for_ollama(path: Path) -> str | bytes:
    """Keep tiny crops compatible with Qwen's resize grid without changing coordinates."""
    from io import BytesIO
    from PIL import Image

    with Image.open(path) as source:
        width, height = source.size
        smallest_side = min(width, height)
        if smallest_side > 32:
            return str(path.resolve())
        # An integer scale preserves the exact aspect ratio. No padding or crop
        # offsets are introduced, so normalized detection boxes retain their frame.
        scale = (64 + smallest_side - 1) // smallest_side
        size = (width * scale, height * scale)
        mode = "RGBA" if "A" in source.getbands() else "RGB"
        compatible = source.convert(mode).resize(size, Image.Resampling.LANCZOS)
        buffer = BytesIO()
        compatible.save(buffer, format="PNG")
    print(f"  Resized model input for small-image compatibility: {width}x{height} -> {size[0]}x{size[1]}")
    return buffer.getvalue()


def vision_chat(
    prompt: str,
    image: Path | str,
    *,
    model: str = "qwen3-vl:8b-instruct",
    vision_device: str | None = None,
    keep_alive: int | str = 0,
    num_ctx: int | None = None,
    num_predict: int | None = 2048,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repeat_penalty: float | None = None,
    presence_penalty: float | None = None,
    repeat_last_n: int | None = None,
    seed: int | None = None,
    think: bool | str | None = False,
    debug_raw_response: bool = False,
    user_content: str = "This is the image.",
    system_prompt: str | None = None,
    response_schema: dict[str, Any] | None = None,
    detail_images: list[Path | str] | None = None,
) -> str:
    """
    Call Ollama vision with a prompt and image path. Returns the assistant text content.

    Args:
        prompt: User text prompt.
        image: Path to the image file (Path or str).
        model: Ollama model name (must support vision).
        vision_device: CPU disables GPU offload; CUDA requests Ollama's dynamic GPU placement.
        keep_alive: How long to keep the model loaded after the request.
            0 = unload immediately (default). Use a duration string (e.g. "5m")
            or -1 to keep loaded indefinitely.
        num_ctx: Context window size override.
        num_predict: Max tokens to generate.
        temperature: Override temperature (higher = more varied outputs; use for ranking tasks).
            When set, also disables fixed seed for better score variation across regions.
        top_k: Override top-k sampling.
        top_p: Override top-p nucleus sampling.
        repeat_penalty: Override repeat penalty.
        presence_penalty: Override the penalty for tokens already present in generated text.
        repeat_last_n: Override repeat window.
        seed: Override seed for reproducible outputs.
        think: Enable/disable thinking mode for models that support it.
            Default False to avoid token budget being consumed by hidden reasoning.
        debug_raw_response: If True, print the full raw Ollama response payload.
        user_content: Generic user message content paired with the image.
        system_prompt: Application-owned instructions, separate from editable analysis goals.
        response_schema: Protected JSON schema used for constrained generation.
        detail_images: Optional additional evidence crops, after the main image.

    Returns:
        Assistant reply text.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ImportError: If the ollama package is not installed.
    """
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image_paths = [path, *(Path(detail) for detail in (detail_images or []))]
    for detail in image_paths[1:]:
        if not detail.is_file():
            raise FileNotFoundError(f"Detail image not found: {detail}")

    try:
        import ollama
    except ImportError as err:
        raise ImportError("Install: pip install ollama") from err

    options = dict(_OPTIONS)
    if vision_device not in (None, "auto", "cpu", "cuda"):
        raise ValueError("Vision device must be auto, cpu or cuda")
    if vision_device == "cpu":
        options["num_gpu"] = 0
    elif vision_device == "cuda":
        # Ollama selects the GPU offload count to fit available device memory.
        options["num_gpu"] = -1
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if num_predict is not None:
        options["num_predict"] = num_predict
    if temperature is not None:
        options["temperature"] = temperature
        options.pop("seed", None)
    if top_k is not None:
        options["top_k"] = top_k
    if top_p is not None:
        options["top_p"] = top_p
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    if presence_penalty is not None:
        options["presence_penalty"] = presence_penalty
    if repeat_last_n is not None:
        options["repeat_last_n"] = repeat_last_n
    if seed is not None:
        options["seed"] = seed
    think_value: bool | str | None = think
    if isinstance(think, str):
        low = think.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            think_value = True
        elif low in {"0", "false", "no", "off"}:
            think_value = False
        elif low in {"low", "medium", "high"}:
            think_value = low
        elif not low:
            think_value = None
    # Structural instructions belong to the application, separate from editable goals.
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
        {
            "role": "user",
            "content": prompt.strip() or user_content.strip() or "This is the image.",
            "images": [_image_for_ollama(image_path) for image_path in image_paths],
        }
    ]
    chat_kwargs = {
        "model": model,
        "messages": messages,
        "keep_alive": keep_alive,
        "options": options,
    }
    if response_schema is not None:
        chat_kwargs["format"] = response_schema
    if think_value is not None:
        chat_kwargs["think"] = think_value

    # Options must be top-level (Ollama API); do not put options inside message dicts.
    if _SESSION_MODEL.get() != model:
        _prepare_vision_model(model)
    try:
        response = ollama.chat(**chat_kwargs)
    except BaseException:
        if _SESSION_MODEL.get() != model:
            offload_model(model, strict=False)
        raise
    if debug_raw_response:
        print(f"Vision raw response payload: {response}")

    content = _response_message_content(response)
    if content:
        return content

    # Debug signal for the common failure mode: reasoning consumed budget and content is empty.
    done_reason = _response_field(response, "done_reason", "")
    thinking = _response_message_thinking(response)
    if done_reason == "length" and thinking:
        print(
            "Vision warning: empty content with done_reason=length; "
            f"thinking consumed tokens (thinking_len={len(thinking)})."
        )
    elif done_reason:
        print(f"Vision warning: empty content (done_reason={done_reason}).")
    else:
        print("Vision warning: empty content from model.")
    return ""


class StructuredResponseError(ValueError):
    """The model failed to produce a response matching the protected contract."""


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def structured_vision_chat(
    prompt: str,
    image: Path | str,
    *,
    system_prompt: str,
    response_schema: dict[str, Any],
    validate: Callable[[Any], Any],
    **model_options: Any,
) -> dict[str, Any]:
    """Request JSON and validate it, allowing one bounded correction attempt.

    Parsing is deliberately strict: no markdown stripping, extracted fragments,
    duplicate fields, non-finite numbers, or default values for invalid results.
    Transport/model failures propagate immediately; only invalid content is retried.
    """
    error: Exception | None = None
    for attempt in range(2):
        correction = ""
        if attempt:
            correction = (
                "\n\nYour previous response did not satisfy the required JSON contract: "
                f"{str(error)[:600]}. Return a corrected complete JSON object. "
                "Do not add fields or invent observations to satisfy the contract."
            )
        content = vision_chat(
            prompt + correction,
            image,
            system_prompt=system_prompt,
            response_schema=response_schema,
            **model_options,
        )
        try:
            data = json.loads(
                content, parse_constant=_reject_nonfinite, object_pairs_hook=_unique_object
            )
            validate(data)
            if not isinstance(data, dict):
                raise ValueError("The response must be a JSON object")
            return data
        except (ValueError, TypeError, KeyError) as exc:
            error = exc
            if not attempt:
                print(f"  Model output failed validation; retrying once: {str(exc)[:200]}")
    raise StructuredResponseError(f"Model output failed validation after 2 attempts: {error}") from error


def offload_vision_model(model: str = "qwen3-vl:8b-instruct") -> None:
    """
    Ask Ollama to unload the vision model from memory (e.g. after finishing a group of segments).
    Uses generate with keep_alive=0 so the model is freed.
    """
    offload_model(model)


@contextmanager
def vision_model_session(model: str, *, keep_alive: str = "5m"):
    """Reuse a model for one stage, then request unload on success or failure."""
    _prepare_vision_model(model)
    token = _SESSION_MODEL.set(model)
    try:
        yield keep_alive
    except BaseException:
        offload_model(model, strict=False)
        raise
    else:
        offload_vision_model(model)
    finally:
        _SESSION_MODEL.reset(token)
