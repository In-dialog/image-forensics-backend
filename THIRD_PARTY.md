# External components

The MIT license covers this project's own source only. Dependencies are obtained from their upstream distributors during installation; model weights are not redistributed in this repository.

| Component | Upstream / applicable model terms |
| --- | --- |
| PyTorch | https://pytorch.org/ |
| Hugging Face Transformers / Diffusers | https://github.com/huggingface/transformers · https://github.com/huggingface/diffusers |
| SAM 3 | https://huggingface.co/facebook/sam3 — gated access and the model's own license. |
| Qwen3-VL through Ollama | https://ollama.com/library/qwen3-vl:8b-instruct · https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct |
| Marigold depth | https://huggingface.co/prs-eth/marigold-depth-v1-1 |
| OpenPose adapter | https://github.com/huggingface/controlnet_aux — check upstream detector/model terms. |
| DeepGaze / CLIP | https://github.com/matthias-k/DeepGaze · https://github.com/openai/CLIP — source and pretrained models may have separate terms. |
| PhotoHolmes (optional) | https://github.com/photoholmes/photoholmes — individual forensic weights have separate upstream sources/terms. |
| Nonescape (optional) | https://github.com/e3ntity/nonescape |
| uv installer/runtime bootstrap | https://github.com/astral-sh/uv |

The installation profiles pin application dependencies where appropriate. Some upstream libraries fetch their pretrained weights on first use; these downloads remain in runtime caches, never in Git.

For the optional PhotoHolmes add-on, the installer prepares its pinned upstream source and changes the declared `opencv-python` dependency to `opencv-python-headless` before building it. This avoids installing two competing `cv2` implementations in the same environment. The upstream implementation and license files are preserved; the dependency change and source provenance are recorded in the local prepared source. Native dependencies such as `jpegio` can still require a C/C++ toolchain.
