# Hardware and model setup

## Before pressing Install

Use Windows x64 or Linux x64. Keep enough free space for both Python packages and model caches: the core installer uses a conservative 50 GiB fresh-install allowance, with additional space for optional models and results. This is not a measured download total. Reusing existing caches can reduce new downloads; it does not make the package sizes disappear.

For the default vision/segmentation workflow, plan for an NVIDIA GPU with 12 GiB or more VRAM and about 32 GiB system RAM. This is a recommendation, not a measured minimum. Image resolution, number of segments, vision context length and simultaneously loaded models all affect memory use. CPU operation is available for experimentation and can be very slow.

The installer detects GPU capabilities and chooses a compatible PyTorch wheel profile. RTX 50-series GPUs require the newer CUDA 12.8 profile. NVIDIA display drivers are machine-level software: install/update them using [NVIDIA's official driver page](https://www.nvidia.com/Download/index.aspx), then restart the installer. The installer does not silently replace system drivers.

See the [official PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/) and [Ollama GPU support](https://docs.ollama.com/gpu). The main backend installation is isolated from system Python.

## Hugging Face: complete these steps once

1. Create/sign into an account at [Hugging Face](https://huggingface.co/join).
2. Open [facebook/sam3](https://huggingface.co/facebook/sam3), read its terms, and request/accept access. Wait for access to be granted to that same account.
3. Create a personal read token at [Access Tokens](https://huggingface.co/settings/tokens). A fine-grained token must be allowed to read this gated model.
4. Supply the token when the installer asks. Do not put it in this repository, a screenshot, or a GitHub issue.

Authentication is stored by Hugging Face outside the repository. A token alone does not grant access to a model whose access request is still pending. The installer cannot accept model terms or approve access on your behalf.

## Ollama

The default vision model is [Qwen3-VL 8B Instruct](https://ollama.com/library/qwen3-vl:8b-instruct); whole-image rankings use [8B Thinking](https://ollama.com/library/qwen3-vl:8b-thinking). Each registry entry lists roughly 6.2 GB and requires Ollama 0.12.7 or newer. Installation prepares both model tags. RAM/VRAM use during inference is greater than the download size.

The installer guides Ollama setup. If a manual install is needed, use [Ollama's official download](https://ollama.com/download), start Ollama, then rerun the project installer. Windows compatibility is documented [here](https://docs.ollama.com/windows).

On Linux, a missing Ollama runtime is installed privately under `.tools/ollama` from a pinned, checksum-verified official archive. This does not install a system service or modify drivers, package repositories or users. Extraction requires 14 GiB of free space on the project drive. An existing Ollama installation is reused. Windows uses the official user installer or the registered Windows package.

## Choosing a large disk

Set cache locations **before** the first installation if the default drive is small:

| Variable | Stored content |
| --- | --- |
| `HF_HOME` | Shared Hugging Face downloads and authentication. |
| `TORCH_HOME` | PyTorch/DeepGaze pretrained checkpoints. |
| `OLLAMA_MODELS` | Ollama model storage; set for the Ollama process and restart Ollama yourself. |
| `ANALYSIS_OUTPUT_DIR` | Backend analysis results. |
| `IMAGE_FORENSICS_WEIGHTS` | Optional PhotoHolmes checkpoints. |

Existing shared caches are reused, not copied into the project. The installer's `--cache-dir PATH` option chooses a new Hugging Face/Torch/forgery cache location while respecting environment overrides. Ollama uses its own model path, and the upstream CLIP adapter uses `~/.cache/clip`. Installation includes large downloads on the end user's computer; `--dry-run` does not download anything.

## Optional stages

PhotoHolmes adds native dependencies; Windows may need a C/C++ build toolchain, and Linux is the preferred platform. Nonescape is installed into a separate environment to preserve its dependency versions. These profiles are opt-in. If an optional install fails, the installer must report it; do not enable the corresponding analysis stage until its prerequisites work.
