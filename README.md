# Image Forensics Analysis — backend

**[Demo](https://forensic-image-analysis-demo.pages.dev/results/)**

**Frontend:** [image-forensics-frontend](https://github.com/In-dialog/image-forensics-frontend)

The local analysis engine for Image Forensics Analysis. It detects and segments objects, examines each object in its scene, describes evidence, and produces configurable rankings and image-analysis outputs.

This repository is independent of the web interface. It contains source code and tests, **not model weights, previous analyses, accounts, or a Python environment**.

**Only want to view saved results? Install the separate image-forensics-frontend repository. You do not need this backend.**

[Install](#install) · [Run](#run-an-analysis) · [Stop and restart](#stop-and-restart) · [Troubleshooting](#troubleshooting)

## Before you install

| Component | Requirement |
| --- | --- |
| Operating system | 64-bit Windows 10 22H2 or newer, or x64 Linux. Linux is the preferred environment for advanced forensic dependencies. |
| GPU | An NVIDIA CUDA GPU is recommended. The installer chooses the CUDA wheel profile from the detected GPU. A deliberately selected CPU profile is much slower. |
| Memory | Plan around 32 GB system RAM and 12 GB or more GPU memory for the default models. These are planning targets, not a verified minimum or a guarantee for every image. |
| Storage | Plan for 50 GiB free for a fresh software/runtime/model setup, more for optional models and generated results. Existing caches reduce new downloads. Use a large disk for shared model caches. |
| Ollama | The installer checks/helps install Ollama and prepares `qwen3-vl:8b-instruct` plus `qwen3-vl:8b-thinking` for whole-image rankings. |
| Hugging Face | A free account, a personal read token, and approved access to **facebook/sam3**. Approval and acceptance of model terms must be completed by the account owner. |

macOS, AMD/Intel GPU acceleration, and ARM backend machines are not supported by this installer. The **frontend and saved database viewer work independently on supported Windows/macOS machines**, without a GPU or Hugging Face account.

See [hardware and model setup](docs/HARDWARE.md) for the official links and cache options.

You do **not** need to install Python or Git yourself. You need an internet connection for the first installation. The installer prepares private Python environments and selects the CPU/CUDA package profile for your machine. NVIDIA drivers must already work; it does not replace them.

### Prepare your Hugging Face account

Complete these steps before the large model downloads:

1. [Create or sign into your Hugging Face account](https://huggingface.co/join).
2. Open [facebook/sam3](https://huggingface.co/facebook/sam3), read its terms, and request/accept access. Wait until that account has access.
3. Open [Access Tokens](https://huggingface.co/settings/tokens) and create a **read** token. A fine-grained token must be allowed to read the gated SAM3 repository.
4. Keep it ready for the installer's hidden-input prompt. Enter the token there, not in the website or a GitHub issue.

A token alone does not grant access to a model whose access request is still pending.

## Install

### 1. Download from GitHub

On this repository's GitHub page, select **Code → Download ZIP**. Extract the ZIP into a normal folder on a drive with enough space. Open the extracted folder containing `Install.cmd` and `install.sh`.

Do not run the installer from inside the ZIP. Keep the installed folder in place; its private Python environment should not be moved between computers.

### 2. Run the installer

**Windows:** double-click **Install.cmd**. Alternatively, open PowerShell in the extracted backend folder and run:

```powershell
.\Install.cmd
```

**Linux:** open Terminal in the extracted backend folder and run:

```bash
bash install.sh
```

Follow the prompts and keep the window open until it says **Setup complete**. The installer checks your hardware and storage, installs dependencies, requests model authentication when needed, and downloads the required models. The first installation can take a while; watch the progress messages.

If setup fails, read the final error, fix that issue, and rerun the installer. Successfully cached models are reused. Running Install again preserves analysis results. Installation does not start the analysis API.

## Run an analysis

### 1. Start the backend

**Windows:** double-click **Start.cmd**, or run this from PowerShell in the backend folder:

```powershell
.\Start.cmd
```

**Linux:** run:

```bash
bash start.sh
```

Wait for **Analysis API: http://127.0.0.1:8766**. Leave that terminal open. The backend has no graphical window of its own.

### 2. Open the web interface

Install the separate **image-forensics-frontend** repository using its README, then start it. Open **http://127.0.0.1:3100** in your browser.

For the normal local setup, both applications run on the same computer. Windows has launchers for both. Linux users can run the frontend with Node.js 24 using its developer commands. A frontend on another computer requires an explicitly configured local tunnel; there is no automatic network discovery.

### 3. Select the work to run

1. Open **New analysis** in the local website.
2. Choose an image and select the stages you want.
3. Adjust the analysis questions/rankings if needed.
4. Select **Start analysis** and watch the progress and previews.
5. Open the completed result in the database viewer.

## Stop and restart

Press **Ctrl+C in the backend terminal** to stop it. Stop the frontend in its own terminal with Ctrl+C. Next time, use **Start** again; you do not need to reinstall.

If a run was interrupted or failed, open **Unfinished** in the website. You can resume a run, resume unfinished runs in order, or delete one. Nothing resumes automatically after startup.

## Safe installation preview

Inspect the plan without installing Python packages, fetching models, or starting services:

```powershell
.\Install.cmd --dry-run
```

```sh
bash install.sh --dry-run
```

The installer also exposes command-line options for CPU operation, deferred model downloads, and optional stages. Run it with `--help` for the current options.

## Analysis stages and recovery

The core profile includes object extraction, bounding-box composition, depth, negative space, composition lines, saliency, original-image export, detailed segmentation, pose, descriptions, and whole-image/per-item rankings. **PhotoHolmes forgery detection and Nonescape synthetic-image detection are separate optional installs**; they are not selected by the default profile.

Per-stage progress includes item counts and substeps. Failed/interrupted runs remain in the web interface's **Unfinished** list. Resume uses saved settings and completed-stage checkpoints; Delete removes the selected unfinished run and its saved input. Nothing resumes automatically after restarting the backend.

Results live in `data/` by default. The web app downloads successful results into its own data directory through the API; the two applications do not share a folder. Stopping the backend does not remove the web app's saved database.

Start reuses an existing Ollama service or starts one for this backend session. Closing the API stops only services started by this application, including the optional synthetic service; it leaves independently running services alone.

GPU models and temporary tensors are released between stages, including failed stages. Ollama models are unloaded before another model takes the GPU. The synthetic service loads its classifier only while handling a request and releases it afterward. Cleanup failures stop the next stage from starting.

Synthetic detection prefers the `.venv-synthetic` environment installed by `--with-synthetic`. It can reuse `.venv` if Nonescape is already installed there, or an interpreter selected through `SYNTHETIC_PYTHON`. Startup never installs packages or downloads models. Missing environments report the installation command; other startup failures include the service's error and the path to `.runtime/synthetic-api.log`. New synthetic installations still use a separate environment to protect the core dependencies.

To install software now and fetch models later, use `.\Install.cmd --skip-models` in Windows PowerShell or `bash install.sh --skip-models` on Linux.

**Complete or retry model preparation:**

Windows PowerShell, opened in the backend folder:

```powershell
.\.venv\Scripts\python.exe -B install\models.py
```

Linux:

```bash
.venv/bin/python -B install/models.py
```

Add `--group sam3` to retry just SAM3, or use `depth`, `pose`, `saliency`, `ollama`, `forgery` or `synthetic` for the corresponding installed feature. Successful cached downloads are reused.

## Troubleshooting

| What you see | What to do |
| --- | --- |
| SAM3 access denied / 401 / 403 | Check access on the model page and make sure your token belongs to the approved account. Retry model preparation. |
| Not enough disk space | Choose a larger drive before installing, or configure a model-cache location using the hardware guide. Do not bypass the check unless you know which existing caches are reused. |
| NVIDIA/CUDA is unavailable | Update the driver using NVIDIA's official tools and retry, or explicitly use `--profile cpu`. CPU mode is slower. |
| Missing Ollama model | Finish model preparation using the command above, then start the backend again. |
| Port 8766 is already in use | A backend may already be running. Keep one instance; close the older terminal before starting another. |
| The website cannot find the backend | Keep the backend terminal open, check for the API-ready message, then select **Try again** in the local website. |
| The Start window closes immediately | Open PowerShell in the backend folder and run `.\Start.cmd` there to keep the error visible. |
| An optional forensic package fails to build | Use the core installation first. Native add-ons can require a C/C++ compiler and JPEG development libraries. |

## Optional installation choices

Run these commands in PowerShell opened in the backend folder. On Linux, use the same options after `bash install.sh`.

| Purpose | Windows PowerShell command |
| --- | --- |
| Use the CPU deliberately | `.\Install.cmd --profile cpu` |
| Defer model downloads | `.\Install.cmd --skip-models` |
| Put new HF/Torch/forgery caches on another drive | `.\Install.cmd --cache-dir "D:\AI-models"` |
| Add advanced forensic tools | `.\Install.cmd --with-forgery` |
| Add isolated synthetic-image detection | `.\Install.cmd --with-synthetic` |
| Show all options | `.\Install.cmd --help` |

Ollama and CLIP have separate cache settings. Read [hardware, models and storage](docs/HARDWARE.md) before choosing a custom location.

## Updating and keeping your results

Stop both applications first. Back up `data/` (including its hidden `.jobs` and `.inputs` directories), `scripts/configuration/settings.json`, and `.install-state.json` if present. These are private files; keep them out of GitHub.

Update the source files in the existing project location, keeping your results and private configuration files. Run Install again using the same optional-feature/cache options you chose originally, then review the selected stages. The installer reuses environments and caches and preserves analysis data. Keeping the project location avoids breaking installed-environment paths.

## Development

The API listens on `127.0.0.1:8766`. It is a local development service, without a public authentication or multi-user deployment layer. Keep it bound to loopback.

```sh
python -m scripts.api.server
python -m unittest discover -s tests -p 'test_*.py'
```

Use the installed project Python environment for development. Tests inject fake model calls; the HTTP contract tests use temporary local sockets. No inference/download is needed for unit tests.

See [architecture](docs/ARCHITECTURE.md) and [third-party components](THIRD_PARTY.md).

See [validation performed and remaining machine tests](docs/VALIDATION.md) for the current release checks.

## License

The project's own code is [MIT licensed](LICENSE). Model weights, external libraries, and analyzed media retain their respective terms. Model terms are not replaced by this license.
