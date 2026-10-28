# Image Forensics Analysis — backend

The local analysis engine for Image Forensics Analysis. It detects and segments objects, examines each object in its scene, describes evidence, and produces configurable rankings and image-analysis outputs.

This repository is independent of the web interface. It contains source code and tests, **not model weights, previous analyses, accounts, or a Python environment**.

## First installation

1. Download this repository using **Code → Download ZIP** on GitHub, then extract it into a normal folder. Do not run it from inside the ZIP.
2. **Windows:** double-click **Install.cmd**. **Linux:** open a terminal in the extracted folder and run `bash install.sh`.
3. Follow the installer. It checks the machine, prepares an isolated Python environment, installs dependencies, and guides model setup. Keep the window open until it reports completion.
4. **Windows:** double-click **Start.cmd**. **Linux:** run `bash start.sh`. Keep that terminal open while analyzing images.
5. Install/start the separate **image-forensics-frontend** repository and open `http://127.0.0.1:3100`.

The install step does not launch the analysis API. Downloads happen on the machine running the installer; cached Hugging Face/Ollama models are reused. Running the installer again preserves analysis results.

### What you need

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

## Safe installation preview

Inspect the plan without installing Python packages, fetching models, or starting services:

```powershell
Install.cmd --dry-run
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

To install software now and fetch models later, use `Install.cmd --skip-models` or `bash install.sh --skip-models`. Then run `.venv\Scripts\python.exe -B install\models.py` on Windows or `.venv/bin/python -B install/models.py` on Linux. `--group sam3` (or `depth`, `pose`, `saliency`, `ollama`, `forgery`, `synthetic`) prepares just that model group. Re-running model preparation reuses successful cached downloads.

## Development

The API listens on `127.0.0.1:8766`. It is a local development service, without a public authentication or multi-user deployment layer. Keep it bound to loopback.

```sh
python -m scripts.api.server
python -m unittest discover -s tests -p 'test_*.py'
```

Use the installed project Python environment for development. Tests inject fake model calls; the HTTP contract tests use temporary local sockets. No inference/download is needed for unit tests.

See [architecture](docs/ARCHITECTURE.md) and [third-party components](THIRD_PARTY.md).

## License

The project's own code is [MIT licensed](LICENSE). Model weights, external libraries, and analyzed media retain their respective terms. Model terms are not replaced by this license.
