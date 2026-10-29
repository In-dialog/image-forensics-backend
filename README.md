# Image Forensics Analysis — Backend

**[Demo](https://forensic-image-analysis-demo.pages.dev/results/)** · **[Frontend repository](https://github.com/In-dialog/image-forensics-frontend)**

This repository contains the **analysis engine and local API** for Image Forensics Analysis. It runs image-processing and AI models, saves their outputs, and makes those results available to the separate web interface.

Start this backend to run analyses. To browse the included example database, use the frontend. Instructions for downloading and running both projects together are [at the bottom](#install-and-build-both-repositories).

## Features

- **Object and part extraction:** detect and segment items, then analyze them in the context of the full image.
- **Image measurements:** produce depth, composition, negative-space, attention, and pose outputs.
- **Descriptions and rankings:** answer configurable questions and evaluate image-level and per-item criteria using structured results.
- **Progress and recovery:** report stages and item progress, save checkpoints, and resume failed or interrupted jobs.
- **Optional forensic stages:** add PhotoHolmes forgery analysis or Nonescape synthetic-image detection during installation.
- **GPU resource management:** release finished models between stages. Each forensic check runs in its own process, which exits before the next check to free native CUDA allocations as well as model weights.

The source download does not contain model weights or a preinstalled Python environment. Installation prepares the software and models on your computer.

## Backend requirements

| Component | Requirement for this backend |
| --- | --- |
| Operating system | Windows 10 22H2 or newer on x64, or Linux x64. |
| GPU | NVIDIA CUDA recommended, with a working NVIDIA driver. A CPU installation is available but much slower. |
| Memory | Planning target: about 32 GB system RAM and 12 GB or more GPU memory for the default models. Actual needs vary by image and settings. |
| Storage | Plan for 50 GiB free for a fresh installation, plus space for results and optional models. Existing caches can reduce downloads. |
| Model service | Ollama; the installer guides its setup and prepares the selected vision models. |
| Model access | A Hugging Face account, a read token, and approved access to `facebook/sam3`. |

The backend installer does not support macOS or ARM computers. AMD/Intel GPU acceleration is not provided by this installer. See [hardware and model setup](docs/HARDWARE.md) for details and cache locations.

You need an internet connection for the initial installation. The installer prepares private Python environments; you do not need to install Python or Git yourself.

## 1. Download this repository

1. [Download the backend ZIP](https://github.com/In-dialog/image-forensics-backend/archive/refs/heads/main.zip), or select **Code → Download ZIP** on this GitHub page.
2. Extract the ZIP into a folder on a drive with enough space.
3. Open the extracted folder containing **Install.cmd** and **install.sh**. Keep this folder in place after installing; Python environments should not be moved between locations.

Run the following steps from the extracted folder, not from inside the ZIP.

## 2. Prepare model access

1. [Sign in to Hugging Face](https://huggingface.co/join).
2. Open [facebook/sam3](https://huggingface.co/facebook/sam3), accept/request access, and wait for approval on that account.
3. Create a **read token** under [Access Tokens](https://huggingface.co/settings/tokens). Enter it when the installer asks.

A token does not replace model-access approval. Keep it out of GitHub and the frontend's settings. If you plan to use an NVIDIA GPU, make sure its driver is installed before running setup.

## 3. Install the backend

### Windows

Double-click **Install.cmd** and follow its prompts. To see errors in an existing PowerShell window, open PowerShell in the backend folder and run:

```powershell
.\Install.cmd
```

### Linux

Open a terminal in the backend folder and run:

```bash
bash install.sh
```

Wait for **Setup complete**. The installer checks the machine, installs Python dependencies, guides Ollama setup, and downloads the required models. If it stops, resolve the reported issue and run Install again; cached downloads are reused.

Installation does not start the API. The core installation enables 12 stages; the two optional forensic stages are installed separately through the options below.

## 4. Start the backend

| Your system | Run from the backend folder |
| --- | --- |
| Windows | Double-click `Start.cmd`, or run `.\Start.cmd` in PowerShell. |
| Linux | Run `bash start.sh` in a terminal. |

Wait for **Analysis API: http://127.0.0.1:8766**. Keep this terminal open. This starts the analysis service; connect the frontend using the combined setup below.

Press **Ctrl+C** in this terminal to stop. Next time, use Start again. The backend stops auxiliary services it started and leaves independently running services alone.

## Results and recovery

Results are saved under `data/` by default. The API reports progress and previews and supplies completed result bundles to clients.

Completed stages and validated model responses are checkpointed. Resume checks the saved input, settings, prompts, images, per-item results, rankings, and forensic outputs. Missing or corrupt results restart that stage and the stages after it; valid earlier work is preserved. A failed or interrupted run can be resumed through the connected frontend; nothing resumes automatically after restarting.

Synthetic model preparation and inference run as a reconnectable task. The frontend receives progress while the model loads, and Resume reuses the active task or completed prediction while the synthetic service stays running. This avoids the previous 30-second inference request timeout. The classifier is released after each task. After updating, restart the backend and any separately started synthetic service before resuming.

Before updating, stop the backend and back up `data/` including its hidden job/input folders, `scripts/configuration/settings.json`, and `.install-state.json`. Update the source in the existing project location, preserve those private files, and rerun Install with the same optional-feature choices.

## Optional installation choices

Run these commands from the backend folder. Linux uses the same options after `bash install.sh`.

| Purpose | Windows PowerShell command |
| --- | --- |
| Print the install plan without downloads or changes | `.\Install.cmd --dry-run` |
| Choose CPU processing | `.\Install.cmd --profile cpu` |
| Install software and download models later | `.\Install.cmd --skip-models` |
| Add PhotoHolmes forgery analysis | `.\Install.cmd --with-forgery` |
| Add Nonescape synthetic-image detection | `.\Install.cmd --with-synthetic` |
| Store new model downloads on another drive | `.\Install.cmd --cache-dir D:\AI-models` |

Optional forensic packages can require additional native build tools. Synthetic detection uses a separate Python environment to keep its dependencies isolated. Details are in the [hardware guide](docs/HARDWARE.md).

<details>
<summary>Complete or retry model downloads after installation</summary>

Windows PowerShell, in the backend folder:

```powershell
.\.venv\Scripts\python.exe -B install\models.py
```

Linux terminal, in the backend folder:

```bash
.venv/bin/python -B install/models.py
```

Add `--group sam3`, `depth`, `pose`, `saliency`, `ollama`, `forgery`, or `synthetic` to prepare only that installed feature.

</details>

## Troubleshooting the backend

| Problem | What to do |
| --- | --- |
| SAM3 access denied / 401 / 403 | Check model approval and the token's account, then retry model preparation. |
| Insufficient disk space | Use a drive with enough space or configure the model cache before installation. |
| CUDA unavailable | Check the NVIDIA driver, or select the CPU profile. |
| Missing Ollama model | Complete model preparation, then start the backend again. |
| Port 8766 already in use | Keep one backend instance running; close the older one before starting another. |
| Synthetic service fails to start | Read the reported error and `.runtime/synthetic-api.log`. Install the optional stage with `--with-synthetic` if it is missing. |
| Start closes immediately | Run `.\Start.cmd` from an existing PowerShell window to read the error. |

<details>
<summary>Development and technical documentation</summary>

Use the installed project Python environment. The API binds to `127.0.0.1:8766` for local use.

```bash
python -m scripts.api.server
python -m unittest discover -s tests -p 'test_*.py'
```

See [architecture](docs/ARCHITECTURE.md), [hardware and model setup](docs/HARDWARE.md), and [validation notes](docs/VALIDATION.md).

</details>

## License

The project's code uses the [MIT license](LICENSE). Model weights, libraries, and analyzed media retain their own terms; see [third-party components](THIRD_PARTY.md).

## Install and build both repositories

Use both projects to analyze new images. To browse saved results, the frontend alone is enough.

### 1. Download both projects first

1. [Download the backend ZIP](https://github.com/In-dialog/image-forensics-backend/archive/refs/heads/main.zip).
2. [Download the frontend ZIP](https://github.com/In-dialog/image-forensics-frontend/archive/refs/heads/main.zip).
3. Extract both ZIPs. Before installing, rename the extracted folders to `image-forensics-backend` and `image-forensics-frontend`. For example:

```text
image-forensics/
  image-forensics-backend/
  image-forensics-frontend/
```

Keep the two folders separate. Do not run either installer from inside a ZIP.

### 2. Install the backend, then build the frontend

**Windows: the automatic setup for both projects**

Use a Windows x64 computer that meets the [backend requirements](https://github.com/In-dialog/image-forensics-backend#backend-requirements). The frontend's Windows ARM support does not extend to the backend.

1. In `image-forensics-backend`, open **Install.cmd**. Complete the model-access prompts and wait for **Setup complete**.
2. In `image-forensics-frontend`, open **Install.cmd**. Wait for **READY**. This installs the frontend dependencies and builds the website automatically.

<details>
<summary>Linux: backend installer and manual frontend build</summary>

Use Linux x64 for the backend and install **Node.js 24 with npm** for the frontend. The frontend's double-click installers are for Windows and macOS; on Linux use these manual commands.

From the parent folder containing both extracted projects:

```bash
cd image-forensics-backend
bash install.sh
cd ../image-forensics-frontend
npm ci --include=dev
npm run build:local
```

</details>

**macOS:** the frontend runs locally, including the saved database viewer. The backend installer does not support macOS, so a Mac alone cannot run this combined local setup. Connecting a Mac frontend to a separate Windows/Linux backend requires a manually configured local tunnel; that advanced setup is outside this walkthrough.

### 3. Start both applications

Keep each application running in its own terminal window:

| Application | Windows | Linux |
| --- | --- | --- |
| Backend: start first, from its folder | Open `Start.cmd` | Run `bash start.sh` |
| Frontend: start second, from its folder | Open `Start.cmd` | Run `npm start` |

Wait for the backend to report `Analysis API: http://127.0.0.1:8766`. Then open the frontend at [http://127.0.0.1:3100](http://127.0.0.1:3100).

### 4. Run an analysis

1. In the frontend, select **New analysis**.
2. Choose an image, select stages, and edit the questions or ranking criteria you need.
3. Select **Start analysis**. The frontend shows the backend's progress, log, and previews.
4. Open the completed result in **View database**. Failed or interrupted runs can be resumed from **Unfinished**.

The frontend sends jobs to the backend through the local API. The backend writes its results to its own `data/` folder; the frontend downloads a separate copy into its own `data/` folder. Saved results remain viewable after the backend stops.

If the frontend says **Backend unavailable**, keep the backend terminal open, check for its API-ready message, and select **Try again**. The default addresses work when both applications run on the same computer.

To stop, press **Ctrl+C** in each application's terminal. Next time, use the two **Start** launchers; installation and building are only needed for initial setup or updates.
