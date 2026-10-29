# Release preparation checks

Checks used the existing development Python environment. No Python environment, runtime archive, dependency package, or model was installed into this release folder. Temporary test services were shut down after verification.

- The full regression suite passed across pipeline execution, structured results, segmentation identity, persistence, progress, installer planning, archive safety, packaging, GPU cleanup, and service startup/shutdown.
- Installer/platform tests use mock hardware and downloads. Archive tests use tiny generated fixtures; these checks do not contact package or model servers.
- Windows `Install.cmd --help` and `--dry-run` ran without creating `.tools`, `.venv`, `.venv-synthetic` or `.install-state.json`. PowerShell/Bash syntax checks passed.
- Core model classes were inspected/imported using the existing development environment without loading pretrained weights. The fresh installer targets Python 3.11; the original development environment uses Python 3.10.
- The real synthetic API started successfully on a temporary local port with downloads disabled. Health, idle unloading, and process cleanup were checked using an existing interpreter; no classifier weights were loaded.
- A real CUDA allocation test used 128 MiB at peak and returned to zero PyTorch allocated/reserved bytes after success, a simulated failure, and a subsequent retry. This checks resource cleanup, not full model inference.
- Follow-up checks on the development installation used existing cached weights with model downloads disabled: the real synthetic stage produced a valid result and released its classifier. Separate GPU tests reproduced residual TruFor allocations in a long-lived process; isolated TruFor and PSCCNet workers then exited successfully and left no worker process holding GPU resources. These checks do not cover every optional method or a fresh installation.
- Recovery tests cover corrupt input images, changed input checksums, damaged extraction metadata, missing image assets, incomplete per-item results, stale criteria, invalid probabilities, and failures before later stages can start. Synthetic HTTP tests cover timeout/reconnection, duplicate-request prevention, and retrying failed tasks. Worker tests cover subprocess errors, timeouts, and cleanup.
- Unit suites suitable for a standard-library-only Python installation also passed. CI is configured for Windows/Linux Python 3.11 and does not download ML models.
- Dependency metadata was audited, including the separate synthetic environment and the optional PhotoHolmes headless-OpenCV packaging change. No full clean dependency resolution/install was performed here.
- Official runtime archive checksums and version metadata were checked. Large archives were not fetched. Linux Ollama decompression/runtime execution on a fresh machine is therefore not yet verified.

HTTP contract tests use temporary local services. Symlink-dependent checks are skipped on Windows when file symlinks are unavailable; the corresponding tests remain available for supported environments.

Fresh Windows/Linux installation, complete analyses using every optional method, native optional forensic builds, and model downloads still need machine-level checks. Cached-model smoke tests used the existing development environment, including a real CUDA worker launched from the release source with its configured weights cache. These checks do not establish that every model and optional feature works on a new computer.
