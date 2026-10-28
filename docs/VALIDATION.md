# Release preparation checks

No Python environment, runtime archive, dependency, model or application service was installed/started during preparation. Existing development Python packages were reused for tests; the release folder contains no environment, model cache or saved analysis.

- **129 tests passed** across pipeline execution, structured prompting/results, segmentation identity, previews, persistence, progress, installer planning, archive safety, PhotoHolmes packaging and owned-process cleanup.
- Installer/platform tests use mock hardware, downloads and process handles. Archive tests use tiny generated fixtures. They do not contact package/model servers or start real services.
- Windows `Install.cmd --help` and `--dry-run` ran without creating `.tools`, `.venv`, `.venv-synthetic` or `.install-state.json`. PowerShell/Bash syntax checks passed.
- Core model classes were inspected/imported using the existing development environment without loading pretrained weights. The fresh installer targets Python 3.11; the original development environment uses Python 3.10.
- Unit suites suitable for a standard-library-only Python installation also passed. CI is configured for Windows/Linux Python 3.11 and does not download ML models.
- Dependency metadata was audited, including the separate synthetic environment and the optional PhotoHolmes headless-OpenCV packaging change. No full clean dependency resolution/install was performed here.
- Official runtime archive checksums and version metadata were checked. Large archives were not fetched. Linux Ollama decompression/runtime execution on a fresh machine is therefore not yet verified.

The live HTTP contract suite was deliberately excluded from this preparation run to keep services stopped. Existing API tests remain in the repository for later execution.

Fresh Windows/Linux installation, GPU inference, native optional forensic builds, actual model downloads and the separate synthetic model's inference still need machine-level smoke tests. They are not claimed to have passed. No online repository or deployment has been created.
