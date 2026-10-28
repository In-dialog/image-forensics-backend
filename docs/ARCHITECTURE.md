# Backend architecture

```text
Web interface → local HTTP API → saved job → ordered execution plan
                                       → immutable prompts / checkpoints
                                       → stage runners → result directory
Web interface ← checked artifact download ← completed result directory
```

| Directory | Responsibility |
| --- | --- |
| `scripts/api` | Upload validation, job state, FIFO resume queue, progress, safe result/preview serving. |
| `scripts/configuration` | Validated settings, default profiles, plan creation and local persistence. |
| `scripts/orchestration` | Canonical stage definitions, execution, substep progress, response checkpoints. |
| `scripts/prompting` | Application-owned JSON schemas, editable analysis goals and output validation. |
| `scripts/executers` | Individual image and per-item stages. |
| `scripts/extensions` | Segmentation, composition, depth and forensic operations shared by stages. |
| `scripts/utilities` | Model adapters, context/evidence images, geometry, persistence and cleanup. |
| `scripts/services` | Optional isolated synthetic-analysis service. |
| `install` / `requirements` | Hardware-aware setup, dependency profiles and platform launch support. |
| `tests` | Model-free contract, geometry, persistence, progress and recovery checks. |

`data/.jobs` keeps input/settings/profile snapshots and validated-response checkpoints. It is private runtime state, excluded from Git and artifact exports. A new process marks formerly running jobs interrupted and waits for an explicit resume request.

Detection uses scene views plus region evidence. Per-item descriptions and rankings receive full-scene/nearby/target context, rather than treating a crop as the whole scene. Structured response validators own the schema; user prompts only change analysis goals. Mask overlap/identity handling removes repeated regions without collapsing legitimate nested objects.

The API has no frontend build or desktop UI dependency. The old experimental layered-image tools and parallel hardcoded pipeline are not included. New stages should be registered in the canonical stage definitions and configuration planner, and report progress through the shared progress module.

The optional synthetic model uses its own environment because its dependency pins conflict with the segmentation stack. Optional native forensic packages are installed separately; a core installation must remain usable without them.
