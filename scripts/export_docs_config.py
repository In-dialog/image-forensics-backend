"""Export public defaults for documentation; never read user settings or start models."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.configuration.schema import default_settings
from scripts.orchestration.stages import STAGES
from scripts.prompting import default_profiles, prompt_presets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON here; otherwise print to stdout")
    args = parser.parse_args()
    settings = default_settings()
    config = dict(stages=[dict(key=s.key, label=s.label, group=s.group) for s in STAGES],
                  settings=settings, prompts=default_profiles(), prompt_presets=prompt_presets(),
                  models=sorted({value["model"] for value in settings["stage_options"].values()}),
                  ollama_available=True, active_job_id=None)
    encoded = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
