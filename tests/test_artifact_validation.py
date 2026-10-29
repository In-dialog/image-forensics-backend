"""Validate complete outputs, damaged assets, and partial typed results offline."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from scripts.configuration.forensics import AVAILABLE_METHODS
from scripts.configuration.store import atomic_json
from scripts.orchestration.artifacts import validate_stage_outputs
from scripts.orchestration.execution import RunPlan, StageCall, execute_plan
from scripts.orchestration.stages import STAGES
from scripts.prompting import default_profiles


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profiles = default_profiles()
        self.options = {"image_description": {"prompt_profile": self.profiles["image_description"]},
                        "full_image_rankings": {"prompt_profile": self.profiles["full_image_rankings"]},
                        "per_item_rankings": {"prompt_profile": self.profiles["per_item_rankings"]}}
        self.entry = dict(name="person_0", group="people", bbox=[0, 0, 16, 20], bbox_position=[8, 10, 0.5],
                          image_path="segments/person_0.webp",
                          bbox_composition={side: dict(start=[0, 0], end=[1, 1], distance_px=1)
                                            for side in ("left", "right", "top", "bottom")})
        self.metadata = dict(extracts=[self.entry], image_width=32, image_height=24,
                             source_image_path=str(self.root / "source.png"))
        self.write("metadata.json", self.metadata)
        for path in ("source.png", "segments/person_0.webp", "depth/depth.webp", "negative_space/negative_space.webp",
                     "composition/composition_lines.webp", "original/original.webp",
                     "saliency_deepgaze_msdb/saliency_deepgaze_msdb.webp", "pose/person_0_pose.webp"):
            self.image(path)
        self.write("composition/composition_lines.json", {})
        self.write("granular_segments/people/person_0/segment_metadata.json", {"parent_name": "person_0", "extracts": []})
        self.write("pose/pose.json", {"items": [{"name": "person_0", "points": [], "lines": []}]})
        forensic = {}
        for method in AVAILABLE_METHODS:
            self.image(f"forgery_detection/{method}/heatmap.png")
            forensic[method] = {"score": 0.25, "heatmap_file": "heatmap.png"}
        self.write("forgery_detection.json", forensic)
        self.write("synthetic_analysis.json", {"score": 7.5, "raw_probs": {"real": 0.25, "synthetic": 0.75}})
        for key in self.options:
            profile = self.profiles[key]
            description = key == "image_description"
            descriptor, prefix = ("questions", "question") if description else ("criteria", "criterion")
            fields = [{**entry, "id": f"{prefix}_{i}"} for i, entry in enumerate(profile[descriptor], 1)]
            values = {entry["id"]: None if description else {"value": None, "reason": "Insufficient visible evidence"} for entry in fields}
            item = {"name": "full_image"} if key == "full_image_rankings" else {k: self.entry[k] for k in ("name", "bbox", "image_path")}
            item["answers" if description else "ratings"] = values
            result = dict(schema_version=2, items=[item], **{descriptor: fields})
            result.update({"style": profile["style"]} if description else {"scope": "full_image" if key == "full_image_rankings" else "per_item"})
            self.write(self.filename(key), result)

    @staticmethod
    def filename(key):
        return "descriptions.v2.json" if key == "image_description" else key + ".v2.json"

    def write(self, path, data):
        atomic_json(self.root / path, data)

    def image(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), "white").save(path)

    def validate(self, stage):
        validate_stage_outputs(stage, self.root, self.options.get(stage, {}))

    def test_complete_outputs_for_all_fourteen_stages(self):
        for stage in STAGES:
            with self.subTest(stage=stage.key):
                self.validate(stage.key)

    def test_missing_or_corrupt_assets_invalidate_completed_stages(self):
        paths = {"extract_objects": "segments/person_0.webp", "depth_map": "depth/depth.webp",
                 "analyze_segments": "granular_segments/people/person_0/segment_metadata.json",
                 "pose_extraction": "pose/person_0_pose.webp", "forgery_detection": "forgery_detection/trufor/heatmap.png"}
        for stage, relative in paths.items():
            path = self.root / relative
            saved = path.read_bytes()
            with self.subTest(stage=stage):
                path.unlink()
                with self.assertRaises(ValueError):
                    self.validate(stage)
                path.write_bytes(b"truncated")
                with self.assertRaises(ValueError):
                    self.validate(stage)
                path.write_bytes(saved)

    def test_incomplete_per_item_files_and_missing_answers_fail(self):
        for key in ("image_description", "per_item_rankings"):
            filename = self.filename(key)
            original = json.loads((self.root / filename).read_text())
            with self.subTest(stage=key):
                incomplete = copy.deepcopy(original)
                incomplete["items"] = []
                self.write(filename, incomplete)
                with self.assertRaisesRegex(ValueError, "missing 1 item"):
                    self.validate(key)
                incomplete = copy.deepcopy(original)
                incomplete["items"][0]["answers" if key == "image_description" else "ratings"] = {}
                self.write(filename, incomplete)
                with self.assertRaises(ValueError):
                    self.validate(key)

    def test_profiles_and_item_identities_must_match(self):
        path = self.root / "per_item_rankings.v2.json"
        original = json.loads(path.read_text())
        for change in ("criteria", "duplicate", "coordinates"):
            data = copy.deepcopy(original)
            if change == "criteria":
                data["criteria"][0]["label"] = "Changed category"
            elif change == "duplicate":
                data["items"].append(copy.deepcopy(data["items"][0]))
            else:
                data["items"][0]["bbox"] = [2, 2, 20, 20]
            self.write(path.name, data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate("per_item_rankings")

    def test_failed_or_missing_forensic_check_is_not_complete(self):
        path = self.root / "forgery_detection.json"
        data = json.loads(path.read_text())
        data["trufor"] = {"error": "Out of memory"}
        self.write(path.name, data)
        with self.assertRaisesRegex(ValueError, "Out of memory"):
            self.validate("forgery_detection")
        del data["trufor"]
        self.write(path.name, data)
        with self.assertRaisesRegex(ValueError, "Missing forensic result"):
            self.validate("forgery_detection")

    def test_path_escape_and_missing_part_crops_are_rejected(self):
        relative = "granular_segments/people/person_0/segment_metadata.json"
        part = {"name": "part", "bbox": [0, 0, 3, 3], "image_path": "../outside.png"}
        self.write(relative, {"parent_name": "person_0", "extracts": [part]})
        with self.assertRaisesRegex(ValueError, "outside this analysis"):
            self.validate("analyze_segments")
        del part["image_path"]
        self.write(relative, {"parent_name": "person_0", "extracts": [part]})
        with self.assertRaisesRegex(ValueError, "unreadable image"):
            self.validate("analyze_segments")

    def test_invalid_json_and_nonfinite_scores_fail(self):
        for content in ('{"score": NaN}', '{"score": 0, "score": 1}', '{'):
            (self.root / "synthetic_analysis.json").write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.validate("synthetic_analysis")

    def test_runner_return_does_not_mark_incomplete_output_successful(self):
        (self.root / "depth/depth.webp").unlink()
        later, progress = Mock(), Mock()
        with self.assertRaisesRegex(RuntimeError, "Depth Map failed"):
            execute_plan(RunPlan(self.root, (StageCall("depth_map"), StageCall("original_export"))),
                         runners={"depth_map": Mock(), "original_export": later},
                         validate_outputs=validate_stage_outputs, on_progress=progress)
        later.assert_not_called()
        self.assertEqual([call.args[-1] for call in progress.call_args_list], ["running"])


if __name__ == "__main__":
    unittest.main()
