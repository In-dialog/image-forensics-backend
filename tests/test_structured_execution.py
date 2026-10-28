"""Exercise structured model transport and runner persistence without model/GPU work."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Keep NumPy loaded while scoped module stubs are restored between runner tests.
import numpy
from PIL import Image

from scripts.prompting import ITEM_CONTEXT, compile_description, compile_detection, compile_rankings, default_profiles, validate_response
from scripts.orchestration.progress import stage_progress
from scripts.orchestration.checkpoints import checkpoint_scope

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vision = load_module("structured_test_vision", "scripts/utilities/ollama_vision.py")


def ranking_profile():
    return {"kind": "ranking", "positive": "Inspect visible evidence.", "negative": "Do not guess.", "criteria": [
        {"label": "Intensity", "type": "score", "positive": "Visual intensity", "negative": "",
         "min": 0, "max": 10, "low_label": "Low", "high_label": "High"},
        {"label": "Tone", "type": "choice", "positive": "Visible tone", "negative": "",
         "choices": ["Positive", "Neutral", "Negative"]},
        {"label": "Text visible", "type": "boolean", "positive": "Readable text is present", "negative": ""},
    ]}


def rating_response():
    return {"ratings": {"criterion_1": {"value": 7.5, "reason": "High contrast edges."},
                        "criterion_2": {"value": "Neutral", "reason": "No emotion is visibly expressed."},
                        "criterion_3": {"value": False, "reason": "No readable letters."}}}


class StructuredExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.image = self.folder / "source.png"
        Image.new("RGB", (80, 80), "white").save(self.image)
        self.ollama = types.ModuleType("ollama")
        self.ollama.chat = Mock()
        self.ollama.generate = Mock()
        self.patch = patch.dict(sys.modules, {"ollama": self.ollama})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def call(self, compiled, responses):
        self.ollama.chat.side_effect = [{"message": {"content": value if isinstance(value, str) else json.dumps(value)}} for value in responses]
        return vision.structured_vision_chat(
            compiled.user, self.image, system_prompt=compiled.system, response_schema=compiled.schema,
            validate=lambda data: validate_response(compiled, data), keep_alive="120s", temperature=0,
        )

    def test_mixed_types_are_not_coerced_and_schema_is_sent_to_ollama(self):
        compiled = compile_rankings(ranking_profile())
        response = rating_response()
        self.assertEqual(self.call(compiled, [response]), response)
        kwargs = self.ollama.chat.call_args.kwargs
        self.assertEqual(kwargs["format"], compiled.schema)
        self.assertEqual(kwargs["keep_alive"], "120s")
        self.assertEqual([message["role"] for message in kwargs["messages"]], ["system", "user"])
        self.assertNotIn("images", kwargs["messages"][0])
        self.assertEqual(kwargs["messages"][1]["images"], [str(self.image.resolve())])
        self.assertIs(kwargs["think"], False)
        self.assertNotIn("think", kwargs["options"])

    def test_score_scales_accept_values_within_each_configured_range(self):
        for minimum, maximum, value in [(0, 10, 7.5), (-1, 1, -0.5), (1, 5, 4)]:
            with self.subTest(scale=(minimum, maximum)):
                profile = ranking_profile()
                profile["criteria"][0].update(min=minimum, max=maximum)
                response = rating_response()
                response["ratings"]["criterion_1"]["value"] = value
                self.assertEqual(self.call(compile_rankings(profile), [response]), response)

    def test_global_cpu_disables_ollama_gpu_and_other_modes_delegate_placement(self):
        for device, expected in [("cpu", 0), ("cuda", -1), ("auto", None)]:
            with self.subTest(device=device):
                self.ollama.chat.side_effect = None
                self.ollama.chat.return_value = {"message": {"content": "{}"}}
                vision.vision_chat("Observe", self.image, vision_device=device, presence_penalty=0.375)
                self.assertEqual(self.ollama.chat.call_args.kwargs["options"].get("num_gpu"), expected)
                self.assertEqual(self.ollama.chat.call_args.kwargs["options"]["presence_penalty"], 0.375)

    def test_tiny_primary_and_detail_inputs_are_upscaled_in_memory_without_changing_files(self):
        primary = self.folder / "tiny.png"
        detail = self.folder / "detail.png"
        Image.new("RGB", (93, 27), "red").save(primary)
        Image.new("RGBA", (32, 64), (0, 0, 255, 128)).save(detail)
        before = {path: path.read_bytes() for path in (primary, detail, self.image)}
        self.ollama.chat.return_value = {"message": {"content": "{}"}}
        vision.vision_chat("Inspect", primary, detail_images=[detail, self.image])
        images = self.ollama.chat.call_args.kwargs["messages"][-1]["images"]
        for payload, original_size in zip(images[:2], [(93, 27), (32, 64)]):
            self.assertIsInstance(payload, bytes)
            with Image.open(io.BytesIO(payload)) as resized:
                self.assertEqual(resized.format, "PNG")
                self.assertGreaterEqual(min(resized.size), 64)
                self.assertEqual(resized.width * original_size[1], resized.height * original_size[0])
        self.assertEqual(images[2], str(self.image.resolve()))
        for path, original_bytes in before.items():
            self.assertEqual(path.read_bytes(), original_bytes)

    def test_thirty_three_pixel_input_keeps_original_path(self):
        path = self.folder / "sufficient.png"
        Image.new("RGB", (93, 33), "white").save(path)
        self.ollama.chat.return_value = {"message": {"content": "{}"}}
        vision.vision_chat("Inspect", path)
        self.assertEqual(self.ollama.chat.call_args.kwargs["messages"][-1]["images"], [str(path.resolve())])

    def test_invalid_content_retries_once_without_accepting_bad_values(self):
        valid = rating_response()
        for bad_value in (11, -1, True, "7.5"):
            with self.subTest(value=bad_value):
                self.ollama.chat.reset_mock()
                invalid = rating_response()
                invalid["ratings"]["criterion_1"]["value"] = bad_value
                self.assertEqual(self.call(compile_rankings(ranking_profile()), [invalid, valid]), valid)
                self.assertEqual(self.ollama.chat.call_count, 2)
                self.assertIn("previous response", self.ollama.chat.call_args.kwargs["messages"][-1]["content"])

    def test_invalid_json_duplicate_keys_and_nonfinite_numbers_fail_bounded(self):
        compiled = compile_rankings(ranking_profile())
        invalid_responses = ["Score: 7", '{"ratings":{},"ratings":{}}', '{"ratings":NaN}', "```json\n{}\n```"]
        for invalid in invalid_responses:
            with self.subTest(response=invalid):
                self.ollama.chat.reset_mock()
                with self.assertRaises(vision.StructuredResponseError):
                    self.call(compiled, [invalid, invalid])
                self.assertEqual(self.ollama.chat.call_count, 2)

    def test_transport_failure_is_not_retried_as_bad_content(self):
        self.ollama.chat.side_effect = ConnectionError("Ollama unavailable")
        compiled = compile_rankings(ranking_profile())
        with self.assertRaises(ConnectionError):
            vision.structured_vision_chat(compiled.user, self.image, system_prompt=compiled.system,
                                          response_schema=compiled.schema, validate=lambda value: validate_response(compiled, value))
        self.assertEqual(self.ollama.chat.call_count, 1)

    def test_unknown_values_remain_null_for_every_ranking_type(self):
        response = rating_response()
        for rating in response["ratings"].values():
            rating.update(value=None, reason="The relevant detail is obscured.")
        self.assertEqual(self.call(compile_rankings(ranking_profile()), [response]), response)

    def test_description_styles_share_a_protected_answer_contract(self):
        profile = default_profiles()["image_description"]
        for style in ("concise", "detailed", "bullets"):
            with self.subTest(style=style):
                profile["style"] = style
                compiled = compile_description(profile, "objects")
                response = {"answers": {f"question_{index}": "A visible red object." if index == 1 else None
                                        for index in range(1, len(profile["questions"]) + 1)}}
                self.assertEqual(self.call(compiled, [response]), response)

    def test_detection_retries_reversed_box_and_accepts_valid_or_empty_items(self):
        compiled = compile_detection(default_profiles()["object_detect"], "objects", max_items=2)
        invalid = {"items": [{"label": "Red chair", "box": [500, 10, 100, 900]}]}
        valid = {"items": [{"label": "Red chair", "box": [100, 10, 500, 900]}]}
        self.assertEqual(self.call(compiled, [invalid, valid]), valid)
        self.assertEqual(self.call(compiled, [{"items": []}]), {"items": []})

    def load_runners(self):
        # Isolate model-free runners from the package's historical eager ML imports.
        utilities = types.ModuleType("scripts.utilities")
        utilities.__path__ = [str(ROOT / "scripts" / "utilities")]
        extensions = types.ModuleType("scripts.extensions")
        extensions.__path__ = [str(ROOT / "scripts" / "extensions")]
        extensions.get_image_path_from_metadata = lambda _: self.image
        save_stage = types.ModuleType("scripts.utilities.save_stage_utils")
        save_stage.METADATA_FILENAME, save_stage.SEGMENTS_SUBFOLDER = "metadata.json", "segments"
        images = types.ModuleType("scripts.utilities.image_utils")
        images.load_rgb = lambda path: Image.open(path).convert("RGB")
        modules = {"scripts.utilities": utilities, "scripts.utilities.ollama_vision": vision,
                   "scripts.utilities.save_stage_utils": save_stage, "scripts.utilities.image_utils": images,
                   "scripts.extensions": extensions}
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        results = load_module("scripts.utilities.structured_results", "scripts/utilities/structured_results.py")
        sys.modules["scripts.utilities.structured_results"] = results
        return results

    def prepare_item(self):
        (self.folder / "segments").mkdir()
        Image.new("RGB", (60, 60), "red").save(self.folder / "segments" / "chair.png")
        (self.folder / "metadata.json").write_text(json.dumps({"image_width": 80, "image_height": 80,
            "source_image_path": str(self.image),
            "extracts": [{"name": "chair", "group": "objects", "bbox": [5, 5, 65, 65]}]}), encoding="utf-8")

    def test_full_and_per_item_runners_save_only_canonical_typed_results(self):
        self.load_runners()
        self.prepare_item()
        for filename, relative, typed_name in [
            ("full_image_rankings.json", "scripts/executers/global/rankings/run_full_image_rankings.py", "full_image_rankings.v2.json"),
            ("rankong_per_item.json", "scripts/executers/global/rankings/run_rankings.py", "per_item_rankings.v2.json"),
        ]:
            with self.subTest(runner=relative):
                module = load_module("test_ranking_runner", relative)
                self.ollama.chat.side_effect = [{"message": {"content": json.dumps(rating_response())}}]
                module.execute(self.folder, prompt_profile=ranking_profile(), vision_device="cpu")
                request = self.ollama.chat.call_args.kwargs
                is_scene = "full_image" in typed_name
                self.assertEqual(request["model"], "qwen3-vl" if is_scene else "qwen3-vl:8b-instruct")
                self.assertEqual(request["think"], is_scene)
                self.assertEqual(request["options"]["num_ctx"], 12288 if is_scene else 8192)
                self.assertEqual(request["options"]["num_predict"], 6144 if is_scene else 2048)
                self.assertEqual(request["options"]["temperature"], 0.6 if is_scene else 0.7)
                self.assertEqual(request["options"]["top_p"], 0.95 if is_scene else 0.8)
                self.assertEqual(request["options"]["top_k"], 20)
                self.assertEqual(request["options"]["repeat_penalty"], 1)
                self.assertEqual(request["options"]["presence_penalty"], 0 if is_scene else 1.5)
                self.assertEqual(self.ollama.chat.call_args.kwargs["options"]["num_gpu"], 0)
                typed = json.loads((self.folder / typed_name).read_text())
                self.assertEqual(typed["items"][0]["ratings"], rating_response()["ratings"])
                self.assertEqual(typed["criteria"][2]["type"], "boolean")
                self.assertFalse((self.folder / filename).exists())

    def test_description_runner_saves_all_answers_in_one_call(self):
        self.load_runners()
        self.prepare_item()
        module = load_module("test_description_runner", "scripts/executers/per_Item/run_image_description.py")
        profile = default_profiles()["image_description"]
        answers = {f"question_{i}": "Visible red surface." if i == 1 else None for i in range(1, len(profile["questions"]) + 1)}
        self.ollama.chat.side_effect = [{"message": {"content": json.dumps({"answers": answers})}}]
        module.execute(self.folder, prompt_profile=profile, vision_device="cpu")
        self.assertEqual(self.ollama.chat.call_args.kwargs["options"]["num_gpu"], 0)
        self.assertEqual(self.ollama.chat.call_args.kwargs["options"]["presence_penalty"], 1.5)
        self.assertEqual(self.ollama.chat.call_count, 1)
        typed = json.loads((self.folder / "descriptions.v2.json").read_text())
        self.assertEqual(typed["items"][0]["name"], "chair")
        self.assertEqual(typed["items"][0]["answers"], answers)
        self.assertEqual(typed["items"][0]["image_path"], "segments/chair.png")
        self.assertEqual(typed["items"][0]["source"], "item")
        self.assertFalse((self.folder / "description.json").exists())

    def test_invalid_runner_response_fails_without_writing_result_files(self):
        self.load_runners()
        self.prepare_item()
        module = load_module("test_failed_runner", "scripts/executers/global/rankings/run_full_image_rankings.py")
        self.ollama.chat.side_effect = [{"message": {"content": "{}"}}] * 2
        with self.assertRaises(vision.StructuredResponseError):
            module.execute(self.folder, prompt_profile=ranking_profile())
        self.assertFalse((self.folder / "full_image_rankings.json").exists())
        self.assertFalse((self.folder / "full_image_rankings.v2.json").exists())

    def test_description_and_ranking_include_granular_children_with_image_and_parent_context(self):
        self.load_runners()
        self.prepare_item()
        child_folder = self.folder / "granular_segments" / "objects" / "chair"
        child_folder.mkdir(parents=True)
        Image.new("RGB", (40, 40), "blue").save(child_folder / "mark_0.png")
        (child_folder / "segment_metadata.json").write_text(json.dumps({
            "parent_name": "chair", "parent_bbox": [5, 5, 65, 65],
            "extracts": [{"name": "mark_0", "label": "mark", "group": "mark", "bbox": [10, 10, 50, 50]}],
        }), encoding="utf-8")
        description = load_module("test_nested_description", "scripts/executers/per_Item/run_image_description.py")
        profile = default_profiles()["image_description"]
        answers = {f"question_{i}": "A blue mark." for i in range(1, len(profile["questions"]) + 1)}
        self.ollama.chat.side_effect = [{"message": {"content": json.dumps({"answers": answers})}}] * 2
        updates = []
        with stage_progress(updates.append):
            description.execute(self.folder, prompt_profile=profile)
        self.assertEqual((updates[-1]["substeps"][0]["completed"], updates[-1]["substeps"][0]["total"]), (2, 2))
        self.assertTrue(any(update["substeps"][0].get("current") == "chair::mark_0" for update in updates))
        typed = json.loads((self.folder / "descriptions.v2.json").read_text())
        self.assertEqual(len(typed["items"]), 2)
        child = typed["items"][1]
        self.assertEqual(child["name"], "chair::mark_0")
        self.assertEqual(child["parent_name"], "chair")
        self.assertEqual(child["image_path"], "granular_segments/objects/chair/mark_0.png")
        self.assertEqual(child["bbox"], [15, 15, 55, 55])
        ranking = load_module("test_nested_ranking", "scripts/executers/global/rankings/run_rankings.py")
        self.ollama.chat.side_effect = [{"message": {"content": json.dumps(rating_response())}}] * 2
        updates = []
        with stage_progress(updates.append):
            ranking.execute(self.folder, prompt_profile=ranking_profile())
        self.assertEqual((updates[-1]["substeps"][0]["completed"], updates[-1]["substeps"][0]["total"]), (2, 2))
        self.assertEqual(updates[-1]["current_substep"], 2)
        typed = json.loads((self.folder / "per_item_rankings.v2.json").read_text())
        self.assertEqual([item["source"] for item in typed["items"]], ["item", "part"])
        self.assertIn(ITEM_CONTEXT, self.ollama.chat.call_args.kwargs["messages"][1]["content"])

    def test_validated_checkpoints_match_image_prompt_and_model_and_reject_corruption(self):
        results = self.load_runners()
        compiled = compile_rankings(ranking_profile())
        self.ollama.chat.side_effect = None
        self.ollama.chat.return_value = {"message": {"content": json.dumps(rating_response())}}
        checkpoint_dir = self.folder / "checkpoints"
        with checkpoint_scope(checkpoint_dir):
            results.run_compiled(compiled, self.image, model="model-a", keep_alive="120s")
            results.run_compiled(compiled, self.image, model="model-a", keep_alive="60s")
            self.assertEqual(self.ollama.chat.call_count, 1)
            next(checkpoint_dir.glob("*.json")).write_text('{"version":1,"response":{"ratings":{}}}', encoding="utf-8")
            results.run_compiled(compiled, self.image, model="model-a")
            self.assertEqual(self.ollama.chat.call_count, 2)
            results.run_compiled(compiled, self.image, model="model-b")
            Image.new("RGB", (80, 80), "blue").save(self.image)
            results.run_compiled(compiled, self.image, model="model-b")
            changed = ranking_profile()
            changed["positive"] = "Inspect only visible geometry."
            results.run_compiled(compile_rankings(changed), self.image, model="model-b")
            self.assertEqual(self.ollama.chat.call_count, 5)

    def test_item_failure_saves_completed_results_and_resume_reuses_them(self):
        self.load_runners()
        self.prepare_item()
        metadata = json.loads((self.folder / "metadata.json").read_text())
        metadata["extracts"].append({**metadata["extracts"][0], "name": "second", "bbox": [10, 10, 75, 75]})
        (self.folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        description = load_module("test_resumed_description", "scripts/executers/per_Item/run_image_description.py")
        profile = default_profiles()["image_description"]
        response = {"message": {"content": json.dumps({"answers": {f"question_{i}": "Visible surface." for i in range(1, len(profile["questions"]) + 1)}})}}
        self.ollama.chat.side_effect = [response, RuntimeError("Model disconnected")]
        with checkpoint_scope(self.folder / "checkpoints"):
            with self.assertRaisesRegex(RuntimeError, "Model disconnected"):
                description.execute(self.folder, prompt_profile=profile)
            partial = json.loads((self.folder / "descriptions.v2.json").read_text())
            self.assertEqual([item["name"] for item in partial["items"]], ["chair"])
            self.ollama.chat.side_effect = [response]
            description.execute(self.folder, prompt_profile=profile)
            self.assertEqual(self.ollama.chat.call_count, 3)
        complete = json.loads((self.folder / "descriptions.v2.json").read_text())
        self.assertEqual([item["name"] for item in complete["items"]], ["chair", "second"])

    def test_downstream_prompts_do_not_repeat_detector_guesses_but_result_identity_is_preserved(self):
        self.load_runners()
        parent_name = "unsupported_parent_zyx"
        child_name = "unsupported_child_zyx"
        (self.folder / "segments").mkdir()
        Image.new("RGB", (60, 60), "white").save(self.folder / "segments" / f"{parent_name}.png")
        (self.folder / "metadata.json").write_text(json.dumps({
            "source_image_path": str(self.image), "image_width": 80, "image_height": 80,
            "extracts": [{"name": parent_name, "label": "unsupported_identity_zyx", "group": "unsupported_country_zyx", "bbox": [5, 5, 65, 65]}],
        }), encoding="utf-8")
        child_folder = self.folder / "granular_segments" / "objects" / parent_name
        child_folder.mkdir(parents=True)
        Image.new("RGB", (40, 40), "white").save(child_folder / f"{child_name}.png")
        (child_folder / "segment_metadata.json").write_text(json.dumps({
            "parent_name": parent_name, "parent_bbox": [5, 5, 65, 65],
            "extracts": [{"name": child_name, "label": "unsupported_symbol_zyx", "group": "unsupported_affiliation_zyx", "bbox": [10, 10, 50, 50]}],
        }), encoding="utf-8")
        profile = default_profiles()["image_description"]
        answers = {f"question_{i}": "A visible shape." for i in range(1, len(profile["questions"]) + 1)}
        cases = [
            ("scripts/executers/per_Item/run_image_description.py", profile, {"answers": answers}, "descriptions.v2.json"),
            ("scripts/executers/global/rankings/run_rankings.py", ranking_profile(), rating_response(), "per_item_rankings.v2.json"),
        ]
        for path, selected_profile, response, output in cases:
            with self.subTest(runner=path):
                self.ollama.chat.reset_mock()
                self.ollama.chat.side_effect = [{"message": {"content": json.dumps(response)}}] * 2
                runner = load_module("test_neutral_context", path)
                runner.execute(self.folder, prompt_profile=selected_profile)
                for call in self.ollama.chat.call_args_list:
                    for message in call.kwargs["messages"]:
                        self.assertNotIn("unsupported_", message["content"])
                result = json.loads((self.folder / output).read_text())
                self.assertEqual(result["items"][0]["name"], parent_name)
                self.assertEqual(result["items"][0]["label"], "unsupported_identity_zyx")
                self.assertEqual(result["items"][0]["group"], "unsupported_country_zyx")
                self.assertEqual(result["items"][1]["parent_name"], parent_name)
                self.assertEqual(result["items"][1]["label"], "unsupported_symbol_zyx")

    def test_description_and_ranking_send_one_context_board_without_changing_original_or_viewer_image(self):
        self.load_runners()
        self.prepare_item()
        metadata_path = self.folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["source_image_path"] = str(self.image)
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        original_bytes = self.image.read_bytes()
        description = load_module("test_original_description", "scripts/executers/per_Item/run_image_description.py")
        profile = default_profiles()["image_description"]
        answers = {f"question_{i}": "A white surface." for i in range(1, len(profile["questions"]) + 1)}
        self.ollama.chat.side_effect = [{"message": {"content": json.dumps({"answers": answers})}}]
        description.execute(self.folder, prompt_profile=profile)
        kwargs = self.ollama.chat.call_args.kwargs
        self.assertEqual(kwargs["keep_alive"], "5m")
        self.assertEqual(len(kwargs["messages"][-1]["images"]), 1)
        board_path = Path(kwargs["messages"][-1]["images"][0])
        self.assertTrue(board_path.is_relative_to(self.folder / "analysis_evidence"))
        self.assertRegex(board_path.name, r"^board-[0-9a-f]{24}\.png$")
        self.assertIn(ITEM_CONTEXT, kwargs["messages"][-1]["content"])
        with Image.open(board_path) as board:
            self.assertEqual(board.size, (288, 136))
            self.assertEqual(board.getpixel((17, 49)), (0, 255, 255))
            self.assertEqual(board.getpixel((240, 80)), (255, 255, 255))
        typed = json.loads((self.folder / "descriptions.v2.json").read_text())
        self.assertEqual(typed["items"][0]["image_path"], "segments/chair.png")
        self.ollama.generate.assert_called_once_with(model="qwen3-vl:8b-instruct", prompt="", keep_alive=0)
        ranking = load_module("test_detail_ranking", "scripts/executers/global/rankings/run_rankings.py")
        self.ollama.chat.side_effect = [{"message": {"content": json.dumps(rating_response())}}]
        ranking.execute(self.folder, prompt_profile=ranking_profile())
        kwargs = self.ollama.chat.call_args.kwargs
        self.assertEqual(kwargs["messages"][-1]["images"], [str(board_path)])
        self.assertIn(ITEM_CONTEXT, kwargs["messages"][-1]["content"])
        self.assertNotIn("bbox_2d", kwargs["messages"][-1]["content"])
        self.assertNotIn("Assess only the supplied original-pixel crop.", kwargs["messages"][-1]["content"])
        self.assertEqual(kwargs["keep_alive"], "5m")
        self.assertEqual(self.ollama.generate.call_count, 2)
        self.assertEqual(self.image.read_bytes(), original_bytes)
        typed = json.loads((self.folder / "per_item_rankings.v2.json").read_text())
        self.assertEqual(typed["items"][0]["image_path"], "segments/chair.png")

    def test_semantic_runners_apply_minimum_to_original_box_not_enlarged_board(self):
        self.load_runners()
        self.prepare_item()
        path = self.folder / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["extracts"][0]["bbox"] = [5, 5, 10, 10]
        path.write_text(json.dumps(metadata), encoding="utf-8")
        for relative, output in [("scripts/executers/per_Item/run_image_description.py", "descriptions.v2.json"),
                                 ("scripts/executers/global/rankings/run_rankings.py", "per_item_rankings.v2.json")]:
            with self.subTest(runner=relative):
                runner = load_module("test_minimum_semantic_region", relative)
                runner.execute(self.folder)
                self.ollama.chat.assert_not_called()
                self.assertEqual(json.loads((self.folder / output).read_text())["items"], [])
                self.assertFalse((self.folder / "analysis_evidence").exists())

    def test_description_and_ranking_sessions_unload_after_validation_failure(self):
        self.load_runners()
        self.prepare_item()
        cases = [
            ("scripts/executers/per_Item/run_image_description.py", default_profiles()["image_description"]),
            ("scripts/executers/global/rankings/run_rankings.py", ranking_profile()),
        ]
        for path, profile in cases:
            with self.subTest(runner=path):
                self.ollama.generate.reset_mock()
                self.ollama.chat.side_effect = [{"message": {"content": "{}"}}] * 2
                runner = load_module("test_failed_session", path)
                with self.assertRaises(vision.StructuredResponseError):
                    runner.execute(self.folder, prompt_profile=profile)
                self.ollama.generate.assert_called_once_with(model="qwen3-vl:8b-instruct", prompt="", keep_alive=0)

    def test_extraction_converts_validated_json_directly_to_sam_calls(self):
        self.load_runners()
        module = load_module("scripts.extensions.extractor_extensions", "scripts/extensions/extractor_extensions.py")
        response = {"items": [{"label": "Red chair", "box": [100, 10, 500, 900]}]}
        self.ollama.chat.side_effect = [{"message": {"content": json.dumps(response)}}]
        calls = module.vision_calls_for_group(self.image, "objects", prompt_profile=default_profiles()["object_detect"], vision_device="cpu")
        self.assertEqual(calls, [("objects", "Red chair", (100, 10, 500, 900))])
        self.assertEqual(self.ollama.chat.call_args.kwargs["options"]["num_gpu"], 0)
        with self.assertRaises(TypeError):
            module.vision_calls_for_group(self.image, "objects", prompt_override="Change output shape")

    def test_every_detection_call_has_a_bounded_scene_schema(self):
        self.load_runners()
        detector = load_module("scripts.extensions.extractor_extensions", "scripts/extensions/extractor_extensions.py")
        for group in ("people", "objects", "texts"):
            self.ollama.chat.side_effect = [{"message": {"content": '{"items":[]}'}}]
            detector.vision_calls_for_group(self.image, group, max_calls_for_group=50)
            self.assertEqual(self.ollama.chat.call_args.kwargs["format"]["properties"]["items"]["maxItems"], 16)

    def test_repeated_ocr_is_merged_without_retry_and_trace_records_the_duplicate(self):
        self.load_runners()
        detector = load_module("scripts.extensions.extractor_extensions", "scripts/extensions/extractor_extensions.py")
        item = {"label": "VISIBLE", "box": [100, 100, 300, 200]}
        self.ollama.chat.side_effect = [{"message": {"content": json.dumps({"items": [item, item]})}}]
        folder = self.folder / "vision_details"
        calls = detector.vision_calls_for_group(self.image, "texts", detail_dir=folder)
        self.assertEqual(calls, [("texts", "VISIBLE", (100, 100, 300, 200))])
        self.assertEqual(self.ollama.chat.call_count, 1)
        trace = json.loads((folder / "texts_detection_passes.json").read_text())
        self.assertEqual(trace["passes"][0]["duplicates_removed"], 1)
        self.assertEqual(len(trace["passes"][0]["local_detections"]), 2)
        self.assertEqual(trace["merged_count"], 1)

    def test_failed_detection_preserves_its_contract_source_and_error_trace(self):
        self.load_runners()
        detector = load_module("scripts.extensions.extractor_extensions", "scripts/extensions/extractor_extensions.py")
        self.ollama.chat.side_effect = [{"message": {"content": '{}'}}] * 2
        folder = self.folder / "vision_details"
        with self.assertRaises(vision.StructuredResponseError):
            detector.vision_calls_for_group(self.image, "people", detail_dir=folder)
        trace = json.loads((folder / "people_detection_passes.json").read_text())
        self.assertEqual(trace["passes"][0]["status"], "failed")
        self.assertIn("failed validation", trace["passes"][0]["error"])
        self.assertEqual(trace["passes"][0]["source_image"], str(self.image.resolve()))
        self.assertIn("schema", trace["passes"][0]["compiled"])
        self.assertEqual(trace["merged_count"], 0)

    def test_scene_object_detail_passes_are_bounded_remapped_and_traced(self):
        self.load_runners()
        Image.new("RGB", (1024, 768), "white").save(self.image)
        detector = load_module("scripts.extensions.extractor_extensions", "scripts/extensions/extractor_extensions.py")
        responses = [{"items": [{"label": f"visible region {index}", "box": [100, 100, 300, 300]}]} for index in range(5)]
        self.ollama.chat.side_effect = [{"message": {"content": json.dumps(value)}} for value in responses]
        folder = self.folder / "vision_details"
        calls = detector.vision_calls_for_group(self.image, "objects", max_calls_for_group=50,
                                               scene_detail_pass=True, detail_dir=folder)
        self.assertEqual(len(calls), 5)
        self.assertEqual(self.ollama.chat.call_count, 5)
        caps = [call.kwargs["format"]["properties"]["items"]["maxItems"] for call in self.ollama.chat.call_args_list]
        self.assertEqual(caps, [16, 8, 8, 8, 8])
        self.assertEqual(len(list(folder.glob("*.png"))), 4)
        trace = json.loads((folder / "object_detection_passes.json").read_text())
        self.assertEqual(len(trace["passes"]), 5)
        self.assertEqual(trace["passes"][1]["local_detections"], responses[1]["items"])
        self.assertEqual(trace["passes"][1]["full_image_detections"][0]["box"], list(calls[1][2]))
        self.assertNotEqual(calls[1][2], (100, 100, 300, 300))
        self.assertEqual(trace["passes"][0]["crop_pixel_box"], [0, 0, 1024, 768])
        self.ollama.generate.assert_called_once_with(model="qwen3-vl:8b-instruct", prompt="", keep_alive=0)

    def test_detail_passes_stop_at_configured_total(self):
        self.load_runners()
        Image.new("RGB", (512, 512), "white").save(self.image)
        detector = load_module("scripts.extensions.extractor_extensions", "scripts/extensions/extractor_extensions.py")
        scene = {"items": [{"label": f"region {index}", "box": [index * 100, 0, index * 100 + 50, 100]} for index in range(3)]}
        detail = {"items": [{"label": "detail", "box": [500, 500, 600, 600]}]}
        self.ollama.chat.side_effect = [{"message": {"content": json.dumps(value)}} for value in (scene, detail)]
        calls = detector.vision_calls_for_group(self.image, "objects", max_calls_for_group=4,
                                               scene_detail_pass=True, detail_dir=self.folder / "views")
        self.assertEqual(len(calls), 4)
        self.assertEqual(self.ollama.chat.call_count, 2)
        self.assertEqual(self.ollama.chat.call_args.kwargs["format"]["properties"]["items"]["maxItems"], 1)

    def test_detail_views_are_disabled_for_people_texts_granular_calls_and_small_images(self):
        self.load_runners()
        detector = load_module("scripts.extensions.extractor_extensions", "scripts/extensions/extractor_extensions.py")
        for group, enabled, size in [("people", True, 1024), ("texts", True, 1024),
                                     ("objects", False, 1024), ("objects", True, 511)]:
            with self.subTest(group=group, enabled=enabled, size=size):
                Image.new("RGB", (size, size), "white").save(self.image)
                self.ollama.chat.reset_mock()
                self.ollama.chat.side_effect = [{"message": {"content": '{"items":[]}'}}]
                detector.vision_calls_for_group(self.image, group, max_calls_for_group=50,
                                               scene_detail_pass=enabled, detail_dir=self.folder / "views")
                self.assertEqual(self.ollama.chat.call_count, 1)

    def test_extraction_and_granular_runners_forward_profiles_without_loading_sam_for_empty_results(self):
        self.load_runners()
        self.prepare_item()
        extractor = load_module("scripts.extensions.extractor_extensions", "scripts/extensions/extractor_extensions.py")
        sys.modules["scripts.extensions.extractor_extensions"] = extractor
        extensions = sys.modules["scripts.extensions"]
        for name in ("DETECTION_GROUPS", "labels_from_plan", "run_sam_calls_with_fallback", "save_sam_results_overlay",
                     "save_vision_calls_debug_json", "save_vision_calls_overlay", "vision_calls_for_group"):
            setattr(extensions, name, getattr(extractor, name))
        utilities = sys.modules["scripts.utilities"]
        utilities.load_sam3 = Mock()
        utilities.unload_sam3 = Mock()
        utilities.offload_vision_model = Mock()
        sam = types.ModuleType("scripts.utilities.sam3_utils")
        sam.Sam3Extract = object
        sam.build_extracts_metadata = Mock()
        sam.named_extracts_for_images = Mock()
        sys.modules["scripts.utilities.sam3_utils"] = sam
        sys.modules["scripts.utilities.image_utils"].ensure_dir = lambda path: path.mkdir(parents=True, exist_ok=True)
        save_stage = sys.modules["scripts.utilities.save_stage_utils"]
        save_stage.save_stage = Mock()
        save_stage._safe_filename = lambda name: name
        self.ollama.chat.side_effect = [{"message": {"content": '{"items":[]}'}}] * 4
        extract = load_module("test_extract_runner", "scripts/executers/global/run_extract_objects.py")
        profile = default_profiles()["object_detect"]
        profile["positive"] = "Find only blue objects."
        updates = []
        with stage_progress(updates.append):
            self.assertEqual(extract.run_extract_objects(self.image, prompt_profile=profile, vision_device="cpu", presence_penalty=0.4), ([], [], []))
        self.assertEqual((updates[-1]["substeps"][0]["completed"], updates[-1]["substeps"][0]["total"]), (3, 3))
        self.assertEqual(updates[-1]["substeps"][1]["total"], 0)
        self.assertEqual(self.ollama.chat.call_count, 3)
        for call in self.ollama.chat.call_args_list:
            self.assertIn("Find only blue objects.", call.kwargs["messages"][1]["content"])
            self.assertEqual(call.kwargs["options"]["num_gpu"], 0)
            self.assertEqual(call.kwargs["options"]["presence_penalty"], 0.4)
        granular = load_module("test_granular_runner", "scripts/executers/per_Item/run_analyze_segments.py")
        updates = []
        with stage_progress(updates.append):
            granular.execute(self.folder, prompt_profile=default_profiles()["analyze_segments"], vision_device="cpu", presence_penalty=0.6)
        self.assertEqual([step["total"] for step in updates[-1]["substeps"]], [1, 1, 0, 1])
        self.assertEqual(updates[-1]["substeps"][1]["skipped"], 1)
        self.assertEqual(self.ollama.chat.call_count, 4)
        self.assertEqual(self.ollama.chat.call_args.kwargs["options"]["num_gpu"], 0)
        self.assertEqual(self.ollama.chat.call_args.kwargs["options"]["presence_penalty"], 0.6)
        utilities.load_sam3.assert_not_called()

    def test_user_labels_remain_values_under_application_generated_ids(self):
        results = self.load_runners()
        profile = ranking_profile()
        profile["criteria"][0]["label"] = "constructor"
        results.save_rankings(self.folder, profile, [{"name": "image", **rating_response()}], scope="full_image")
        document = json.loads((self.folder / "full_image_rankings.v2.json").read_text())
        self.assertEqual(document["criteria"][0]["id"], "criterion_1")
        self.assertEqual(document["criteria"][0]["label"], "constructor")
        self.assertNotIn("constructor", document["items"][0]["ratings"])

    def test_canonical_writer_preserves_existing_user_artifacts(self):
        results = self.load_runners()
        existing = self.folder / "full_image_rankings.json"
        original = b'{"user_saved_result": true}'
        existing.write_bytes(original)
        results.save_rankings(self.folder, ranking_profile(), [{"name": "image", **rating_response()}], scope="full_image")
        self.assertEqual(existing.read_bytes(), original)
        self.assertTrue((self.folder / "full_image_rankings.v2.json").is_file())


if __name__ == "__main__":
    unittest.main()
