"""Regression cases for localization, small-region preservation, and nested evidence."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from scripts.orchestration.progress import stage_progress, start_step

ROOT = Path(__file__).resolve().parents[1]


def module_from_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ExtractionAccuracyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        utilities = types.ModuleType("scripts.utilities")
        utilities.__path__ = [str(ROOT / "scripts" / "utilities")]
        extensions = types.ModuleType("scripts.extensions")
        extensions.__path__ = [str(ROOT / "scripts" / "extensions")]
        patcher = patch.dict(sys.modules, {"scripts.utilities": utilities, "scripts.extensions": extensions})
        patcher.start()
        self.addCleanup(patcher.stop)
        module_from_file("scripts.utilities.image_utils", "scripts/utilities/image_utils.py")
        self.sam = module_from_file("scripts.utilities.sam3_utils", "scripts/utilities/sam3_utils.py")
        self.extension = module_from_file("scripts.extensions.extractor_extensions", "scripts/extensions/extractor_extensions.py")
        self.items = module_from_file("scripts.utilities.analysis_items", "scripts/utilities/analysis_items.py")
        self.utilities = utilities
        self.utilities.extract_by_prompt = Mock()

    def image(self, width=80, height=60):
        path = self.folder / "source.png"
        Image.new("RGB", (width, height), "white").save(path)
        return path

    def extract(self, box, size=(80, 60), *, score=0.8, foreground_box=None):
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        left, top, right, bottom = foreground_box or box
        mask[top:bottom, left:right] = 1
        crop = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (255, 0, 0, 255))
        return self.sam.Sam3Extract(crop, box, score, mask)

    def test_small_box_uses_pixel_minimum_after_normalization(self):
        image = self.image(6000, 50)
        found = self.extract((600, 5, 612, 15), (6000, 50))
        self.utilities.extract_by_prompt.return_value = [found]
        rows = self.extension.run_sam_calls_with_fallback(image, [("objects", "small mark", (100, 100, 102, 300))], object())
        self.assertEqual(rows, [("objects", "small mark", found)])
        self.assertEqual(self.utilities.extract_by_prompt.call_args.kwargs["box"], (600, 5, 612, 15))

    def test_requested_score_threshold_reaches_sam_before_postprocessing(self):
        image = self.image()
        found = self.extract((8, 6, 40, 30), score=0.2)
        self.utilities.extract_by_prompt.return_value = [found]
        rows = self.extension.run_sam_calls_with_fallback(image, [("objects", "mark", (100, 100, 500, 500))], object(), min_sam_score=0.1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.utilities.extract_by_prompt.call_args.kwargs["threshold"], 0.1)

    def test_text_fallback_keeps_requested_instance_and_excludes_container_or_remote_matches(self):
        image = self.image()
        near = self.extract((8, 6, 20, 18), score=0.2)
        remote = self.extract((55, 35, 75, 55), score=0.99)
        container = self.extract((0, 0, 80, 60), score=0.99)
        self.utilities.extract_by_prompt.side_effect = [[], [remote, container, near]]
        rows = self.extension.run_sam_calls_with_fallback(image, [("objects", "mark", (100, 100, 300, 350))], object(), min_sam_score=0.1)
        self.assertEqual(rows, [("objects", "mark", near)])
        self.assertEqual(self.utilities.extract_by_prompt.call_args.kwargs["threshold"], 0.1)

    def test_duplicate_masks_merge_but_nearby_small_parts_survive(self):
        first = self.extract((1, 1, 4, 5))
        second = self.extract((8, 1, 11, 5))
        duplicate = self.extract((1, 1, 4, 5))
        rows = [("objects", "left mark", first), ("objects", "right mark", second), ("texts", "alias", duplicate)]
        self.assertEqual(self.sam.deduplicate_results(rows, tolerance=20), rows[:2])

    def test_distinct_masks_with_same_box_are_not_deduplicated(self):
        first = self.extract((0, 0, 20, 20), foreground_box=(2, 2, 8, 8))
        second = self.extract((0, 0, 20, 20), foreground_box=(12, 12, 18, 18))
        rows = [("objects", "mark", first), ("texts", "letters", second)]
        self.assertEqual(self.sam.deduplicate_results(rows, tolerance=20), rows)

    def test_recorded_bottle_and_table_aliases_keep_the_strongest_mask(self):
        fixture = json.loads((ROOT / "tests/fixtures/duplicate_segment_masks.json").read_text(encoding="utf-8"))
        rows = []
        for entry in fixture["regions"]:
            left, top, right, bottom = entry["bbox"]
            values = (np.arange(len(entry["mask_runs"])) % 2 == (0 if entry["first_pixel"] else 1))
            crop = np.repeat(values, entry["mask_runs"]).reshape(bottom - top, right - left)
            extract = self.extract(tuple(entry["bbox"]), tuple(fixture["image_size"]), score=entry["sam_score"])
            extract.mask_original[:] = 0
            extract.mask_original[top:bottom, left:right] = crop
            rows.append((entry["category"], entry["label"], extract))
        unique = self.sam.deduplicate_results(rows, tolerance=20)
        self.assertEqual([label for _, label, _ in unique], ["glass bottle with water", "white table"])
        metadata = self.sam.build_extracts_metadata(unique)
        self.assertEqual([len(item["duplicate_detections"]) for item in metadata["extracts"]], [1, 2])
        self.assertEqual(self.sam.deduplicate_results(unique), unique)

    def test_mask_identity_ignores_box_padding_and_keeps_best_score_with_zero_tolerance(self):
        first = self.extract((0, 0, 55, 55), foreground_box=(25, 20, 40, 40), score=0.5)
        best = self.extract((24, 19, 41, 41), foreground_box=(25, 20, 40, 40), score=0.99)
        separate = self.extract((60, 20, 75, 40), score=0.8)
        rows = [("objects", "mark", first), ("objects", "mark", separate), ("texts", "other label", best)]
        unique = self.sam.deduplicate_results(rows, tolerance=0)
        self.assertEqual(unique, rows[1:])
        self.assertEqual(best.duplicate_detections[0]["label"], "mark")

    def test_overlapping_boxes_and_nested_parts_are_not_duplicate_instances(self):
        container = self.extract((0, 0, 55, 55), foreground_box=(0, 0, 50, 50))
        part = self.extract((15, 15, 40, 40))
        thin = self.extract((20, 20, 21, 35))
        rows = [("objects", "target", item) for item in (container, part, thin)]
        self.assertEqual(self.sam.deduplicate_results(rows, tolerance=100), rows)

    def test_discarded_overlap_cannot_bridge_separate_retained_masks(self):
        rows = [("objects", "target", self.extract(box, (140, 40), score=score)) for box, score in (
            ((0, 0, 100, 40), 0.9), ((6, 0, 106, 40), 0.5), ((12, 0, 112, 40), 0.8))]
        self.assertEqual(self.sam.deduplicate_results(rows), [rows[0], rows[2]])

    def test_one_pixel_masks_and_configured_mask_threshold(self):
        first = self.extract((0, 0, 40, 1), (40, 1))
        second = self.extract((0, 0, 40, 1), (40, 1))
        for extract in (first, second):
            extract.mask_original = extract.mask_original.astype(float) * 0.3
        rows = [("objects", "line", first), ("objects", "edge", second)]
        self.assertEqual(self.sam.deduplicate_results(rows, mask_threshold=0.2), rows[:1])

    def test_shared_sam_loop_removes_aliases_before_export(self):
        source = self.image()
        first = self.extract((0, 0, 55, 55), foreground_box=(25, 20, 40, 40), score=0.7)
        second = self.extract((24, 19, 41, 41), foreground_box=(25, 20, 40, 40), score=0.9)
        self.utilities.extract_by_prompt.side_effect = [[first], [second]]
        updates = []
        with stage_progress(updates.append):
            start_step(1, total=2, unit="regions")
            rows = self.extension.run_sam_calls_with_fallback(source, [
                ("objects", "object", (0, 0, 700, 950)), ("objects", "alias", (300, 300, 550, 700))], object())
        self.assertEqual((updates[-1]["substeps"][0]["completed"], updates[-1]["substeps"][0]["total"]), (2, 2))
        self.assertEqual(rows, [("objects", "alias", second)])

    def test_boxed_sam_output_discards_empty_crop_and_clips_reconstruction_mask(self):
        image = self.image()
        outside = np.zeros((60, 80), dtype=np.uint8)
        outside[35:55, 55:75] = 1
        containing = np.ones((60, 80), dtype=np.uint8)
        processor = Mock()

        class Inputs(dict):
            def to(self, _):
                return self

        processor.return_value = Inputs(original_sizes=[[60, 80]])
        processor.post_process_instance_segmentation.return_value = [{
            "masks": np.stack([outside, containing]), "boxes": np.array([[55, 35, 75, 55], [0, 0, 80, 60]]),
            "scores": np.array([0.9, 0.6]),
        }]
        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = type("UnusedTensor", (), {})
        fake_torch.no_grad = contextlib.nullcontext
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.Sam3Model = fake_transformers.Sam3Processor = object
        loaded = self.sam.Sam3Loaded(Mock(), processor, "cpu")
        with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers}):
            result = self.sam.extract_by_prompt("mark", image, box=(8, 6, 24, 21), model_and_processor=loaded, threshold=0.1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].bbox, (8, 6, 24, 21))
        self.assertEqual(result[0].image.size, (16, 15))
        expected = np.zeros((60, 80), dtype=np.uint8)
        expected[6:21, 8:24] = 1
        np.testing.assert_array_equal(result[0].mask_original, expected)
        self.assertEqual(processor.post_process_instance_segmentation.call_args.kwargs["threshold"], 0.1)

    def test_parent_and_child_enumeration_preserves_identity_and_global_coordinates(self):
        (self.folder / "segments").mkdir()
        Image.new("RGB", (40, 35)).save(self.folder / "segments" / "shirt_0.png")
        metadata = {"extracts": [{"name": "shirt_0", "label": "shirt", "group": "shirt", "bbox": [10, 20, 50, 55]}]}
        child_folder = self.items.granular_directory(self.folder, "shirt_0", "shirt")
        child_folder.mkdir(parents=True)
        Image.new("RGB", (10, 8)).save(child_folder / "graphic_0.png")
        (child_folder / "segment_metadata.json").write_text(json.dumps({
            "parent_name": "shirt_0", "parent_bbox": [10, 20, 50, 55],
            "extracts": [{"name": "graphic_0", "label": "graphic", "group": "graphic", "bbox": [3, 4, 13, 12]}],
        }), encoding="utf-8")
        items = self.items.enumerate_analysis_items(self.folder, metadata)
        self.assertEqual(len(items), 2)
        child = items[1]
        self.assertEqual(child["name"], "shirt_0::graphic_0")
        self.assertEqual(child["bbox"], [13, 24, 23, 32])
        self.assertEqual(child["parent_name"], "shirt_0")
        self.assertEqual(child["source"], "part")
        self.assertEqual(child["image_path"], "granular_segments/shirt/shirt_0/graphic_0.png")
        self.assertNotIn("parent_name", items[0])

    def load_granular_runner(self):
        extensions = sys.modules["scripts.extensions"]
        for name in ("run_sam_calls_with_fallback", "vision_calls_for_group", "save_vision_calls_debug_json"):
            setattr(extensions, name, getattr(self.extension, name))
        self.utilities.load_sam3 = Mock()
        self.utilities.unload_sam3 = Mock()
        self.utilities.offload_vision_model = Mock()
        return module_from_file("accuracy_granular_runner", "scripts/executers/per_Item/run_analyze_segments.py")

    def test_granular_exports_deduplicate_across_parents_and_scene_in_global_coordinates(self):
        runner = self.load_granular_runner()
        segments = self.folder / "segments"
        segments.mkdir()
        parents = [
            {"name": "a_0", "label": "container a", "group": "objects", "bbox": [10, 10, 70, 60]},
            {"name": "b_0", "label": "container b", "group": "objects", "bbox": [20, 0, 80, 50]},
            {"name": "badge_0", "label": "badge", "group": "objects", "bbox": [30, 20, 40, 30]},
        ]
        (self.folder / "metadata.json").write_text(json.dumps({"extracts": parents}), encoding="utf-8")
        for parent in parents[:2]:
            Image.new("RGB", (60, 50)).save(segments / f"{parent['name']}.png")
        Image.new("RGBA", (10, 10), (255, 0, 0, 255)).save(segments / "badge_0.png")
        first = [("objects", label, self.extract(box, (60, 50))) for label, box in (
            ("container a", (0, 0, 60, 50)), ("badge alias", (20, 10, 30, 20)),
            ("shared mark", (35, 20, 45, 30)), ("container a", (3, 3, 5, 5)))]
        second = [("objects", label, self.extract(box, (60, 50))) for label, box in (
            ("different label", (25, 30, 35, 40)), ("shared mark", (40, 30, 50, 40)))]
        instructions = [(parent["name"], "objects", segments / f"{parent['name']}.png", []) for parent in parents[:2]]
        with patch.object(runner, "_run_sam_extraction", side_effect=[first, second]):
            runner.stage2_run_sam(instructions, self.folder)
        outputs = [json.loads((self.items.granular_directory(self.folder, parent["name"], "objects") /
                              "segment_metadata.json").read_text(encoding="utf-8")) for parent in parents[:2]]
        self.assertEqual([item["name"] for item in outputs[0]["extracts"]], ["shared mark_0", "container a_1"])
        self.assertEqual([item["name"] for item in outputs[1]["extracts"]], ["shared mark_0"])
        self.assertEqual([item["duplicate_of"] for item in outputs[0]["duplicate_detections"]], ["a_0", "badge_0"])
        self.assertEqual(outputs[1]["duplicate_detections"][0]["duplicate_of"], "a_0::shared mark_0")
        self.assertEqual(outputs[1]["extracts"][0]["bbox_global"], [60, 30, 70, 40])
        self.assertEqual(len(self.items.enumerate_analysis_items(self.folder)), 6)
        self.assertEqual(len(list((self.folder / "granular_segments").rglob("*.webp"))), 3)
        for output in outputs:
            for item in output["extracts"]:
                self.assertTrue((self.folder / item["image_path"]).is_file())

    def test_granular_source_uses_lossless_original_pixels(self):
        runner = self.load_granular_runner()
        source = self.image()
        with Image.open(source) as original:
            original.putpixel((15, 25), (11, 31, 71))
            original.save(source)
        path = runner._original_crop(self.folder, {"name": "shirt_0", "bbox": [10, 20, 50, 55]}, "shirt", source)
        with Image.open(path) as crop:
            self.assertEqual(crop.size, (40, 35))
            self.assertEqual(crop.getpixel((5, 5)), (11, 31, 71))

    def test_recorded_duplicate_ocr_responses_pass_the_granular_runner_without_retry(self):
        runner = self.load_granular_runner()
        image = self.image()
        ollama = types.ModuleType("ollama")
        ollama.chat = Mock()
        # Original and correction responses from the failed text_1 segment.
        for box in ([47, 20, 856, 230], [40, 25, 875, 236]):
            item = {"label": "STARSHIP 19", "box": box}
            ollama.chat.reset_mock()
            ollama.chat.side_effect = [{"message": {"content": json.dumps({"items": [item, item]})}}]
            with self.subTest(box=box), patch.dict(sys.modules, {"ollama": ollama}):
                calls = runner.stage1_vision_instructions(image, "texts", print_raw_vision_response=False)
            self.assertEqual(calls, [("texts", "STARSHIP 19", tuple(box))])
            self.assertEqual(ollama.chat.call_count, 1)

    def test_exported_paths_preserve_custom_group_and_label_independent_identity(self):
        runner = self.load_granular_runner()
        from scripts.utilities.save_stage_utils import save_stage
        original = self.image()
        parent = {"name": "region_01", "label": "Custom / display label", "group": "Symbols / Marks",
                  "bbox": [10, 10, 50, 45], "bbox_size": [40, 35]}
        folder = save_stage(images=[("region_01", Image.new("RGB", (40, 35)))],
                            metadata={"extracts": [parent]}, base_dir=self.folder / "run", source_image_path=original)
        child = self.extract((3, 4, 13, 12), (40, 35))
        with patch.object(runner, "stage1_vision_instructions", return_value=[("objects", "word/mark", (1, 1, 999, 999))]), \
             patch.object(runner, "_run_sam_extraction", return_value=[("objects", "word/mark", child)]):
            runner.execute(folder, print_raw_vision_response=False)
        saved = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["extracts"][0]
        self.assertEqual(saved["name"], "region_01")
        self.assertEqual(saved["label"], parent["label"])
        self.assertEqual(saved["group"], "Symbols / Marks")
        self.assertEqual(saved["image_path"], "segments/region_01.webp")
        self.assertEqual(saved["parts_metadata_path"], "granular_segments/Symbols _ Marks/region_01/segment_metadata.json")
        details = json.loads((folder / saved["parts_metadata_path"]).read_text(encoding="utf-8"))
        entry = details["extracts"][0]
        self.assertEqual(entry["label"], "word/mark")
        self.assertTrue((folder / entry["image_path"]).is_file())
        self.assertNotIn("\\", entry["image_path"])

    def test_granular_duplicate_merge_preserves_separate_nested_and_differently_labeled_targets(self):
        runner = self.load_granular_runner()
        image = self.image()
        first = {"label": "patch", "box": [100, 100, 400, 400]}
        separate = {"label": "patch", "box": [500, 100, 800, 400]}
        nested = {"label": "patch", "box": [200, 200, 250, 250]}
        other = {"label": "printed graphic", "box": first["box"]}
        ollama = types.ModuleType("ollama")
        ollama.chat = Mock(side_effect=[{"message": {"content": json.dumps({
            "items": [first, first, separate, nested, other],
        })}}])
        with patch.dict(sys.modules, {"ollama": ollama}):
            calls = runner.stage1_vision_instructions(image, "objects", print_raw_vision_response=False)
        self.assertEqual(calls, [("objects", item["label"], tuple(item["box"]))
                                 for item in (first, separate, nested, other)])
        self.assertEqual(ollama.chat.call_count, 1)

    def test_evidence_board_keeps_whole_scene_and_enlarges_the_unmarked_complete_crop(self):
        source = self.image(600, 300)
        with Image.open(source) as original:
            original.paste((255, 0, 0), (100, 100, 120, 120))
            original.paste((0, 0, 255), (120, 100, 140, 120))
            original.paste((0, 255, 0), (100, 85, 120, 95))
            original.putpixel((599, 299), (0, 255, 0))
            original.save(source)
        before = source.read_bytes()
        item = {"name": "unverified_detector_name", "bbox": [100, 100, 140, 120]}
        path = self.items.item_evidence_board(self.folder, item, source)
        self.assertEqual(source.read_bytes(), before)
        with Image.open(path) as board:
            self.assertEqual(board.format, "PNG")
            self.assertEqual(board.size, (1148, 356))
            self.assertEqual(board.getpixel((12 + 599, 44 + 299)), (0, 255, 0))
            self.assertEqual(board.getpixel((12 + 100, 44 + 100)), (0, 255, 255))
            # A 40x20 crop fills the 200px-wide target panel at a uniform 5x scale.
            self.assertEqual(board.getpixel((936, 144)), (255, 0, 0))
            self.assertEqual(board.getpixel((1135, 243)), (0, 0, 255))
            self.assertEqual(board.getpixel((936, 143)), (242, 244, 247))
            self.assertEqual(board.getpixel((1135, 244)), (242, 244, 247))
            # Nearby contains pixels above the tight box, expanded by 40px on every side.
            self.assertEqual(board.getpixel((740, 145)), (0, 255, 0))
            self.assertEqual(board.getpixel((724, 169)), (0, 255, 255))
            self.assertEqual(board.getpixel((624, 68)), (242, 244, 247))
            self.assertEqual(board.getpixel((624, 69)), (255, 255, 255))
            self.assertEqual(board.getpixel((923, 318)), (255, 255, 255))
            self.assertEqual(board.getpixel((923, 319)), (242, 244, 247))
            right_panel = np.asarray(board)[44:, 936:1136]
            self.assertFalse(np.any(np.all(right_panel == (0, 255, 255), axis=2)))
        saved_time = path.stat().st_mtime_ns
        self.assertEqual(self.items.item_evidence_board(self.folder, item, source), path)
        self.assertEqual(path.stat().st_mtime_ns, saved_time)

    def test_evidence_board_limits_scene_size_and_invalidates_changed_source_cache(self):
        source = self.image(2560, 1280)
        item = {"name": "../../outside", "bbox": [-10, -10, 2560, 1280]}
        path = self.items.item_evidence_board(self.folder, item, source)
        self.assertRegex(path.name, r"^board-[0-9a-f]{24}\.png$")
        self.assertTrue(path.resolve().is_relative_to(self.folder.resolve()))
        with Image.open(path) as board:
            self.assertEqual(board.size, (2200, 696))
            self.assertEqual(board.getpixel((12, 44)), (0, 255, 255))
            self.assertEqual(board.getpixel((1291, 683)), (0, 255, 255))
        Image.new("RGB", (2560, 1280), "red").save(source)
        changed = self.items.item_evidence_board(self.folder, item, source)
        self.assertNotEqual(changed, path)
        self.assertTrue(path.is_file())
        with Image.open(changed) as board:
            self.assertEqual(board.getpixel((500, 300)), (255, 0, 0))

    def test_evidence_board_requires_original_and_an_intersecting_box(self):
        source = self.image()
        item = {"name": "target", "bbox": [10, 10, 40, 40], "image_path": "source.png"}
        self.assertIsNone(self.items.item_evidence_board(self.folder, item, None))
        for box in (None, [0, 0, 0, 8], [90, 0, 110, 10], [0, 70, 10, 80], [0, 0, float("nan"), 10]):
            with self.subTest(box=box):
                self.assertIsNone(self.items.item_evidence_board(self.folder, {**item, "bbox": box}, source))
        self.assertFalse((self.folder / "analysis_evidence").exists())

    def test_evidence_board_nearby_view_clamps_to_source_and_preserves_its_aspect(self):
        source = self.image(600, 300)
        with Image.open(source) as original:
            original.paste((0, 255, 0), (65, 45, 80, 60))
            original.save(source)
        path = self.items.item_evidence_board(self.folder, {"name": "edge", "bbox": [0, 0, 40, 20]}, source)
        with Image.open(path) as board:
            # Clamping yields an 80x60 neighborhood, displayed at 300x225 (3.75x).
            self.assertEqual(board.getpixel((624, 81)), (0, 255, 255))
            self.assertEqual(board.getpixel((923, 305)), (0, 255, 0))
            self.assertEqual(board.getpixel((624, 80)), (242, 244, 247))
            self.assertEqual(board.getpixel((923, 306)), (242, 244, 247))

    def test_detail_grid_has_overlap_and_maps_back_to_full_image(self):
        views = module_from_file("accuracy_detection_views", "scripts/utilities/detection_views.py")
        boxes = views.detail_view_boxes(1024, 768)
        self.assertEqual(len(boxes), 4)
        self.assertGreater(boxes[0][2], boxes[1][0])
        self.assertGreaterEqual(min(boxes[0][2], boxes[0][3]), 256)
        self.assertEqual(views.detail_view_boxes(511, 511), [])
        self.assertEqual(views.remap_box((0, 0, 1000, 1000), (512, 384, 1024, 768), 1024, 768), (500, 500, 1000, 1000))

    def test_detail_merge_removes_repeated_region_and_preserves_nested_or_separate_instances(self):
        views = module_from_file("accuracy_merge_views", "scripts/utilities/detection_views.py")
        existing = [("objects", "Visible label", (100, 100, 500, 500))]
        incoming = [("objects", "visible-label", (105, 105, 495, 495)),
                    ("objects", "visible label", (200, 200, 250, 250)),
                    ("objects", "visible label", (600, 600, 800, 800))]
        result = views.merge_calls(existing, incoming, 20)
        self.assertEqual(result, [existing[0], incoming[1], incoming[2]])
        self.assertEqual(len(views.merge_calls(existing, incoming, 2)), 2)

    def test_detail_grid_eligibility_respects_long_and_short_side_boundaries(self):
        views = module_from_file("accuracy_boundary_views", "scripts/utilities/detection_views.py")
        for width, height in [(512, 255), (255, 512), (511, 511)]:
            with self.subTest(size=(width, height)):
                self.assertEqual(views.detail_view_boxes(width, height), [])
        for width, height in [(512, 256), (256, 512), (600, 338), (338, 600)]:
            with self.subTest(size=(width, height)):
                boxes = views.detail_view_boxes(width, height)
                self.assertEqual(len(boxes), 4)
                self.assertEqual(len(set(boxes)), 4)
                for left, top, right, bottom in boxes:
                    self.assertGreaterEqual(right - left, 128)
                    self.assertGreaterEqual(bottom - top, 128)
                    self.assertGreaterEqual(left, 0)
                    self.assertGreaterEqual(top, 0)
                    self.assertLessEqual(right, width)
                    self.assertLessEqual(bottom, height)
        self.assertEqual(views.detail_view_boxes(600, 338)[0], (0, 0, 325, 183))

    def test_same_label_moderate_overlap_merges_but_nearby_instances_survive(self):
        views = module_from_file("accuracy_overlap_views", "scripts/utilities/detection_views.py")
        first = ("objects", "cloth panel", (100, 100, 500, 500))
        repeated = ("objects", "Cloth-panel", (130, 130, 470, 470))
        nearby = ("objects", "cloth panel", (380, 100, 780, 500))
        nested = ("objects", "cloth panel", (200, 200, 250, 250))
        different_label = ("objects", "printed graphic", (130, 130, 470, 470))
        self.assertAlmostEqual(views.box_iou(first[2], repeated[2]), 0.7225)
        self.assertEqual(views.merge_calls([first], [repeated, nearby, nested, different_label], 20),
                         [first, nearby, nested, different_label])

    def test_granular_results_record_source_frame_and_global_child_box(self):
        runner = self.load_granular_runner()
        parent = {"name": "shirt_0", "bbox": [10, 20, 50, 55]}
        (self.folder / "metadata.json").write_text(json.dumps({"extracts": [parent]}), encoding="utf-8")
        folder = self.items.granular_directory(self.folder, "shirt_0", "shirt")
        folder.mkdir(parents=True)
        source = folder / "source.png"
        Image.new("RGB", (40, 35)).save(source)
        child = self.extract((3, 4, 13, 12), size=(40, 35))
        with patch.object(runner, "_run_sam_extraction", return_value=[("objects", "graphic", child)]):
            runner.stage2_run_sam([("shirt_0", "shirt", source, [])], self.folder)
        metadata = json.loads((folder / "segment_metadata.json").read_text())
        self.assertEqual(metadata["parent_bbox"], [10, 20, 50, 55])
        self.assertEqual(metadata["extracts"][0]["bbox_global"], [13, 24, 23, 32])
        self.assertEqual(metadata["coordinate_space"], "parent_crop")
        self.assertEqual(metadata["source_image"], "granular_segments/shirt/shirt_0/source.png")


if __name__ == "__main__":
    unittest.main()
