"""Execution contracts without model downloads or GPU work."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.orchestration.execution import RunPlan, StageCall, execute_plan, validate_plan
from scripts.orchestration.stages import STAGES


class PipelineExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "source.png"
        self.image.touch()
        self.folder = self.root / "outputs" / "source"

    def metadata(self, folder=None):
        folder = folder or self.folder
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "metadata.json").write_text(json.dumps({"extracts": []}), encoding="utf-8")
        return folder

    def test_all_stages_share_extraction_result_and_run_in_order(self):
        calls = []
        actual = self.root / "actual_run"

        def extract(**kwargs):
            calls.append(("extract_objects", kwargs))
            return self.metadata(actual)

        runners = {stage.key: (lambda key=stage.key, **kwargs: calls.append((key, kwargs))) for stage in STAGES}
        runners["extract_objects"] = extract
        progress = Mock()
        service = Mock()
        result = execute_plan(
            RunPlan(self.folder, tuple(StageCall(stage.key) for stage in STAGES), self.image),
            runners=runners, on_progress=progress, before_synthetic=service,
        )
        self.assertEqual(result, actual)
        self.assertEqual([key for key, _ in calls], [stage.key for stage in STAGES])
        self.assertEqual(calls[0][1], {"image_path": self.image, "base_dir": self.folder.parent})
        self.assertTrue(all(kwargs["run_folder"] == actual for _, kwargs in calls[1:]))
        self.assertEqual(progress.call_count, 2 * len(STAGES))
        service.assert_called_once_with()

    def test_existing_run_only_executes_selected_stages(self):
        self.metadata()
        depth, description = Mock(), Mock()
        execute_plan(RunPlan(self.folder, (StageCall("image_description"),)), runners={
            "depth_map": depth, "image_description": description,
        })
        depth.assert_not_called()
        description.assert_called_once_with(run_folder=self.folder)

    def test_missing_extraction_fails_before_any_backend_runs(self):
        runner = Mock()
        with self.assertRaisesRegex(ValueError, "Enable Extract Objects"):
            execute_plan(RunPlan(self.folder, (StageCall("depth_map"),), self.image), runners={"depth_map": runner})
        runner.assert_not_called()

    def test_failed_stage_stops_later_stages(self):
        self.metadata()
        later = Mock()
        with self.assertRaisesRegex(RuntimeError, "Depth Map failed: model unavailable"):
            execute_plan(RunPlan(self.folder, (StageCall("depth_map"), StageCall("image_description"))), runners={
                "depth_map": Mock(side_effect=ValueError("model unavailable")), "image_description": later,
            })
        later.assert_not_called()

    def test_extraction_must_produce_metadata(self):
        later = Mock()
        with self.assertRaisesRegex(RuntimeError, "Extract Objects failed"):
            execute_plan(RunPlan(self.folder, (StageCall("extract_objects"), StageCall("depth_map")), self.image), runners={
                "extract_objects": Mock(return_value=self.folder), "depth_map": later,
            })
        later.assert_not_called()

    def test_empty_out_of_order_and_existing_extraction_are_rejected(self):
        for plan in (
            RunPlan(self.folder, ()),
            RunPlan(self.folder, (StageCall("image_description"), StageCall("depth_map"))),
            RunPlan(self.folder, (StageCall("extract_objects"),)),
        ):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_service_failure_prevents_synthetic_inference(self):
        self.metadata()
        runner = Mock()
        with self.assertRaisesRegex(RuntimeError, "Synthetic Analysis failed"):
            execute_plan(
                RunPlan(self.folder, (StageCall("synthetic_analysis"),)),
                runners={"synthetic_analysis": runner},
                before_synthetic=Mock(side_effect=RuntimeError("service unavailable")),
            )
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()

