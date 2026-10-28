"""Real loop progress, interruption and observer isolation without model execution."""
import threading
import unittest

from scripts.orchestration.progress import (
    fail_item, finish_step, progress_activity, progress_items, progress_steps,
    skip_item, stage_progress, start_step,
)


class StageProgressTests(unittest.TestCase):
    def test_batches_share_a_total_and_count_only_completed_work(self):
        updates = []
        with stage_progress(updates.append):
            progress_steps("Find parts", "Extract masks", "Save outputs")
            start_step(1, total=3)
            for batch in [["first", "too small"], ["last"]]:
                for item in progress_items(batch):
                    if item == "too small":
                        skip_item()
                        continue
                    progress_activity("Detail view 2 of 4")
                    self.assertEqual(updates[-1]["substeps"][0]["current"], item)
                    self.assertEqual(updates[-1]["substeps"][0]["completed"], 0 if item == "first" else 2)
            start_step(2, total=2, unit="regions")
            for item in progress_items(["mask one", "mask two"]):
                if item == "mask two":
                    fail_item()
            start_step(3)
            self.assertIsNone(updates[-1]["substeps"][2]["total"])
        first, second, third = updates[-1]["substeps"]
        self.assertEqual((first["completed"], first["total"], first["skipped"]), (3, 3, 1))
        self.assertEqual((second["completed"], second["total"], second["failed"]), (2, 2, 1))
        self.assertTrue(all(step["status"] == "succeeded" for step in (first, second, third)))
        self.assertEqual(updates[0]["current_substep"], None)  # Snapshots aren't mutated later.

    def test_error_keeps_the_current_item_unfinished_and_scope_is_cleared(self):
        updates = []
        with self.assertRaisesRegex(RuntimeError, "model stopped"):
            with stage_progress(updates.append):
                progress_steps("Rank items", "Save rankings")
                for item in progress_items(["done", "in flight", "pending"], step=1):
                    if item == "in flight":
                        raise RuntimeError("model stopped")
        step = updates[-1]["substeps"][0]
        self.assertEqual((step["status"], step["completed"], step["total"], step["current"]),
                         ("failed", 1, 3, "in flight"))
        count = len(updates)
        list(progress_items([1, 2, 3], step=99))  # A CLI run needs no observer.
        self.assertEqual(len(updates), count)

    def test_thread_and_nested_scope_do_not_overwrite_another_stage(self):
        outer, inner = [], []
        with stage_progress(outer.append):
            progress_steps("Outer")
            start_step(1, total=2)
            count = len(outer)
            thread = threading.Thread(target=lambda: list(progress_items(["other thread"], step=1)))
            thread.start()
            thread.join()
            with stage_progress(inner.append):
                list(progress_items(["nested"], step=1))
            self.assertEqual(len(outer), count)
            list(progress_items(["one", "two"]))
        self.assertEqual(outer[-1]["substeps"][0]["completed"], 2)
        self.assertEqual(inner[-1]["substeps"][0]["completed"], 1)

    def test_empty_batch_and_future_steps_do_not_invent_work(self):
        updates = []
        with stage_progress(updates.append):
            progress_steps("Find items", "Analyze items")
            list(progress_items([], step=1))
        first, second = updates[-1]["substeps"]
        self.assertEqual((first["completed"], first["total"], first["skipped"]), (0, 0, 0))
        self.assertEqual(second["status"], "skipped")

    def test_count_overflow_and_invalid_totals_are_rejected(self):
        with stage_progress(lambda _: None):
            with self.assertRaises(ValueError):
                start_step(1, total=-1)
            start_step(1, total=1)
            with self.assertRaises(ValueError):
                list(progress_items(["one", "two"]))
            finish_step()


if __name__ == "__main__":
    unittest.main()
