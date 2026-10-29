"""Model ownership and GPU handoffs, without downloading/loading model weights."""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
import types
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from scripts.runtime import gpu, ollama
from scripts.orchestration.execution import RunPlan, StageCall, execute_plan

ROOT = Path(__file__).resolve().parents[1]


class Resource:
    pass


class GpuLifetimeTests(unittest.TestCase):
    def test_success_releases_locals_and_cycles_before_allocator_cleanup(self):
        refs = []

        @gpu.release_after_call
        def inference():
            resource = Resource()
            resource.cycle = resource
            refs.append(weakref.ref(resource))
            return {"score": 0.8}

        def cleanup():
            gc.collect()
            self.assertIsNone(refs[0]())

        with patch.object(gpu, "release_cuda_memory", side_effect=cleanup):
            self.assertEqual(inference(), {"score": 0.8})

    def test_failed_inference_releases_chained_traceback_locals(self):
        refs = []

        def model_forward():
            tensor = Resource()
            refs.append(weakref.ref(tensor))
            raise ValueError("GPU OOM")

        @gpu.release_after_call
        def inference():
            try:
                model_forward()
            except ValueError as exc:
                raise RuntimeError("inference failed") from exc

        with patch.object(gpu, "release_cuda_memory"):
            try:
                inference()
            except RuntimeError as error:
                self.assertIsNone(refs[0]())
                formatted = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                self.assertIn("model_forward", formatted)
                self.assertIn("GPU OOM", formatted)
            else:
                self.fail("expected an error")

    def test_cancellation_releases_references(self):
        refs = []

        @gpu.release_after_call
        def inference():
            tensor = Resource()
            refs.append(weakref.ref(tensor))
            raise KeyboardInterrupt()

        with patch.object(gpu, "release_cuda_memory") as cleanup:
            with self.assertRaises(KeyboardInterrupt):
                inference()
            self.assertIsNone(refs[0]())
            cleanup.assert_called_once()

    def test_unused_cuda_is_not_initialized_by_cleanup(self):
        cuda = Mock()
        cuda.is_initialized.return_value = False
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            gpu.release_cuda_memory()
        cuda.empty_cache.assert_not_called()
        cuda.is_available.assert_not_called()
        cuda.device_count.assert_not_called()

    def cuda(self):
        cuda = Mock()
        cuda.is_initialized.return_value = True
        cuda.device_count.return_value = 3
        cuda.memory_reserved.side_effect = lambda device: [0, 512, 0][device]
        cuda.memory_allocated.return_value = 0
        return cuda

    def test_only_used_device_synchronized_and_context_never_reset(self):
        cuda = self.cuda()
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            gpu.release_cuda_memory()
        cuda.synchronize.assert_called_once_with(1)
        cuda.empty_cache.assert_called_once()
        cuda.ipc_collect.assert_called_once()
        cuda.cudart.assert_not_called()

    def test_cleanup_continues_after_synchronization_failure(self):
        cuda = self.cuda()
        cuda.synchronize.side_effect = RuntimeError("device error")
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(RuntimeError, "device error"):
                gpu.release_cuda_memory(strict=True)
        cuda.empty_cache.assert_called_once()
        cuda.ipc_collect.assert_called_once()

    def test_stage_guard_detects_live_tensor_allocations(self):
        cuda = self.cuda()
        cuda.memory_allocated.return_value = 1024
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(RuntimeError, "live tensors remain"):
                gpu.release_cuda_memory(strict=True, report=True)


class OllamaHandoffTests(unittest.TestCase):
    def client(self, names):
        resident = list(names)
        client = Mock()
        client.ps.side_effect = lambda: {"models": [{"model": name} for name in resident]}
        client.generate.side_effect = lambda **kwargs: resident.remove(kwargs["model"])
        return client, resident

    def test_all_resident_tags_unloaded_including_non_default_models(self):
        client, resident = self.client(["custom-vlm:v2", "embedding:latest"])
        with patch.object(ollama, "_client", return_value=client):
            ollama.offload_all_models()
        self.assertEqual(resident, [])
        self.assertEqual(client.generate.call_count, 2)
        self.assertTrue(all(c.kwargs["keep_alive"] == 0 for c in client.generate.call_args_list))

    def test_idle_cleanup_never_loads_an_absent_model(self):
        client, _ = self.client([])
        with patch.object(ollama, "_client", return_value=client):
            ollama.offload_model("not-installed:latest")
        client.generate.assert_not_called()

    def test_current_model_kept_but_other_model_released(self):
        client, resident = self.client(["wanted:latest", "stale:8b"])
        with patch.object(ollama, "_client", return_value=client):
            ollama.offload_all_models(except_model="wanted")
        self.assertEqual(resident, ["wanted:latest"])

    def test_unload_waits_for_residency_to_end(self):
        client = Mock()
        client.ps.side_effect = [
            {"models": [{"model": "model:v1"}]},
            {"models": [{"model": "model:v1"}]},
            {"models": []},
        ]
        with patch.object(ollama, "_client", return_value=client), patch.object(ollama.time, "sleep") as wait:
            ollama.offload_model("model:v1")
        wait.assert_called_once_with(0.1)

    def test_unload_failure_prevents_torch_handoff(self):
        client, _ = self.client(["stuck:latest"])
        client.generate.side_effect = RuntimeError("server rejected unload")
        with patch.object(ollama, "_client", return_value=client), patch.object(gpu, "release_cuda_memory") as local:
            with self.assertRaisesRegex(RuntimeError, "server rejected unload"):
                ollama.prepare_torch_model()
        local.assert_not_called()

    def test_server_absent_does_not_block_non_ollama_stages(self):
        client = Mock()
        client.ps.side_effect = ConnectionError("not running")
        with patch.object(ollama, "_client", return_value=client):
            ollama.offload_all_models()

    def test_resident_model_timeout_stops_handoff(self):
        client, _ = self.client(["stuck:latest"])
        client.generate.side_effect = None
        with patch.object(ollama, "_client", return_value=client), patch.object(ollama.time, "monotonic", side_effect=[0, 11]):
            with self.assertRaisesRegex(RuntimeError, "did not release"):
                ollama.offload_all_models()


class SamLifetimeTests(unittest.TestCase):
    def setUp(self):
        parent = types.ModuleType("scripts.utilities")
        parent.__path__ = [str(ROOT / "scripts/utilities")]
        image_utils = types.SimpleNamespace(ensure_rgb_hwc=Mock(), load_rgb_flat=Mock(), numpy_to_pil=Mock())
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_initialized=lambda: False))
        self.transformers = types.SimpleNamespace(Sam3Model=Mock(), Sam3Processor=Mock())
        spec = importlib.util.spec_from_file_location("scripts.utilities.sam3_utils", ROOT / "scripts/utilities/sam3_utils.py")
        self.sam = importlib.util.module_from_spec(spec)
        stubs = patch.dict(sys.modules, {"scripts.utilities": parent, "scripts.utilities.image_utils": image_utils,
                                       "scripts.utilities.sam3_utils": self.sam, "torch": fake_torch,
                                       "transformers": self.transformers})
        stubs.start()
        self.addCleanup(stubs.stop)
        spec.loader.exec_module(self.sam)
        self.sam.prepare_torch_model = Mock()

    def test_unload_drops_wrapper_and_local_references(self):
        model = Resource()
        model.to = lambda device: None
        reference = weakref.ref(model)
        loaded = self.sam.Sam3Loaded(model, Resource(), "cuda:0")
        self.sam._LIVE_SAM3[id(loaded)] = loaded
        del model
        self.assertEqual(self.sam.unload_all_sam3(), 1)
        self.assertIsNone(loaded.model)
        self.assertIsNone(loaded.processor)
        self.assertIsNone(reference())
        self.assertEqual(len(self.sam._LIVE_SAM3), 0)

    def test_partial_gpu_load_failure_does_not_retain_model(self):
        refs = []

        class Model:
            def to(self, device):
                raise RuntimeError("GPU allocation failed")

        def load(*args):
            model = Model()
            refs.append(weakref.ref(model))
            return model

        self.transformers.Sam3Model.from_pretrained.side_effect = load
        try:
            self.sam.load_sam3(device="cuda")
        except RuntimeError as error:
            self.assertIn("GPU allocation failed", str(error))
            self.assertIsNone(refs[0]())
        else:
            self.fail("expected load error")


@unittest.skipUnless(os.environ.get("CHECK_GPU_LIFECYCLE") == "1", "opt-in GPU allocation smoke test")
class CudaAllocationSmokeTests(unittest.TestCase):
    def test_success_failure_and_next_inference_return_allocations_to_baseline(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        gpu.release_cuda_memory()
        baseline = torch.cuda.memory_allocated()

        @gpu.inference_call
        def inference(fail=False):
            weights = torch.ones(16 * 1024 * 1024, device="cuda")
            activations = weights * 2
            if fail:
                raise RuntimeError("simulated inference failure")
            return activations[0].item()

        for fail in (False, True, False):
            if fail:
                try:
                    inference(True)
                except RuntimeError as error:
                    self.assertEqual(str(error), "simulated inference failure")
                    self.assertEqual(torch.cuda.memory_allocated(), baseline)
            else:
                self.assertEqual(inference(), 2)
            self.assertEqual(torch.cuda.memory_allocated(), baseline)
            self.assertEqual(torch.cuda.memory_reserved(), baseline)
        print(f"CUDA smoke test: peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB; "
              f"after success/failure/retry {torch.cuda.memory_allocated()} allocated, {torch.cuda.memory_reserved()} reserved bytes")


class StageBoundaryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        (self.folder / "metadata.json").write_text(json.dumps({"extracts": []}))
        self.plan = RunPlan(self.folder, (StageCall("depth_map"), StageCall("image_description")))

    def test_cleanup_runs_before_first_stage_and_between_stages(self):
        events = []
        execute_plan(self.plan, runners={
            "depth_map": lambda **kwargs: events.append("depth"),
            "image_description": lambda **kwargs: events.append("vision"),
        }, cleanup=lambda: events.append("cleanup"))
        self.assertEqual(events, ["cleanup", "depth", "cleanup", "vision", "cleanup"])

    def test_cleanup_failure_blocks_the_next_stage(self):
        later = Mock()
        with self.assertRaisesRegex(RuntimeError, "still resident"):
            execute_plan(self.plan, runners={"depth_map": Mock(), "image_description": later},
                         cleanup=Mock(side_effect=[None, RuntimeError("still resident")]))
        later.assert_not_called()

    def test_stage_failure_clears_tensor_locals_before_cleanup_and_keeps_original_error(self):
        refs = []

        def failed_stage(**kwargs):
            tensor = Resource()
            refs.append(weakref.ref(tensor))
            raise ValueError("bad inference")

        def cleanup():
            if refs:
                self.assertIsNone(refs[0]())
                raise RuntimeError("cleanup also failed")

        later = Mock()
        with self.assertRaisesRegex(RuntimeError, "Depth Map failed: bad inference"):
            execute_plan(self.plan, runners={"depth_map": failed_stage, "image_description": later}, cleanup=cleanup)
        later.assert_not_called()


class ForensicFallbackTests(unittest.TestCase):
    def test_gpu_model_and_activations_are_freed_before_cpu_retry(self):
        import numpy as np
        for name, class_name, preprocessing_name in (
            ("adaptive_cfa_net", "AdaptiveCFANet", "adaptive_cfa_net_preprocessing"),
            ("psccnet", "PSCCNet", "psccnet_preprocessing"),
        ):
            with self.subTest(method=name):
                resident = []
                checked = []

                class Tensor:
                    def __init__(self, device="cpu"):
                        self.device = device
                        if device == "cuda":
                            resident.append(weakref.ref(self))

                    def to(self, device):
                        return Tensor(device)

                    def cpu(self):
                        return self.to("cpu")

                class Method:
                    def __init__(self, **kwargs):
                        self.device = "cpu"

                    def to_device(self, device):
                        self.device = device
                        if device == "cuda":
                            resident.append(weakref.ref(self))

                    def benchmark(self, **kwargs):
                        if self.device == "cuda":
                            intermediate = Resource()
                            resident.append(weakref.ref(intermediate))
                            raise RuntimeError("Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same")
                        self_outer.assertTrue(all(ref() is None for ref in resident))
                        checked.append(True)
                        return {"detection": np.array(0.7)}

                self_outer = self
                torch_stub = types.SimpleNamespace(Tensor=Tensor, no_grad=nullcontext,
                                                  cuda=types.SimpleNamespace(is_initialized=lambda: False))
                method_stub = types.SimpleNamespace(**{class_name: Method, preprocessing_name: lambda **data: data})
                with patch.dict(sys.modules, {"torch": torch_stub, "cv2": types.SimpleNamespace(),
                                              f"photoholmes.methods.{name}": method_stub}), \
                        patch.object(sys, "_project_root", str(ROOT), create=True):
                    spec = importlib.util.spec_from_file_location("forensic_lifetime_test", ROOT / "scripts/extensions/forgery_detection_extensions.py")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    module.prepare_torch_model = Mock()
                    module.ensure_weights = lambda name: Path("cached-weights")
                    module._torch_load_weights_only_false = nullcontext
                    result = module.run_method(name, Tensor(), device="cuda")
                    self.assertAlmostEqual(float(result["detection"]), 0.7)
                    self.assertEqual(checked, [True])


class SyntheticLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.image = Path(self.temporary.name) / "image.png"
        Image.new("RGB", (8, 8)).save(self.image)
        fake_torch = types.SimpleNamespace(no_grad=nullcontext, cuda=types.SimpleNamespace(is_initialized=lambda: False))
        fake_model = types.SimpleNamespace(NonescapeClassifier=Mock(), preprocess_image=lambda image: Mock())
        modules = {"torch": fake_torch, "nonescape": fake_model,
                   "huggingface_hub": types.SimpleNamespace(hf_hub_download=Mock(side_effect=AssertionError("no downloads")))}
        self.stub = patch.dict(sys.modules, modules)
        self.stub.start()
        self.addCleanup(self.stub.stop)
        spec = importlib.util.spec_from_file_location("synthetic_lifetime_test", ROOT / "scripts/services/synthetic_api_server.py")
        self.service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.service)
        self.service._resolve_model_path = lambda model: "already-cached.safetensors"

    def test_service_releases_classifier_after_success_and_error(self):
        refs = []
        error = [False]

        class Classifier:
            def __call__(self, image):
                if error[0]:
                    raise ValueError("inference failed")
                class Probabilities:
                    def __getitem__(self, index):
                        return types.SimpleNamespace(item=lambda: [0.8, 0.2][index[1]])
                return Probabilities()

        def load(path):
            classifier = Classifier()
            refs.append(weakref.ref(classifier))
            return classifier

        self.service.NonescapeClassifier.from_pretrained.side_effect = load

        def cleanup():
            gc.collect()
            self.assertTrue(self.service._prediction_lock.locked())
            self.assertTrue(all(ref() is None for ref in refs))

        with patch.object(gpu, "release_cuda_memory", side_effect=cleanup):
            self.assertFalse(self.service._predict(self.image)["is_synthetic"])
            self.assertIsNone(self.service._active_model_path)
            error[0] = True
            with self.assertRaisesRegex(ValueError, "inference failed"):
                self.service._predict(self.image)
            self.assertIsNone(self.service._active_model_path)
        self.assertFalse(self.service._prediction_lock.locked())

    def test_requests_are_serialized_through_cleanup(self):
        active = []
        overlapping = []

        @gpu.release_after_call
        def inference(*args):
            if active:
                overlapping.append(True)
            active.append(True)
            time.sleep(0.01)
            return {"ok": True}

        def cleanup():
            self.assertTrue(self.service._prediction_lock.locked())
            time.sleep(0.01)
            active.clear()

        self.service._predict_once = inference
        with patch.object(gpu, "release_cuda_memory", side_effect=cleanup), ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: self.service._predict(self.image), range(3)))
        self.assertEqual(overlapping, [])
        self.assertTrue(all(result["ok"] for result in results))


if __name__ == "__main__":
    unittest.main()
