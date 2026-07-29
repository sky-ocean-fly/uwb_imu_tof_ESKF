#!/usr/bin/env python3

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from uwb_bias_learning.models import (  # noqa: E402
    POSE12_FEATURES,
    FeatureNormalizer,
    build_mlp,
    build_normalized_inference_module,
    checkpoint_payload,
    load_mlp_checkpoint,
    predict_mlp,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this environment")
class InferenceConsistencyTest(unittest.TestCase):
    def test_eager_state_torchscript_and_cli_match(self):
        import torch

        torch.manual_seed(17)
        identity = np.eye(3).reshape(-1).tolist()
        train = np.asarray(
            [
                [-1.0, -0.5, 0.0] + identity,
                [0.0, 0.0, 0.5] + identity,
                [1.0, 0.5, 1.0] + identity,
            ],
            dtype=np.float64,
        )
        sample = np.asarray([[0.25, 0.1, 0.75] + identity], dtype=np.float64)
        normalizer = FeatureNormalizer.fit(train)
        model = build_mlp([8, 4], "relu").eval()
        original_prediction = predict_mlp(model, sample, normalizer)

        with tempfile.TemporaryDirectory(prefix="uwb_bias_synthetic_") as directory:
            root = Path(directory)
            checkpoint_path = root / "model_state.pt"
            torch.save(
                checkpoint_payload(model, [8, 4], "relu", seed=17),
                str(checkpoint_path),
            )
            restored, _ = load_mlp_checkpoint(checkpoint_path)
            restored_prediction = predict_mlp(restored, sample, normalizer)

            wrapper = build_normalized_inference_module(restored, normalizer).eval()
            traced = torch.jit.trace(wrapper, torch.zeros((1, 12), dtype=torch.float32))
            torchscript_path = root / "model_torchscript.pt"
            traced.save(str(torchscript_path))
            loaded_script = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
            with torch.no_grad():
                torchscript_prediction = (
                    loaded_script(torch.from_numpy(sample.astype(np.float32)))
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

            per_feature = {
                name: {
                    "min": float(normalizer.train_min[index]),
                    "max": float(normalizer.train_max[index]),
                }
                for index, name in enumerate(POSE12_FEATURES)
            }
            spec = {
                "input": {"fields_in_exact_order": list(POSE12_FEATURES)},
                "training_domain": {"per_feature": per_feature},
            }
            (root / "model_spec.json").write_text(
                json.dumps(spec), encoding="utf-8"
            )
            command = [
                sys.executable,
                str(PACKAGE_ROOT / "scripts" / "predict_bias.py"),
                "--model", str(torchscript_path),
                "--tx", str(sample[0, 0]), "--ty", str(sample[0, 1]),
                "--tz", str(sample[0, 2]), "--rotation-matrix",
                *[str(value) for value in sample[0, 3:]], "--json",
            ]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(PACKAGE_ROOT / "src")
            completed = subprocess.run(
                command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, env=environment,
            )
            cli_prediction = float(json.loads(completed.stdout)["predicted_bias_m"])

        np.testing.assert_allclose(restored_prediction, original_prediction, rtol=0, atol=1e-7)
        np.testing.assert_allclose(torchscript_prediction, original_prediction, rtol=0, atol=1e-7)
        self.assertAlmostEqual(cli_prediction, float(original_prediction[0]), places=7)


if __name__ == "__main__":
    unittest.main()

