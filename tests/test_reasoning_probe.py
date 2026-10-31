from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cot_oversight_case_studies.reasoning_probe import (
    cross_validated_probe_predictions,
    relative_layer_sweep,
    sample_confirmation_indices,
    step_presence_labels,
)


class ReasoningProbeTests(unittest.TestCase):
    def test_step_presence_rules_require_positive_non_negated_claims(self) -> None:
        positive = (
            "A command injection vulnerability exists. User-controlled name is passed to "
            "Runtime.exec without sanitization, so an attacker could trigger execution."
        )
        labels = step_presence_labels(positive)
        self.assertEqual(labels["identifies_vulnerability"], 1)
        self.assertEqual(labels["traces_data_flow"], 1)
        self.assertEqual(labels["assesses_exploitability"], 1)

        negative = "The code does not contain a security vulnerability and is not exploitable."
        self.assertEqual(
            step_presence_labels(negative),
            {
                "identifies_vulnerability": 0,
                "traces_data_flow": 0,
                "assesses_exploitability": 0,
            },
        )

    def test_relative_layer_sweep_maps_qwen_depth_to_llama_depth(self) -> None:
        mapped = relative_layer_sweep(
            [8, 12, 16, 20, 24, 28],
            source_depth=48,
            target_depth=80,
        )
        self.assertEqual(mapped, [13, 20, 27, 33, 40, 47])

    def test_confirmation_sample_matches_frozen_sampling_rule(self) -> None:
        rows = [
            {"record_id": f"row-{index:03d}", "is_correct": index % 3 != 0}
            for index in range(120)
        ]
        first = sample_confirmation_indices(rows, n=50, seed=442)
        second = sample_confirmation_indices(rows, n=50, seed=442)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertTrue(all(rows[index]["is_correct"] for index in first))

    def test_cross_validated_probe_returns_out_of_fold_probabilities(self) -> None:
        rng = np.random.default_rng(7)
        negative = rng.normal(loc=-1.0, scale=0.2, size=(20, 4))
        positive = rng.normal(loc=1.0, scale=0.2, size=(20, 4))
        metrics = cross_validated_probe_predictions(
            np.concatenate([negative, positive]),
            [0] * len(negative) + [1] * len(positive),
        )
        self.assertEqual(len(metrics["probabilities"]), 40)
        self.assertGreater(metrics["auc_mean"], 0.95)
        self.assertEqual(metrics["cv_folds"], 5)


if __name__ == "__main__":
    unittest.main()
