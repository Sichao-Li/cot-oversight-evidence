from __future__ import annotations

import numpy as np

from experiments.analyze_ablation_specificity import bootstrap_case_statistic
from experiments.analyze_baseline_repeat import first_divergence


def test_first_divergence_handles_token_mismatch_and_prefix() -> None:
    assert first_divergence([1, 2, 3], [1, 4, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert first_divergence([1, 2], [1, 2]) is None


def test_case_bootstrap_preserves_paired_statistic_point_estimate() -> None:
    target = np.asarray([1.0, 0.0, 1.0, 0.0])
    off_target = np.asarray([0.0, 0.0, 1.0, 0.0])

    def statistic(indices: np.ndarray) -> float:
        return float(np.mean(target[indices]) - np.mean(off_target[indices]))

    result = bootstrap_case_statistic(
        case_count=4,
        statistic=statistic,
        draws=100,
        seed=7,
    )
    assert result["point_estimate"] == 0.25
    assert result["bootstrap_draws"] == 100
