import numpy as np
import pytest
from src.analysis.master_session1 import paired_metric_bootstrap, seed_average, course_jobs, METRICS
import pandas as pd


def test_joint_bootstrap_known_two_user_distribution_and_shared_metric_pairs():
    # N=2: bootstrap means are 0, .5, 1, with probabilities .25, .5, .25.
    left = np.asarray([[0., 0.], [1., 2.]])
    point, low, high = paired_metric_bootstrap(left, np.zeros_like(left), 10000, 42)
    np.testing.assert_array_equal(point, [.5, 1.])
    np.testing.assert_array_equal(low, [0., 0.])
    np.testing.assert_array_equal(high, [1., 2.])
    again = paired_metric_bootstrap(left, np.zeros_like(left), 10000, 42)
    for actual, expected in zip(again, (point, low, high)):
        np.testing.assert_array_equal(actual, expected)


def test_bootstrap_preserves_user_pairing_zero_delta():
    left = np.arange(24).reshape(8, 3)
    for result in paired_metric_bootstrap(left, left, 100, 7):
        np.testing.assert_array_equal(result, [0., 0., 0.])
    with pytest.raises(ValueError, match="Paired"):
        paired_metric_bootstrap(left, left[:2])


def test_seed_averaging_aligns_users_and_does_not_fake_baseline_seeds():
    frames = {}
    for seed, value in ((7, .1), (42, .2), (2026, .3)):
        frames[("sasrec_ce", seed)] = pd.DataFrame({"user_id": [2, 1], **{m: [value, value+1] for m in METRICS}})
    frames[("mostpopular", 7)] = frames[("sasrec_ce", 7)]
    np.testing.assert_allclose(seed_average(frames, "sasrec_ce")[:, 0], [1.2, .2])
    np.testing.assert_allclose(seed_average(frames, "mostpopular")[:, 0], [1.1, .1])


def test_course_scope_is_resource_defined_and_reuse_is_not_success():
    jobs = []
    for i, (exp, model, seed, ratio) in enumerate([("E1", "sasrec_ce", 7, 1.), ("E2", "attention_ablation", 2026, 1.),
                                                ("E3", "sasrec_ce", 7, .4), ("E3", "sasrec_ce", 7, 1.),
                                                ("E3", "sasrec_ce", 42, .4), ("E3", "bert4rec", 7, .7),
                                                ("E4", "fenrec", 7, .4), ("E5", "bert4rec", 7, 1.)]):
        jobs.append({"job": i, "experiment": exp, "model": model, "seed": seed, "retention": ratio,
                     "path": f"results/core/{exp}/beauty/{model}/hash/{seed}"})
    result = course_jobs(jobs, {jobs[0]["path"]})
    assert [r["archive_job_id"] for r in result] == [0, 1, 2, 3]
    assert result[-1]["evidence_state"] == "REUSE_AVAILABLE"
    assert result[-1]["reuse_source"] == jobs[0]["path"]
    assert result[2]["evidence_state"] == "PENDING"
