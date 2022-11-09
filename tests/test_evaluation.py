import os
import sys
from subprocess import check_call

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.metrics import auc, roc_curve

from picai_eval import evaluate
from picai_eval.data_utils import load_metrics
from picai_eval.eval import Metrics, evaluate_folder
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)

subject_list = [
    f"case-{i}"
    for i in range(10)
]
lesion_results_expected = [
    (1, 0.0, 0.),
    (0, 1.0, 0.),
    (1, 0.0, 0.), (0, 1.0, 0.0),
    (1, 1.0, 1/3),
    (1, 1.0, 9/41), (0, 2.0, 0),
    (1, 1.0, 1/3),
    (1, 1.0, 1/3), (1, 0.0, 0.0),
    (1, 1.0, 5/11), (1, 0.0, 0.0),
    (1, 0.0, 0.0), (0, 1.0, 0.0),
]
in_dir_det = "tests/test-maps/"
in_dir_annot = "tests/test-maps/"


@pytest.fixture(scope="session")
def y_det():
    """
    Read crafted detection maps
    """
    assert os.path.exists(in_dir_det), "Folder with softmax files not found!"
    y_det = [
        read_prediction(os.path.join(in_dir_det, f"{subject_id}_detection_map.nii.gz"))
        for subject_id in subject_list
    ]

    yield y_det


@pytest.fixture(scope="session")
def y_true():
    """
    Read crafted labels
    """
    assert os.path.exists(in_dir_annot), "Folder with annotations not found!"
    y_true = [
        read_label(os.path.join(in_dir_annot, f"{subject_id}_label.nii.gz"))
        for subject_id in subject_list
    ]

    yield y_true


def test_evaluation(y_det, y_true, num_parallel_calls=3):
    """
    Test standard evaluation pipeline
    The 10 crafted cases in subject_list should have:
    - TPs: 5
    - FPs: 4
    - FNs: 5
    """

    metrics = evaluate(
        y_det=y_det,
        y_true=y_true,
        subject_list=subject_list,
        num_parallel_calls=num_parallel_calls,
    )

    # check metrics
    assert metrics.lesion_TP[-2] == 5
    assert metrics.lesion_FP[-2] == 4
    assert metrics.num_lesions - metrics.lesion_TP[-2] == 5
    assert metrics.AP == (5/9)*(1/2) + 0*(1/2)

    # check lesion_results
    lesion_results_flat = [
        (is_lesion, confidence, overlap)
        for subject_id in subject_list
        for is_lesion, confidence, overlap in metrics.lesion_results[subject_id]
    ]
    assert_allclose(lesion_results_expected, lesion_results_flat)


def test_evaluation_DSC(y_det, y_true):
    """
    Test DSC overlap function
    """
    metrics = evaluate(
        y_det=y_det,
        y_true=y_true,
        subject_list=subject_list,
        overlap_func="DSC"
    )

    # check metrics
    assert metrics.lesion_TP[-2] == 5
    assert metrics.lesion_FP[-2] == 4
    assert metrics.num_lesions - metrics.lesion_TP[-2] == 5
    assert metrics.AP == (5/9)*(1/2) + 0*(1/2)


def test_evaluation_prediction_padding(y_det, y_true):
    """
    Test if functions will pad detection maps
    """

    y_true_padded = [
        resize_image_with_crop_or_pad(lbl, img_size=np.array(lbl.shape)+2)
        for lbl in y_true
    ]

    assert y_true_padded[0].shape != y_det[0].shape

    metrics = evaluate(
        y_det=y_det,
        y_true=y_true_padded,
        subject_list=subject_list
    )

    # check lesion_results
    lesion_results_flat = [
        (is_lesion, confidence, overlap)
        for subject_id in subject_list
        for is_lesion, confidence, overlap in metrics.lesion_results[subject_id]
    ]
    assert_allclose(lesion_results_expected, lesion_results_flat)


def test_sample_weights():
    """
    Test if functions apply sample weights correctly
    """

    lesion_results = {
        "0": [(0, 1, 1.)],
        "1": [(0, 2, 1.)],
        "2": [(1, 3, 1.)],
        "3": [(1, 1, 1.), (1, 1, 1.)],
    }

    lesion_weight = {
        "0": [1],
        "1": [1],
        "2": [1],
        "3": [4, 5],
    }

    # calculate metrics without weights
    metrics = Metrics(lesion_results=lesion_results)

    # calculate metrics with case/leison-level weights
    metrics_weighted = Metrics(
        lesion_results=lesion_results,
        case_weight=np.random.uniform(size=metrics.num_cases),
        lesion_weight=lesion_weight
    )

    # verify weights are applied
    assert metrics.auroc != metrics_weighted.auroc, "Sample weight not applied to AUROC!"
    assert metrics.AP != metrics_weighted.AP, f"Sample weight not applied to AP! ({metrics.AP} == {metrics_weighted.AP})"
    assert metrics_weighted.AP == 1*(1/10) + (10/12)*(9/10), f"Sample weight not applied correctly to AP! ({metrics.AP} != 0.85)"

    # check if sample weights are applied as expected
    # example adapted from: https://rdrr.io/cran/WeightedROC/f/inst/doc/Definition.pdf
    fpr, tpr, _ = roc_curve(
        y_true=(0, 0, 1, 1, 1),
        y_score=(1, 2, 3, 1, 1),
        sample_weight=(1, 1, 1, 4, 5),
    )
    auc_score = auc(fpr, tpr)
    assert auc_score == 0.325
    assert_allclose(tpr, [0., 0.1, 0.1, 1.])
    assert_allclose(fpr, [0., 0.,  0.5, 1.])


@pytest.mark.xfail
def test_evaluation_negative_predictions(y_det, y_true):
    """
    Test if evaluation works if (some) confidence scores are negative
    """
    # shift confidence scores to include both positive and negative confidences
    shifted_y_det = [y_det - 1.5 for y_det in y_det]

    if np.min(shifted_y_det) > 0 or np.max(shifted_y_det) < 0:
        # This test requires a sufficiently spread out prediction set, don't
        # trigger the expected fail
        return

    evaluate(
        y_det=shifted_y_det,
        y_true=y_true,
        subject_list=subject_list,
    )


@pytest.mark.xfail
def test_softmax_input(y_det, y_true, num_parallel_calls=3):
    """
    Test if evaluation throws an error when the input is a softmax volume (instead of detection maps)
    """
    evaluate(
        y_det=[np.random.normal(size=pred.shape) for pred in y_det],
        y_true=y_true,
        subject_list=subject_list,
        num_parallel_calls=num_parallel_calls,
    )


def test_evaluation_from_dir_with_subject_list(num_parallel_calls=3):
    detection_map_dir = "tests/test-maps"
    metrics = evaluate_folder(
        detection_map_dir,
        subject_list=subject_list,
        num_parallel_calls=num_parallel_calls,
        verbose=1,
    )

    # check metrics
    assert metrics.lesion_TP[-2] == 5
    assert metrics.lesion_FP[-2] == 4
    assert metrics.num_lesions - metrics.lesion_TP[-2] == 5
    assert metrics.AP == (5/9)*(1/2) + 0*(1/2)


def test_evaluation_from_dir_without_subject_list(num_parallel_calls=3):
    detection_map_dir = "tests/test-maps"
    metrics = evaluate_folder(detection_map_dir, num_parallel_calls=num_parallel_calls, verbose=1)

    # check metrics
    assert metrics.lesion_TP[-2] == 5
    assert metrics.lesion_FP[-2] == 4
    assert metrics.num_lesions - metrics.lesion_TP[-2] == 5
    assert metrics.AP == (5/9)*(1/2) + 0*(1/2)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="GitHub Actions' Windows does not like check_call")
def test_main(metrics_path="tests/output/metrics.json"):
    """Test usage from command line"""
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    check_call([
        "python", "-m", "picai_eval",
        "--input", os.path.join(os.getcwd(), "tests/test-maps"),
        "--output", os.path.join(os.getcwd(), metrics_path)
    ])

    metrics = load_metrics(metrics_path)
    metrics_expected = load_metrics("tests/output-expected/metrics-expected.json")

    assert metrics == metrics_expected


def test_metrics_save(y_det, y_true, metrics_path="tests/output/metrics.json"):
    """Test saving of metrics"""
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    metrics = evaluate(
        y_det=y_det,
        y_true=y_true,
        subject_list=subject_list,
    )

    metrics.save(metrics_path)
    metrics_reloaded = Metrics(metrics_path)
    metrics_expected = Metrics("tests/output-expected/metrics-expected.json")

    assert metrics == metrics_reloaded == metrics_expected


def test_minimal_metrics_save(y_det, y_true, metrics_path="tests/output/metrics-minimal.json"):
    """Test saving of metrics"""
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    metrics = evaluate(
        y_det=y_det,
        y_true=y_true,
        subject_list=subject_list,
    )

    metrics.save_minimal(metrics_path)
    metrics_reloaded = load_metrics(metrics_path)
    metrics_expected = load_metrics("tests/output-expected/metrics-minimal-expected.json")

    assert metrics_reloaded == metrics_expected


def test_full_metrics_save(y_det, y_true, metrics_path="tests/output/metrics-full.json"):
    """Test saving of metrics"""
    if os.path.exists(metrics_path):
        os.remove(metrics_path)

    metrics = evaluate(
        y_det=y_det,
        y_true=y_true,
        subject_list=subject_list,
    )

    metrics.save_full(metrics_path)
    metrics_reloaded = load_metrics(metrics_path)
    metrics_expected = load_metrics("tests/output-expected/metrics-full-expected.json")

    assert metrics_reloaded == metrics_expected


def test_select_subset_lesion_results():
    lesion_results = {
        "0": [(0, 1, 1.)],
        "1": [(0, 2, 1.)],
        "2": [(1, 3, 1.)],
        "3": [(1, 1, 1.), (1, 1, 1.)],
    }
    lesion_results_subset_expected = [
        (1, 3, 1.), (1, 1, 1.), (1, 1, 1.)
    ]

    # calculate metrics without weights
    metrics = Metrics(lesion_results=lesion_results)

    lesion_results_subset = metrics.get_lesion_results_flat(subject_list=["2", "3"])

    assert lesion_results_subset_expected == lesion_results_subset


def test_single_threaded(y_det, y_true):
    """Test if single threaded evaluation works"""
    test_evaluation(y_det=y_det, y_true=y_true, num_parallel_calls=1)
    test_evaluation_from_dir_with_subject_list(num_parallel_calls=1)
    test_evaluation_from_dir_without_subject_list(num_parallel_calls=1)
