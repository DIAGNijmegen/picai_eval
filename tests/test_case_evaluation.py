import os

import pytest
from numpy.testing import assert_allclose

from picai_eval.eval import evaluate_case


@pytest.mark.parametrize("y_det,y_true,expected_y_list", [
    # Case 0: no ground truth lesions, no detection maps.
    ("case-0_detection_map.nii.gz", "case-0_label.nii.gz", []),

    # Case 1: single ground truth lesion, no detection maps. 1 FN.
    ("case-1_detection_map.nii.gz", "case-1_label.nii.gz", [(1, 0.0, 0.)]),

    # Case 2: no ground truth lesion, single ground truth lesion. 1 FP.
    ("case-2_detection_map.nii.gz", "case-2_label.nii.gz", [(0, 1.0, 0.)]),

    # Case 3: single ground truth lesion, single detection map. Candidate 1: IoU = 0. 1 FN + 1 FP.
    ("case-3_detection_map.nii.gz", "case-3_label.nii.gz", [(1, 0.0, 0.), (0, 1.0, 0.0)]),

    # Case 4: single ground truth lesion, single detection map. Candidate 1: IoU = 1/3. 1 TP.
    ("case-4_detection_map.nii.gz", "case-4_label.nii.gz", [(1, 1.0, 1/3)]),

    # Case 5: single ground truth lesion, two detection maps. Candidate 1: IoU = 9/41, confidence = 1. Candidate 2: IoU = 3/47, confidence = 2. 1 TP + 1 FP.
    ("case-5_detection_map.nii.gz", "case-5_label.nii.gz", [(1, 1.0, 9/41), (0, 2.0, 0)]),

    # Case 6: single ground truth lesion, two detection maps. Candidate 1: IoU = 1/3, confidence = 1. Candidate 2: IoU = 2/7, confidence = 2. 1 TP.
    ("case-6_detection_map.nii.gz", "case-6_label.nii.gz", [(1, 1.0, 1/3)]),

    # Case 7: two ground truth lesion, single detection map. Candidate 1: IoU = 1/3 with either ground truth lesion. 1 TP + 1 FN.
    ("case-7_detection_map.nii.gz", "case-7_label.nii.gz", [(1, 1.0, 1/3), (1, 0.0, 0.0)]),

    # Case 8: two ground truth lesion, single detection map. Candidate 1: IoU = 5/11 with one gt lesion, 1/11 with other gt lesion. 1 TP + 1 FN.
    ("case-8_detection_map.nii.gz", "case-8_label.nii.gz", [(1, 1.0, 5/11), (1, 0.0, 0.0)]),

    # Case 9: single ground truth lesion, one lesion detection map. Candidate 1: IoU = 1/25. 1 FN + 1 FP.
    ("case-9_detection_map.nii.gz", "case-9_label.nii.gz", [(1, 0.0, 0.0), (0, 1.0, 0.0)]),
])
def test_evaluate_case(y_det, y_true, expected_y_list):
    """
    Lesion candidate extraction and FROC evaluation Unit Test
    Evaluate predictions and calculate FROC statistics
    """

    y_list, *_ = evaluate_case(
        y_det=os.path.join("tests/test-maps", y_det),
        y_true=os.path.join("tests/test-maps", y_true),
    )

    assert_allclose(y_list, expected_y_list)
