#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from sklearn.metrics import auc

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass


label_structure = np.ones((3, 3, 3))


# parse detection maps to individual lesions + confidences
def parse_detection_map(
    y_det: "npt.NDArray[np.float32]"
) -> "Tuple[Dict[int, float], npt.NDArray[np.int32]]":
    """Extract confidence scores per lesion candidate"""
    # label all non-connected components in the detection map
    blobs_index, num_blobs = ndimage.label(y_det, structure=label_structure)

    # input verification
    if num_blobs < len(set(np.unique(y_det))-{0}):
        raise ValueError(
            "It looks like you provided your predictions as softmax volumes instead of detection maps. "
            "If this is the case, please convert your softmax volumes to detection maps (e.g. via postprocessing). "
            "Visit https://github.com/DIAGNijmegen/picai_eval/ for documentation on what we expect of `detection maps` "
            "and how they can be generated from softmax predictions."
        )

    # extract confidence per lesion candidate
    confidences = {}
    for lesion_candidate_id in range(num_blobs):
        max_prob = y_det[blobs_index == (1+lesion_candidate_id)].max()
        confidences[lesion_candidate_id] = float(max_prob)

    return confidences, blobs_index


def calculate_dsc(y_det: "npt.NDArray[np.float32]", y_true: "npt.NDArray[np.int32]") -> float:
    """Calculate Dice similarity coefficient (DSC) for N-D Arrays"""
    epsilon = 1e-8
    dsc_num = np.sum(y_det[y_true == 1]) * 2.0
    dsc_denom = np.sum(y_det) + np.sum(y_true)
    return float((dsc_num + epsilon) / (dsc_denom + epsilon))


def calculate_iou(y_det: "npt.NDArray[np.float32]", y_true: "npt.NDArray[np.int32]") -> float:
    """Calculate Intersection over Union (IoU) for N-D Arrays"""
    epsilon = 1e-8
    iou_num = np.sum(y_det[y_true == 1])
    iou_denom = np.sum(y_det) + np.sum(y_true) - iou_num
    return float((iou_num + epsilon) / (iou_denom + epsilon))


def calculate_operating_points(y, x, op_match=None, verbose=1):
    """
    Calculate operating points for a curve.

    Input:
    - y: (monotonically increasing) performance metric, such as the True Positive Rate
    - x: (monotonically increasing) performance metric, such as the False Positive Rate
    - op_match: dictionary that specifies the target operating point: {
        'x': target x value, 'y': target y value
    }

    Returns:
    - dictionary with operating point(s): {
        'op_closest_xy_y': y_op, # y value at operating point that matches both x and y of target operating point
        'op_closest_xy_x': x_op, # x value at operating point that matches both x and y of target operating point
        ...
    }
    """
    # TODO: currently, a lower sensitivity is preferrred over a higher sensitivity if that means the operating point is matched better.
    # Would be better to go for the best sensitivity/specificity, if that can be done without hurting the other performance metric.
    # In practice, this should not be an issue, as we have many points then.
    y = np.array(y)
    x = np.array(x)
    operating_points = {}

    if not np.all(np.diff(y) >= 0) and verbose:
        print("Warning: y performance metric is not monotonically increasing, this could lead to unexpected behaviour!")
    if not np.all(np.diff(x) >= 0) and verbose:
        print("Warning: x performance metric is not monotonically increasing, this could lead to unexpected behaviour!")

    # grab index of intersection -> compute y/TPR and x/FPR @ index -> store
    op_best_roc_idx = np.argmin(np.abs(y - (1 - x)))
    op_best_roc_y = y[op_best_roc_idx]
    op_best_roc_x = x[op_best_roc_idx]
    operating_points.update(dict(
        op_best_roc_idx=op_best_roc_idx,
        op_best_roc_y=op_best_roc_y,
        op_best_roc_x=op_best_roc_x
    ))

    if op_match is not None:
        # calculate operating point closest to target operating point
        abs_deficit_x, abs_deficit_y = None, None
        optional_x_keys = ['x', 'fpr', 'FPR']
        optional_y_keys = ['y', 'tpr', 'TPR', 'sensitivity', 'sens']

        # if the target x value is specified, calculate the difference between target and oberved value
        for key in optional_x_keys:
            if key in op_match:
                op_match_x = op_match[key]
                abs_deficit_x = np.abs(x - op_match_x)
                break

        # if the target y value is specified, calculate the difference between target and oberved value
        for key in optional_y_keys:
            if key in op_match:
                op_match_y = op_match[key]
                abs_deficit_y = np.abs(y - op_match_y)
                break

        # if both target x and y values are specified, calculate the difference between the target pair and observed pair
        # at the best match, store the observed x and y values
        if abs_deficit_x is not None and abs_deficit_y is not None:
            # determine the index of the the closest point to the target pair
            abs_deficit = abs_deficit_x + abs_deficit_y
            op_closest_xy_idx = np.argmin(abs_deficit)
            op_closest_xy_y = y[op_closest_xy_idx]
            op_closest_xy_x = x[op_closest_xy_idx]

            # store
            operating_points.update(dict(
                op_closest_xy_idx=op_closest_xy_idx,
                op_closest_xy_y=op_closest_xy_y,
                op_closest_xy_x=op_closest_xy_x
            ))

        # same for matching x only
        if abs_deficit_x is not None:
            # determine the index of the the closest point to the target value
            op_closest_x_idx = np.argmin(abs_deficit_x)
            op_closest_x_y = y[op_closest_x_idx]
            op_closest_x_x = x[op_closest_x_idx]

            # store
            operating_points.update(dict(
                op_closest_x_idx=op_closest_x_idx,
                op_closest_x_y=op_closest_x_y,
                op_closest_x_x=op_closest_x_x
            ))

        # same for matching y only
        if abs_deficit_y is not None:
            # determine the index of the the closest point to the target value
            op_closest_y_idx = np.argmin(abs_deficit_y)
            op_closest_y_y = y[op_closest_y_idx]
            op_closest_y_x = x[op_closest_y_idx]

            # store
            operating_points.update(dict(
                op_closest_y_idx=op_closest_y_idx,
                op_closest_y_y=op_closest_y_y,
                op_closest_y_x=op_closest_y_x
            ))

    return operating_points


# calculate statistics for multiple curves
def calculate_statistics(metrics, op_match=None, x_start=0., x_end=1., verbose=1):
    """
    Calculate statistics, such as the area under the curve, for multiple (independent) curves.
    To calculate shared statistics, the curves must be translated to a shared x domain. To
    achieve this with virtually no loss of the step-like nature of curves like ROC and FROC,
    the shared x values are derived from the input, and offset with ± 1e-7.

    Input:
    - metrics should be a list of tuples with the y & x coordinates for each run:
    [([y1, y2, y3, ...], [x1, x2, x3]), # run 1
     ([y1, y2, y3, ...], [x1, x2, x3]), # run 2
     ]
    - op_match: {
        'y': value of y metric (e.g., TPR/sensitivity) to match,
        'x': value of x metric (e.g., FPR/false positive rate) to match,
    }

    Note: mean and 95% CI are calculated as function of the shared x.
    """
    # construct the array of shared x values
    eps = 1e-10
    x_shared = np.array([xi for _, x in metrics for xi in x], dtype=np.float64)  # collect list of all possible x-values
    x_shared = np.ravel(x_shared)  # flatten list, if necessary
    x_shared = np.append(x_shared, [x_start, x_end])  # add x_start and x_end to ensure correct pAUC calculation
    x_shared = np.concatenate((x_shared+eps, x_shared-eps))
    x_shared = np.unique(x_shared)  # only keep unique x values
    x_shared.sort()  # sort in ascending order (inplace)

    # validate x_start and x_end
    assert x_start < x_end, f"x_start must be smaller than x_end! Got x_start={x_start} and x_end={x_end}."

    # convert the per-model y (e.g., TPR) vs x (e.g., FPR) to a shared domain
    y_shared_all = np.zeros(shape=(len(metrics), len(x_shared)), dtype=np.float32)
    auroc_all = []
    individually_matched_operating_points = []
    for i, (y, x) in enumerate(metrics):
        # if necessary, unpack x and y
        if len(y) == 1:
            y = y[0]
        if len(x) == 1:
            x = x[0]

        # interpolate the y values to the shared x values
        y_shared_domain = np.interp(x_shared, x, y)
        y_shared_all[i] = y_shared_domain

        # calculate AUROC for macro stats
        mask = (x_shared >= x_start) & (x_shared <= x_end)
        auc_score = auc(x_shared[mask], y_shared_domain[mask])
        auroc_all += [auc_score]

        # match operating point for each run individually
        operating_points = calculate_operating_points(y=y, x=x, op_match=op_match)
        individually_matched_operating_points += [operating_points]

    # calculate statistics in shared domain
    y_shared_mean = np.mean(y_shared_all, axis=0)
    y_shared_std = np.std(y_shared_all, axis=0)
    y_shared_CI_lower = np.percentile(y_shared_all, 2.5, axis=0)
    y_shared_CI_higher = np.percentile(y_shared_all, 97.5, axis=0)
    auroc_mean = np.mean(auroc_all)
    auroc_std = np.std(auroc_all)

    # calculate operating points in shared domain
    operating_points = calculate_operating_points(y=y_shared_mean, x=x_shared,
                                                  op_match=op_match, verbose=verbose)

    # collect results
    results = {
        # overview statistics
        'auroc_mean': auroc_mean,
        'auroc_std': auroc_std,
        'auroc_all': auroc_all,

        # for plotting
        'x_shared': x_shared,
        'y_shared_all': y_shared_all,
        'y_shared_mean': y_shared_mean,
        'y_shared_std': y_shared_std,
        'y_shared_CI_lower': y_shared_CI_lower,
        'y_shared_CI_higher': y_shared_CI_higher,

        # individually matched operating point
        'individually_matched_operating_points': individually_matched_operating_points,
    }
    results.update(operating_points)

    # calculate standard deviation of each metric (op_closest_xy_y, etc.) between individual runs
    individually_matched_operating_points_std = {
        f"{key}_std": np.std([
            operating_point_info[key]
            for operating_point_info in individually_matched_operating_points
        ])
        for key in individually_matched_operating_points[0].keys()
    }
    results.update(individually_matched_operating_points_std)

    return results


def calculate_pAUC_from_graph(x, y, pAUC_start: float = 0.0, pAUC_end: float = 1.0, full: bool = False):
    """
    Calculate (partial) Area Under Curve (pAUC) using (x,y) coordinates from the given curve.

    Input:
    For a single curve:
    - x: x values of a curve (e.g., the False Positive Rate points). [x1, x2, .., xn]
    - y: y values of a curve (e.g., the True Positive Rate points). [y1, y2, .., yn]

    For multiple curves:
    - list of x curves, for example the x values observed across multiple runs. [[x1m1, x2m1, .., xnm1], [x1m2, x2m2, ...., xnm2], ..]
    - list of y curves, for example the y values observed across multiple runs. [[y1m1, y2m1, .., ynm1], [y1m2, y2m2, ...., ynm2], ..]

    - pAUC_start: lower bound of x (e.g., FPR) to compute pAUC
    - pAUC_end: higher bound of x (e.g., FPR) to compute pAUC

    Returns:
    - if (full==False): List of pAUC values for each set of ([x1, ..], [y1, ..]) coordinates
    - if (full==True): Metrics as returned by `calculate_statistics` [see there]

    Note: function is not specific to the FROC curve
    """

    if not isinstance(x[0], (list, np.ndarray)) or not isinstance(y[0], (list, np.ndarray)):
        # have a single set of (x,y) coordinates

        assert not isinstance(x[0], (list, np.ndarray)) and not isinstance(y[0], (list, np.ndarray)), \
            "Either provide multiple sequences of (x,y) coordinates, or a single sequence. Obtained a mix of both now. "

        # pack coordinates in format expected by `calculate_statistics`
        coordinates_joined = [(y, x)]
    else:
        # have multiple sets of (x,y) coordinates
        # pack coordinates in format expected by `calculate_statistics`
        coordinates_joined = list(zip(y, x))

    # calculate AUC in the given range
    results = calculate_statistics(metrics=coordinates_joined, x_start=pAUC_start, x_end=pAUC_end)
    if full:
        return results
    return results['auroc_all']
