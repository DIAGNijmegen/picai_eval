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

import concurrent.futures
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (Callable, Dict, Hashable, Iterable, List, Optional, Sized,
                    Tuple, Union)

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics

PathLike = Union[str, Path]


# Compute base prediction metrics TP/FP/FN with associated model confidences
def evaluate_case(
    y_det: "Union[npt.NDArray[np.float32], str, Path]",
    y_true: "Union[npt.NDArray[np.int32], str, Path]",
    min_overlap: float = 0.10,
    overlap_func: "Union[str, Callable[[npt.NDArray[np.float32], npt.NDArray[np.int32]], float]]" = 'IoU',
    case_confidence_func: "Union[str, Callable[[npt.NDArray[np.float32]], float]]" = 'max',
    allow_unmatched_candidates_with_minimal_overlap: bool = True,
    y_det_postprocess_func: "Optional[Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]]" = None,
    y_true_postprocess_func: "Optional[Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]]" = None,
    weight: Optional[float] = None,
    idx: Optional[str] = None,
) -> Tuple[List[Tuple[int, float, float]], float]:
    """
    Gather the list of lesion candidates, and classify in TP/FP/FN.

    Lesion candidates are matched to ground truth lesions, by maximizing the number of candidates
    with sufficient overlap (i.e., matches), and secondly by maximizing the total overlap of all candidates.

    Parameters:
    - y_det: Detection map, which should be a 3D volume containing connected components (in 3D) of the
        same confidence. Each detection map may contain an arbitrary number of connected components,
        with different or equal confidences. Alternatively, y_det may be a filename ending in
        .nii.gz/.mha/.mhd/.npy/.npz, which will be loaded on-the-fly.
    - y_true: Ground truth label, which should be a 3D volume of the same shape as the detection map.
        Alternatively, `y_true` may be the filename ending in .nii.gz/.mha/.mhd/.npy/.npz, which should
        contain binary labels and will be loaded on-the-fly. Use `1` to encode ground truth lesion, and
        `0` to encode background.
    - min_overlap: defines the minimal required overlap (e.g., Intersection over Union or Dice similarity
        coefficient) between a lesion candidate and ground truth lesion, to be counted as a true positive
        detection.
    - overlap_func: function to calculate overlap between a lesion candidate and ground truth mask.
        May be 'IoU' for Intersection over Union, or 'DSC' for Dice similarity coefficient. Alternatively,
        provide a function with signature `func(detection_map, annotation) -> overlap [0, 1]`.
    - allow_unmatched_candidates_with_minimal_overlap: when multiple lesion candidates have sufficient
        overlap with the ground truth lesion mask, this determines whether the lesion that is not selected
        counts as a false positive.
    - y_det_postprocess_func: function to apply to detection map. Can for example be used to extract
        lesion candidates from a softmax prediction volume.
    - y_true_postprocess_func: function to apply to annotation. Can for example be used to select the lesion
        masks from annotations that also contain other structures (such as organ segmentations).

    Returns:
    - a list of tuples with:
        (is_lesion, prediction confidence, overlap)
    - case level confidence score derived from the detection map
    """
    y_list: List[Tuple[int, float, float]] = []
    if isinstance(y_true, (str, Path)):
        y_true = read_label(y_true)
    if isinstance(y_det, (str, Path)):
        y_det = read_prediction(y_det)
    if overlap_func == 'IoU':
        overlap_func = calculate_iou
    elif overlap_func == 'DSC':
        overlap_func = calculate_dsc
    elif isinstance(overlap_func, str):
        raise ValueError(f"Overlap function with name {overlap_func} not recognized. Supported are 'IoU' and 'DSC'")

    # convert dtype to float32
    y_true = y_true.astype('int32')
    y_det = y_det.astype('float32')

    # if specified, apply postprocessing functions
    if y_det_postprocess_func is not None:
        y_det = y_det_postprocess_func(y_det)
    if y_true_postprocess_func is not None:
        y_true = y_true_postprocess_func(y_true)

    # check if detection maps need to be padded
    if y_det.shape[0] < y_true.shape[0]:
        print("Warning: padding prediction to match label!")
        y_det = resize_image_with_crop_or_pad(y_det, y_true.shape)
    if np.min(y_det) < 0:
        raise ValueError("All detection confidences must be positive!")

    # perform connected-components analysis on detection maps
    confidences, indexed_pred = parse_detection_map(y_det)
    lesion_candidate_ids = np.arange(len(confidences))

    if not y_true.any():
        # benign case, all predictions are FPs
        for lesion_confidence in confidences.values():
            y_list.append((0, lesion_confidence, 0.))
    else:
        # malignant case, collect overlap between each prediction and ground truth lesion
        labeled_gt, num_gt_lesions = ndimage.label(y_true, structure=label_structure)
        gt_lesion_ids = np.arange(num_gt_lesions)
        overlap_matrix = np.zeros((num_gt_lesions, len(confidences)))

        for lesion_id in gt_lesion_ids:
            # for each lesion in ground-truth (GT) label
            gt_lesion_mask = (labeled_gt == (1+lesion_id))

            # calculate overlap between each lesion candidate and the current GT lesion
            for lesion_candidate_id in lesion_candidate_ids:
                # calculate overlap between lesion candidate and GT mask
                lesion_pred_mask = (indexed_pred == (1+lesion_candidate_id))
                overlap_score = overlap_func(lesion_pred_mask, gt_lesion_mask)

                # store overlap
                overlap_matrix[lesion_id, lesion_candidate_id] = overlap_score

        # match lesion candidates to ground truth lesion (for documentation on how this works, please see
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html)
        overlap_matrix[overlap_matrix < min_overlap] = 0  # don't match lesions with insufficient overlap
        overlap_matrix[overlap_matrix > 0] += 1  # prioritize matching over the amount of overlap
        matched_lesion_indices, matched_lesion_candidate_indices = linear_sum_assignment(overlap_matrix, maximize=True)

        # remove indices where overlap is zero
        mask = (overlap_matrix[matched_lesion_indices, matched_lesion_candidate_indices] > 0)
        matched_lesion_indices = matched_lesion_indices[mask]
        matched_lesion_candidate_indices = matched_lesion_candidate_indices[mask]

        # all lesion candidates that are matched are TPs
        for lesion_id, lesion_candidate_id in zip(matched_lesion_indices, matched_lesion_candidate_indices):
            lesion_confidence = confidences[lesion_candidate_id]
            overlap = overlap_matrix[lesion_id, lesion_candidate_id]
            overlap -= 1  # return overlap to [0, 1]

            assert overlap > min_overlap, "Overlap must be greater than min_overlap!"

            y_list.append((1, lesion_confidence, overlap))

        # all ground truth lesions that are not matched are FNs
        unmatched_gt_lesions = set(gt_lesion_ids) - set(matched_lesion_indices)
        y_list += [(1, 0., 0.) for _ in unmatched_gt_lesions]

        # all lesion candidates with insufficient overlap/not matched to a gt lesion are FPs
        if allow_unmatched_candidates_with_minimal_overlap:
            candidates_sufficient_overlap = lesion_candidate_ids[(overlap_matrix > 0).any(axis=0)]
            unmatched_candidates = set(lesion_candidate_ids) - set(candidates_sufficient_overlap)
        else:
            unmatched_candidates = set(lesion_candidate_ids) - set(matched_lesion_candidate_indices)
        y_list += [(0, confidences[lesion_candidate_id], 0.) for lesion_candidate_id in unmatched_candidates]

    # determine case-level confidence score
    if case_confidence_func == 'max':
        # take highest lesion confidence as case-level confidence
        case_confidence = np.max(y_det)
    elif case_confidence_func == 'bayesian':
        # if c_i is the probability the i-th lesion is csPCa, then the case-level
        # probability to have one or multiple csPCa lesion is 1 - Π_i{ 1 - c_i}
        case_confidence = 1 - np.prod([(1 - c) for c in confidences.values()])
    else:
        # apply user-defines case-level confidence score function
        case_confidence = case_confidence_func(y_det)

    return y_list, case_confidence, weight, idx


# Evaluate all cases
def evaluate(
    y_det: "Iterable[Union[npt.NDArray[np.float64], str, Path]]",
    y_true: "Iterable[Union[npt.NDArray[np.float64], str, Path]]",
    sample_weight: "Optional[Iterable[float]]" = None,
    subject_list: Optional[Iterable[Hashable]] = None,
    min_overlap: float = 0.10,
    overlap_func: "Union[str, Callable[[npt.NDArray[np.float32], npt.NDArray[np.int32]], float]]" = 'IoU',
    case_confidence_func: "Union[str, Callable[[npt.NDArray[np.float32]], float]]" = 'max',
    allow_unmatched_candidates_with_minimal_overlap: bool = True,
    y_det_postprocess_func: "Optional[Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]]" = None,
    y_true_postprocess_func: "Optional[Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]]" = None,
    num_parallel_calls: int = 3,
    verbose: int = 0,
) -> Metrics:
    """
    Evaluate 3D detection performance.

    Parameters:
    - y_det: iterable of all detection_map volumes to evaluate. Each detection map should a 3D volume
        containing connected components (in 3D) of the same confidence. Each detection map may contain
        an arbitrary number of connected components, with different or equal confidences.
        Alternatively, y_det may contain filenames ending in .nii.gz/.mha/.mhd/.npy/.npz, which will
        be loaded on-the-fly.
    - y_true: iterable of all ground truth labels. Each label should be a 3D volume of the same shape
        as the corresponding detection map. Alternatively, `y_true` may contain filenames ending in
        .nii.gz/.mha/.mhd/.npy/.npz, which should contain binary labels and will be loaded on-the-fly.
        Use `1` to encode ground truth lesion, and `0` to encode background.
    - sample_weight: case-level sample weight. These weights will also be applied to the lesion-level
        evaluation, with same weight for all lesion candidates of the same case.
    - subject_list: list of sample identifiers, to give recognizable names to the evaluation results.
    - min_overlap: defines the minimal required Intersection over Union (IoU) or Dice similarity
        coefficient (DSC) between a lesion candidate and ground truth lesion, to be counted as a true
        positive detection.
    - overlap_func: function to calculate overlap between a lesion candidate and ground truth mask.
        May be 'IoU' for Intersection over Union, or 'DSC' for Dice similarity coefficient. Alternatively,
        provide a function with signature `func(detection_map, annotation) -> overlap [0, 1]`.
    - case_confidence_func: function to derive case-level confidence from detection map. Default: max.
    - allow_unmatched_candidates_with_minimal_overlap: when multiple lesion candidates have sufficient
        overlap with the ground truth lesion mask, this determines whether the lesion that is not selected
        counts as a false positive.
    - y_det_postprocess_func: function to apply to detection map. Can for example be used to extract
        lesion candidates from a softmax prediction volume.
    - y_true_postprocess_func: function to apply to annotation. Can for example be used to select the lesion
        masks from annotations that also contain other structures (such as organ segmentations).
    - num_parallel_calls: number of threads to use for evaluation. Set to 1 to disable parallelization.
    - verbose: (optional) controll amount of printed information.

    Returns:
    - Metrics
    """
    if sample_weight is None:
        sample_weight = itertools.repeat(1)
    if subject_list is None:
        # generate indices to keep track of each case during multiprocessing
        subject_list = itertools.count()

    # initialize placeholders
    case_target: Dict[Hashable, int] = {}
    case_weight: Dict[Hashable, float] = {}
    case_pred: Dict[Hashable, float] = {}
    lesion_results: Dict[Hashable, List[Tuple[int, float, float]]] = {}
    lesion_weight: Dict[Hashable, List[float]] = {}

    # construct case evaluation kwargs
    evaluate_case_kwargs = dict(
        min_overlap=min_overlap,
        overlap_func=overlap_func,
        case_confidence_func=case_confidence_func,
        allow_unmatched_candidates_with_minimal_overlap=allow_unmatched_candidates_with_minimal_overlap,
        y_det_postprocess_func=y_det_postprocess_func,
        y_true_postprocess_func=y_true_postprocess_func,
    )

    with ThreadPoolExecutor(max_workers=num_parallel_calls) as pool:
        if num_parallel_calls >= 2:
            # process the cases in parallel
            futures = {
                pool.submit(
                    evaluate_case,
                    y_det=y_det_case,
                    y_true=y_true_case,
                    weight=weight,
                    idx=idx,
                    **evaluate_case_kwargs
                ): idx
                for (y_det_case, y_true_case, weight, idx) in zip(y_det, y_true, sample_weight, subject_list)
            }

            iterator = concurrent.futures.as_completed(futures)
        else:
            # process the cases sequentially
            def func(y_det_case, y_true_case, weight, idx):
                return evaluate_case(
                    y_det=y_det_case,
                    y_true=y_true_case,
                    weight=weight,
                    idx=idx,
                    **evaluate_case_kwargs
                )

            iterator = map(func, y_det, y_true, sample_weight, subject_list)

        if verbose:
            total: Optional[int] = None
            if isinstance(subject_list, Sized):
                total = len(subject_list)
            iterator = tqdm(iterator, desc='Evaluating', total=total)

        for result in iterator:
            if isinstance(result, tuple):
                # single-threaded evaluation
                lesion_results_case, case_confidence, weight, idx = result
            elif isinstance(result, concurrent.futures.Future):
                # multi-threaded evaluation
                lesion_results_case, case_confidence, weight, idx = result.result()
            else:
                raise TypeError(f'Unexpected result type: {type(result)}')

            # aggregate results
            case_weight[idx] = weight
            case_pred[idx] = case_confidence
            if len(lesion_results_case):
                case_target[idx] = np.max([a[0] for a in lesion_results_case])
            else:
                case_target[idx] = 0

            # accumulate outputs
            lesion_results[idx] = lesion_results_case
            lesion_weight[idx] = [weight] * len(lesion_results_case)

    # collect results in a Metrics object
    metrics = Metrics(
        lesion_results=lesion_results,
        case_target=case_target,
        case_pred=case_pred,
        case_weight=case_weight,
        lesion_weight=lesion_weight,
    )

    return metrics


def evaluate_folder(
    y_det_dir: Union[Path, str],
    y_true_dir: Optional[Union[Path, str]] = None,
    subject_list: Optional[Union[List[str], PathLike]] = None,
    pred_extensions: Optional[List[str]] = None,
    label_extensions: Optional[List[str]] = None,
    detection_map_postfixes: Optional[List[str]] = None,
    label_postfixes: Optional[List[str]] = None,
    verbose: int = 1,
    **kwargs
) -> Metrics:
    """
    Evaluate 3D detection performance, for all samples in y_det_dir,
    or the samples specified in the subject_list.

    Parameters:
    - y_det_dir: path to folder containing the detection maps.
    - y_true_dir: (optional) path to folder containing the annotations. Defaults to y_true_dir.
    - subject_list: (optional) list of cases to evaluate. Allows to evaluate a subset of cases in a folder,
                    and ensures all specified cases were found. If str or Path, will load the subject list
                    from the provided JSON file, which should contain a dictionary with "subject_list" entry.
    - pred_extensions: allowed file extensions for detection maps.
    - label_extensions: allowed file extensions for annotations.
    - detection_map_postfixes: allowed postifxes for detection maps.
    - label_postfixes: allowed postifxes for annotations.
    - verbose: (optional) controll amount of printed information.
    **kwargs: (optional) see `evaluate` for additional options

    Returns:
    - Metrics
    """
    if y_true_dir is None:
        y_true_dir = y_det_dir
    if pred_extensions is None:
        pred_extensions = [".npz", ".npy", ".nii.gz", ".nii", ".mha", ".mhd"]
    if label_extensions is None:
        label_extensions = [".nii.gz", ".nii", ".mha", ".mhd", ".npz", ".npy"]
    if detection_map_postfixes is None:
        detection_map_postfixes = ["_detection_map"]
        if y_true_dir != y_det_dir:
            # if annotation directory is specified, also look for [subject_id].nii.gz etc
            detection_map_postfixes += [""]
    if label_postfixes is None:
        label_postfixes = ["_label"]
        if y_true_dir != y_det_dir:
            # if annotation directory is specified, also look for [subject_id].nii.gz etc
            label_postfixes += [""]
    if isinstance(subject_list, (str, Path)):
        with open(subject_list) as fp:
            subject_list = json.load(fp)["subject_list"]

    # combine postfixes and extensions in a single list
    detection_map_postfixes = [
        f"{postfix}{extension}"
        for postfix in detection_map_postfixes
        for extension in pred_extensions
    ]
    label_postfixes = [
        f"{postfix}{extension}"
        for postfix in label_postfixes
        for extension in label_extensions
    ]

    y_det = []
    y_true = []
    if subject_list:
        # collect the detection maps and labels for each case specified in the subject list
        for subject_id in subject_list:
            # construct paths to detection maps and labels
            found_pred, found_label = False, False
            for postfix in detection_map_postfixes:
                detection_path = os.path.join(y_det_dir, f"{subject_id}{postfix}")
                if os.path.exists(detection_path):
                    y_det += [detection_path]
                    found_pred = True
                    break

            for postfix in label_postfixes:
                label_path = os.path.join(y_true_dir, f"{subject_id}{postfix}")
                if os.path.exists(label_path):
                    y_true += [label_path]
                    found_label = True
                    break

            # check if detection map and label are found
            assert found_pred, f"Did not find prediction for {subject_id} in {y_det_dir}!"
            assert found_label, f"Did not find label for {subject_id} in {y_true_dir}!"
    else:
        # collect all detection maps found in detection_map_dir
        file_list = sorted(os.listdir(y_det_dir))
        subject_list = []
        if verbose >= 1:
            print(f"Found {len(file_list)} files in the input directory, collecting detection_mapes with " +
                  f"{detection_map_postfixes} and labels with {label_postfixes}.")

        # collect filenames of detection_map predictions
        for fn in file_list:
            for postfix in detection_map_postfixes:
                if postfix in fn:
                    y_det += [os.path.join(y_det_dir, fn)]
                    subject_id = fn.replace(postfix, "")
                    if subject_id in subject_list:
                        print(f"Found multiple detection maps for {subject_id}, skipping {fn}!")
                        continue

                    subject_list += [subject_id]
                    break

        # collect filenames of labels
        for subject_id in subject_list:
            found_label = False
            for postfix in label_postfixes:
                label_path = os.path.join(y_true_dir, f"{subject_id}{postfix}")
                if os.path.exists(label_path):
                    y_true += [label_path]
                    found_label = True
                    break

            assert found_label, f"Did not find label for {subject_id} in {y_true_dir}!"

    # ensure files exist
    for subject_id, detection_path in zip(subject_list, y_det):
        assert os.path.exists(detection_path), f"Could not find prediction for {subject_id} at {detection_path}!"
    for subject_id, label_path in zip(subject_list, y_true):
        assert os.path.exists(label_path), f"Could not find label for {subject_id} at {label_path}!"

    if verbose >= 1:
        print(f"Found prediction and label for {len(y_det)} cases. Here are some examples:")
        print(subject_list[0:5])

    # check if predictions were found
    assert len(y_det), f"Did not find any predictions in {y_det_dir}!"

    # perform evaluation with compiled file lists
    return evaluate(y_det=y_det, y_true=y_true, subject_list=subject_list, verbose=verbose, **kwargs)
