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

import os
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import itertools

from typing import List, Tuple, Dict, Union, Optional, Callable, Iterable, Hashable, Sized
try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

from picai_eval.metrics import Metrics
from picai_eval.image_utils import (
    resize_image_with_crop_or_pad, read_label, read_prediction
)
from picai_eval.analysis_utils import (
    parse_detection_map, calculate_iou, calculate_dsc, label_structure
)


# Compute base prediction metrics TP/FP/FN with associated model confidences
def evaluate_case(
    y_det: "Union[npt.NDArray[np.float32], str]",
    y_true: "Union[npt.NDArray[np.int32], str]",
    min_overlap: float = 0.10,
    overlap_func: "Union[str, Callable[[npt.NDArray[np.float32], npt.NDArray[np.int32]], float]]" = 'IoU',
    case_confidence_func: "Union[str, Callable[[npt.NDArray[np.float32]], float]]" = 'max',
    multiple_lesion_candidates_selection_criteria: str = 'overlap',
    allow_unmatched_candidates_with_minimal_overlap: bool = True,
    y_det_postprocess_func: "Optional[Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]]" = None,
    y_true_postprocess_func: "Optional[Callable[[npt.NDArray[np.int32]], npt.NDArray[np.int32]]]" = None,
) -> Tuple[List[Tuple[int, float, float]], float]:
    """
    Gather the list of lesion candidates, and classify in TP/FP/FN.

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
    - multiple_lesion_candidates_selection_criteria: when multiple lesion candidates have sufficient
        overlap with the ground truth lesion mask, this determines which lesion candidate is selected as TP.
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
    if isinstance(y_true, str):
        y_true = read_label(y_true)
    if isinstance(y_det, str):
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

    lesion_candidates_best_overlap: Dict[str, float] = {}

    if y_true.any():
        # for each malignant scan
        labeled_gt, num_gt_lesions = ndimage.label(y_true, structure=label_structure)

        for lesion_id in range(1, num_gt_lesions+1):
            # for each lesion in ground-truth (GT) label
            gt_lesion_mask = (labeled_gt == lesion_id)

            if overlap_func in [calculate_iou, calculate_dsc]:
                # collect indices of lesion candidates that have any overlap with the current GT lesion
                overlapping_lesion_candidate_indices = set(np.unique(indexed_pred[gt_lesion_mask]))
            else:
                # we do not know which overlap function is employed, so can't speed up with above method
                overlapping_lesion_candidate_indices = set(np.unique(indexed_pred))
            overlapping_lesion_candidate_indices -= {0}  # remove index 0, if present

            # collect lesion candidates for current GT lesion
            lesion_candidates_for_target_gt: List[Dict[str, Union[int, float]]] = []
            for lesion_candidate_id, lesion_confidence in confidences:
                if lesion_candidate_id in overlapping_lesion_candidate_indices:
                    # calculate overlap between lesion candidate and GT mask
                    lesion_pred_mask = (indexed_pred == lesion_candidate_id)
                    overlap_score = overlap_func(lesion_pred_mask, gt_lesion_mask)

                    # keep track of the highest overlap a lesion candidate has with any GT lesion
                    lesion_candidates_best_overlap[lesion_candidate_id] = max(
                        overlap_score, lesion_candidates_best_overlap.get(lesion_candidate_id, 0)
                    )

                    # store lesion candidate info for current GT mask
                    lesion_candidates_for_target_gt.append({
                        'id': lesion_candidate_id,
                        'confidence': lesion_confidence,
                        'overlap': overlap_score,
                    })

            if len(lesion_candidates_for_target_gt) == 0:
                # no lesion candidate matched with GT mask. Add FN.
                y_list.append((1, 0., 0.))
            else:
                # multiple predictions for current GT lesion
                # sort lesion candidates based on overlap or confidence
                key = multiple_lesion_candidates_selection_criteria
                lesion_candidates_for_target_gt = sorted(lesion_candidates_for_target_gt, key=lambda x: x[key], reverse=True)

                gt_lesion_matched = False
                for candidate_info in lesion_candidates_for_target_gt:
                    lesion_pred_mask = (indexed_pred == candidate_info['id'])

                    if candidate_info['overlap'] > min_overlap:
                        indexed_pred[lesion_pred_mask] = 0
                        y_list.append((1, candidate_info['confidence'], candidate_info['overlap']))
                        gt_lesion_matched = True
                        break

                if not gt_lesion_matched:
                    # ground truth lesion not matched to a lesion candidate. Add FN.
                    y_list.append((1, 0., 0.))

        # Remaining lesions are FPs
        remaining_lesions = set(np.unique(indexed_pred))
        remaining_lesions -= {0}  # remove index 0, if present
        for lesion_candidate_id, lesion_confidence in confidences:
            if lesion_candidate_id in remaining_lesions:
                overlap_score = lesion_candidates_best_overlap.get(lesion_candidate_id, 0)
                if allow_unmatched_candidates_with_minimal_overlap and overlap_score > min_overlap:
                    # The lesion candidate was not matched to a GT lesion, but did have overlap > min_overlap
                    # with a GT lesion. The GT lesion is, however, matched to another lesion candidate.
                    # In this operation mode, this lesion candidate is not considered as a false positive.
                    pass
                else:
                    y_list.append((0, lesion_confidence, 0.))  # add FP

    else:
        # for benign case, all predictions are FPs
        for _, lesion_confidence in confidences:
            y_list.append((0, lesion_confidence, 0.))

    # determine case-level confidence score
    if case_confidence_func == 'max':
        # take highest lesion confidence as case-level confidence
        case_confidence = np.max(y_det)
    elif case_confidence_func == 'bayesian':
        # if a_i is the probability the i-th lesion is csPCa, then the case-level
        # probability to have one or multiple csPCa lesion is 1 - Π_i{ 1 - a_i}
        case_confidence = 1 - np.prod([(1 - a[1]) for a in confidences])
    else:
        # apply user-defines case-level confidence score function
        case_confidence = case_confidence_func(y_det)

    return y_list, case_confidence


# Evaluate all cases
def evaluate(
    y_det: "Iterable[Union[npt.NDArray[np.float64], str, Path]]",
    y_true: "Iterable[Union[npt.NDArray[np.float64], str, Path]]",
    sample_weight: "Optional[Iterable[float]]" = None,
    subject_list: Optional[Iterable[Hashable]] = None,
    min_overlap: float = 0.10,
    overlap_func: "Union[str, Callable[[npt.NDArray[np.float32], npt.NDArray[np.int32]], float]]" = 'IoU',
    case_confidence_func: "Union[str, Callable[[npt.NDArray[np.float32]], float]]" = 'max',
    multiple_lesion_candidates_selection_criteria: str = 'overlap',
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
    - multiple_lesion_candidates_selection_criteria: when multiple lesion candidates have sufficient
        overlap with the ground truth lesion mask, this determines which lesion candidate is selected as TP.
    - allow_unmatched_candidates_with_minimal_overlap: when multiple lesion candidates have sufficient
        overlap with the ground truth lesion mask, this determines whether the lesion that is not selected
        counts as a false positive.
    - y_det_postprocess_func: function to apply to detection map. Can for example be used to extract
        lesion candidates from a softmax prediction volume.
    - y_true_postprocess_func: function to apply to annotation. Can for example be used to select the lesion
        masks from annotations that also contain other structures (such as organ segmentations).
    - num_parallel_calls: number of threads to use for evaluation.
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

    with ThreadPoolExecutor(max_workers=num_parallel_calls) as pool:
        # define the functions that need to be processed: compute_pred_vector, with each individual
        # detection_map prediction, ground truth label and parameters
        future_to_args = {
            pool.submit(
                evaluate_case,
                y_det=y_det_case,
                y_true=y_true_case,
                min_overlap=min_overlap,
                overlap_func=overlap_func,
                case_confidence_func=case_confidence_func,
                multiple_lesion_candidates_selection_criteria=multiple_lesion_candidates_selection_criteria,
                allow_unmatched_candidates_with_minimal_overlap=allow_unmatched_candidates_with_minimal_overlap,
                y_det_postprocess_func=y_det_postprocess_func,
                y_true_postprocess_func=y_true_postprocess_func
            ): (idx, weight)
            for (y_det_case, y_true_case, weight, idx) in zip(y_det, y_true, sample_weight, subject_list)
        }

        # process the cases in parallel
        iterator = concurrent.futures.as_completed(future_to_args)
        if verbose:
            total: Optional[int] = None
            if isinstance(subject_list, Sized):
                total = len(subject_list)
            iterator = tqdm(iterator, desc='Evaluating', total=total)
        for future in iterator:
            # unpack results
            lesion_results_case, case_confidence = future.result()

            # aggregate results
            idx, weight = future_to_args[future]
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
    subject_list: Optional[List[str]] = None,
    extensions: Optional[List[str]] = None,
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
                    and ensures all specified cases were found.
    - extensions: allowed file formats for detection maps and annotations.
    - detection_map_postfixes: allowed postifxes for detection maps.
    - label_postfixes: allowed postifxes for annotations.
    - verbose: (optional) controll amount of printed information.
    **kwargs: (optional) see `evaluate` for additional options

    Returns:
    - Metrics
    """
    if y_true_dir is None:
        y_true_dir = y_det_dir
    if extensions is None:
        extensions = [".npz", ".npy", ".nii.gz", ".nii", ".mha", ".mhd"]
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

    # combine postfixes and extensions in a single list
    detection_map_postfixes = [
        f"{postfix}{extension}"
        for postfix in detection_map_postfixes
        for extension in extensions
    ]
    label_postfixes = [
        f"{postfix}{extension}"
        for postfix in label_postfixes
        for extension in extensions
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

    # perform evaluation with compiled file lists
    return evaluate(y_det=y_det, y_true=y_true, subject_list=subject_list, verbose=verbose, **kwargs)
