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
from collections.abc import Sequence
from itertools import repeat
from multiprocessing import Pool
from typing import Any, Callable, Dict, Hashable, List, Optional, Union

import numpy as np
from mlxtend.evaluate import permutation_test
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from tqdm import tqdm

try:
    from picai_eval.stat_util.stat_util import pvalue_stat
except ImportError:  # pragma: no cover
    print("Could not find picai_eval.stat_util. Please install picai_eval including submodules (use pip >= 21.0)")

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

"""
Binary PCa Detection in mpMRI
Script:         Statistical helper - (non-parametric) statistical test
Contributor:    joeranbosma
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
"""

IntDictOrArraylike = "Union[Dict[Hashable, int], Sequence[int], npt.NDArray[np.int_]]"
FloatDictOrArraylike = "Union[Dict[Hashable, float], Sequence[float], npt.NDArray[np.float_]]"


def calc_sensitivity(y_true, y_pred, sample_weight=None):
    """Calculate sensitivity"""
    _, _, fn, tp = confusion_matrix(y_true, y_pred, sample_weight=sample_weight).ravel()
    return tp / (tp+fn)


def calc_specificity(y_true, y_pred, sample_weight=None):
    """Calculate specificity"""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, sample_weight=sample_weight).ravel()
    return tn / (tn+fp)


def calc_order_auc_score(
    scores_baseline: "Union[Sequence[float], npt.NDArray[np.float_]]",
    scores_alternative: "Union[Sequence[float], npt.NDArray[np.float_]]"
) -> float:
    """Calculate the order AUC for performance(alternative) > performance(baseline)"""
    # pose order AUC as classification AUC to use roc_auc_score
    scores_all = list(scores_baseline) + list(scores_alternative)
    labels_all = [0]*len(list(scores_baseline)) + [1]*len(list(scores_alternative))
    auc_score = roc_auc_score(y_true=labels_all, y_score=scores_all)

    return float(auc_score)


def perform_permutation_test(
    scores_baseline: "Union[Sequence[float], npt.NDArray[np.float_]]",
    scores_alternative: "Union[Sequence[float], npt.NDArray[np.float_]]",
    method: Optional[str] = None,
    iterations: int = 1_000_000
) -> float:
    """
    Calculate p-value for comparison of baseline with alternative configuration.
    H0 (null hypothesis): difference in performance comes from random fluctuation
    H1 (alternative hypothesis): alternative configuration is better

    Definitions:
    - configuration: either baseline or alternative
    - performance: metric for performance (e.g., AUC, pAUC between 0.01 and 2.5, accuracy, ...), where higher is better.
    - run: a single fully trained model. Having 3 restarts and 5-fold cross-validation will give 15 runs.

    Usage:
    Collect the performances observed for each run in lists:
    - scores_baseline: [performance baseline run 1, performance baseline run 2, ..., performance baseline run N]
    - scores_alternative: [performance alternative run 1, performance alternative run 2, ..., performance alternative run M]

    - iterations: number of permuations to perform to estimate the distribution of the null hypothesis (H0)

    Returns:
    - p: p-value for null hypothesis (so alternative configuration is better than the baseline if p is sufficiently small)
    """
    if method is None:
        # use exact permutation test if feasible, or approximate permutation test otherwise
        method = 'exact' if len(scores_baseline)+len(scores_alternative) <= 20 else 'approximate'

    # perform permutation test
    p = permutation_test(
        x=scores_baseline, y=scores_alternative, func=calc_order_auc_score,
        method=method, num_rounds=iterations
    )

    return float(p)


def permutation_test_wrapper(args) -> float:
    return perform_permutation_test(**args)


def multiple_permutation_tests(
    all_scores: "Dict[str, Dict[str, Union[Sequence[float], npt.NDArray[np.float_]]]]",
    iterations: int = 1_000_000
) -> Dict[str, float]:
    """
    Perform multiple permutation tests using multiprocessing.

    Input:
    - scores: {
        'name of comparison 1': {
            'scores_baseline': [baseline scores comparison 1 (see `perform_permutation_test`)],
            'scores_alternative': [alternative scores comparison 1 (see `perform_permutation_test`)],
        },
        'name of comparison 2': {
            'scores_baseline': [baseline scores comparison 2 (see `perform_permutation_test`)],
            'scores_alternative': [alternative scores comparison 2 (see `perform_permutation_test`)],
        },
        ...,
        'name of comparison n': {
            'scores_baseline': [baseline scores comparison n (see `perform_permutation_test`)],
            'scores_alternative': [alternative scores comparison n (see `perform_permutation_test`)],
        },
    }

    Example scores:

    Returns:
    - p_values: {
        'name of comparison 1': p-value for H0 ≥ H1 of comparison 1 (see `perform_permutation_test`),
        'name of comparison 2': p-value for H0 ≥ H1 of comparison 2 (see `perform_permutation_test`),
        ...,
        'name of comparison n': p-value for H0 ≥ H1 of comparison n (see `perform_permutation_test`),
    }
    """
    # define placeholder
    p_values: Dict[str, float] = {}

    with concurrent.futures.ProcessPoolExecutor() as pool:
        jobs = {}
        for name, scores in all_scores.items():
            # prepare input arguments of permutation test
            args: Dict[str, Any] = dict(scores)
            args['iterations'] = iterations

            # add permutation test to the queue
            future = pool.submit(permutation_test_wrapper, args)
            jobs[future] = name

        for future in tqdm(concurrent.futures.as_completed(jobs), total=len(scores), desc="Performing permutation tests"):
            # store result of permutation test
            name = jobs[future]
            p_values[name] = future.result()

    return p_values


def match_reader(y_true, y_pred_ai, y_pred_reader, sample_weight=None, match='sensitivity'):
    # input conversion and validation
    y_true = np.array(y_true)
    y_pred_ai = np.array(y_pred_ai)
    y_pred_reader = np.array(y_pred_reader)
    assert len(y_true) == len(y_pred_reader)

    if y_true.shape != y_pred_ai.shape:
        return np.array([
            match_reader(
                y_true=y_true,
                y_pred_ai=y_pred_ai_restart,
                y_pred_reader=y_pred_reader,
                sample_weight=sample_weight,
                match=match
            )
            for y_pred_ai_restart in y_pred_ai
        ])

    assert len(y_true) == len(y_pred_ai)

    # calculate metric (sensitivity/specificity/recall/precision) for reader
    if match == 'sensitivity':
        performance_reader = calc_sensitivity(y_true=y_true, y_pred=y_pred_reader, sample_weight=sample_weight)
    elif match == 'specificity':
        performance_reader = calc_specificity(y_true=y_true, y_pred=y_pred_reader, sample_weight=sample_weight)
    else:
        raise ValueError

    # match AI operating point to reader
    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_pred_ai, sample_weight=sample_weight)
    if match == 'sensitivity':
        performance_ai = tpr
    elif match == 'specificity':
        performance_ai = 1 - fpr

    diff = np.abs(performance_ai - performance_reader)

    # grab index of closest point
    viable_thresholds = thresholds[diff == np.min(diff)]
    threshold = np.random.choice(viable_thresholds)
    # closest_idx = np.argmin(diff)
    # threshold = thresholds[closest_idx]

    return (y_pred_ai >= threshold).astype(int)


def match_then_compare(y_true, y_pred_ai, y_pred_reader, sample_weight=None, match='sensitivity', verbose=False):
    # match AI operating point to the reader
    y_pred_ai_matched = match_reader(
        y_true=y_true,
        y_pred_ai=y_pred_ai,
        y_pred_reader=y_pred_reader,
        sample_weight=sample_weight,
        match=match
    )

    # evaluate AI and reader at matched operating point
    kappa_ai = cohen_kappa_score(y1=y_true, y2=y_pred_ai_matched, sample_weight=sample_weight)
    kappa_reader = cohen_kappa_score(y1=y_true, y2=y_pred_reader, sample_weight=sample_weight)

    print(f"κR = {kappa_reader:.4f}, κAI = {kappa_ai:.4f}") if verbose else None

    # return test statistic
    return kappa_reader - kappa_ai


def sample_then_match_then_compare(y_true, y_pred_ai, y_pred_reader, sample_weight, match='sensitivity'):
    indices = np.random.randint(0, len(y_true), len(y_true))
    while len(np.unique(y_true[indices])) < 2:
        # Reject one class samples
        indices = np.random.randint(0, len(y_true), len(y_true))

    z = match_then_compare(
        y_true=y_true[indices],
        y_pred_ai=y_pred_ai[indices],
        y_pred_reader=y_pred_reader[indices],
        sample_weight=sample_weight[indices],
        match=match
    )

    return z


def sample_then_match_then_compare_wrapper(args):  # pragma: no cover
    return sample_then_match_then_compare(**args)


def validate_and_convert(*inputs) -> "List[npt.NDArray[Any]]":
    """
    Validate inputs:
    - If the inputs consists of at least one dictionary, all inputs should be a dictionary
    - If the inputs are dictionaries, check if the keys are the same

    Convert inputs:
    - If the inputs are dictionaries, reduce to flat lists, with the same order for each input
    - Convert to numpy arrays
    """
    # check if any of the inputs is a dictionary
    any_dict = False
    for inp in inputs:
        if isinstance(inp, dict):
            any_dict = True

    if any_dict:
        # check if all inputs are dictionaries
        assert all(isinstance(inp, dict) for inp in inputs), (
            "Inputs must either all be dictionaries, or all "
            "iterables (with cases in the same order)!"
        )

        # check if all cases are present in each dictionary
        cases = set(list(inputs[0]))
        assert all(cases == set(list(inp)) for inp in inputs), \
            "Inputs must all contain the same cases!"

        # collect flat lists with cases in the same order
        cases = sorted(list(cases))
        inputs = ([inp[c] for c in cases] for inp in inputs)

    # convert to numpy arrays
    return [np.array(inp) for inp in inputs]


def perform_matched_boostrapping(
    y_true: "Union[Dict[Hashable, int], Sequence[int], npt.NDArray[np.int_]]",
    y_pred_ai: "Union[Dict[Hashable, float], Sequence[float], npt.NDArray[np.float_]]",
    y_pred_reader: "Union[Dict[Hashable, int], Sequence[int], npt.NDArray[np.int_]]",
    match: str = "sensitivity",
    conjugate_performance_metric: Optional[Union[str, Callable[[Any, Any, Optional[Any]], float]]] = None,
    flavour: str = "match>sample>compare",
    sample_weight: "Optional[Union[Dict[Hashable, float], Sequence[float], npt.NDArray[np.float_]]]" = None,
    iterations: int = 1_000_000,
    seed: Optional[int] = None,
    verbose: bool = True
) -> float:
    """
    Perform bootstrapping to estimate the probability that AI outperforms the reader.

    For sample>match>compare, in each iteration of bootstrapping:
    1. select ~U(0, N) samples with replacement
    2. match the AI operating point to the operating point of the reader, by selecting
       the operating point with sensitivity/specificity/recall/precision closest to the
       reader. After matching, the AI predictions are binarized using the selected threshold.
    3. calculate the target metric: reader κ - AI κ
    Iterations that sampled only one class are rejected.

    For match>sample>compare, in each iteration of bootstrapping:
    1. Match the AI operating point to the operating point of the reader, by selecting
       the operating point with sensitivity/specificity/recall/precision closest to the
       reader. After matching, the AI predictions are binarized using the selected threshold.
    2. select ~U(0, N) samples with replacement
    3. calculate the target metric: reader κ - AI κ
    Iterations that sampled only one class are rejected.

    Input:
    - y_true: ground truth containing 0 or 1 for each case, as dictionary (recommended) or list
    - y_pred_ai: continuous predictions for each case, as dictionary (recommended) or list
    - y_pred_reader: binarized predictions for each case, as dictionary (recommended) or list
    - flavour: either "match>sample>compare" or "sample>match>compare"

    Returns:
    - p: estimate of probability for Performance(AI) ≥ Performance(Reader)
    """
    if sample_weight is not None:
        y_true, y_pred_ai, y_pred_reader, sample_weight = validate_and_convert(
            y_true, y_pred_ai, y_pred_reader, sample_weight
        )
    else:
        y_true, y_pred_ai, y_pred_reader = validate_and_convert(
            y_true, y_pred_ai, y_pred_reader
        )

    # convert values
    y_true = y_true.astype(int)
    y_pred_ai = y_pred_ai.astype(float)
    y_pred_reader = y_pred_reader.astype(int)
    assert np.min(y_pred_reader) >= 0 and np.max(y_pred_reader) <= 1, \
        "Reader predictions should be thresholded (e.g., PI-RADS ≥ 4)."
    if sample_weight is not None:
        sample_weight = sample_weight.astype(float)
    else:
        sample_weight = np.ones_like(y_true, dtype=float)

    assert y_true.shape == y_pred_reader.shape and len(y_true.shape) == 1, \
        f"Recieved unexpected shape for y_true (after conversion): {y_true.shape}"

    assert len(y_pred_ai.shape) <= 2, f"Unexpected shape for y_pred_ai (after conversion): {y_pred_ai.shape}"
    if y_pred_ai.shape[-1] != y_true.shape[-1]:
        y_pred_ai = y_pred_ai.transpose()

    if conjugate_performance_metric is None:
        # determine conjugate performance metric
        conjugate_performance_metric = get_conjugate_performance_metric(match)

    if isinstance(conjugate_performance_metric, str):
        # determine conjugate performance metric function
        conjugate_performance_metric = get_performance_metric_func(conjugate_performance_metric)

    if flavour == "match>sample>compare":
        # match AI operating point to the reader
        y_pred_ai_matched = match_reader(
            y_true=y_true,
            y_pred_ai=y_pred_ai,
            y_pred_reader=y_pred_reader,
            sample_weight=sample_weight,
            match=match
        )

        # perform bootstrapping, see https://github.com/mateuszbuda/ml-stat-util
        p, z_subsets = pvalue_stat(
            y_true=y_true,
            y_preds1=y_pred_reader,
            y_preds2=y_pred_ai_matched,
            score_fun=conjugate_performance_metric,
            stat_fun=lambda x: x,
            compare_fun=lambda reader1_scores, reader2_scores: 0.5 - calc_order_auc_score(reader1_scores, reader2_scores),
            sample_weight=sample_weight,
            two_tailed=False,
            n_bootstraps=iterations,
            seed=seed,
        )
    elif flavour == "sample>match>compare":
        with Pool(8) as pool:
            # perform bootstrapping
            # with itertool's repeat the input parameters do not have to be initialized thousands of times
            args = dict(y_true=y_true, y_pred_ai=y_pred_ai, y_pred_reader=y_pred_reader, match=match, sample_weight=sample_weight)
            z_subsets = pool.map(sample_then_match_then_compare_wrapper, repeat(args, iterations))

        # compute probability estimate
        p = np.mean([z < 0 for z in z_subsets])
    else:
        raise ValueError

    if verbose:
        print(f"Probability for Performance(AI) ≥ Performance(Reader): p = {p:.4e}")
        # print(f"z_subsets = {np.mean(z_subsets):.4f} ± {np.std(z_subsets):.4f}")

    return float(p)


def get_conjugate_performance_metric(performance_metric: str):
    # determine conjugate performance metric
    conjugate_performance_metrics = {
        'sensitivity': 'specificity',
        'specificity': 'sensitivity',
        'recall': 'precision',
        'precision': 'recall',
    }
    assert performance_metric in conjugate_performance_metrics, \
        f"Conjugate performance metric for {performance_metric} not found!"

    return conjugate_performance_metrics[performance_metric]


def get_performance_metric_func(performance_metric: str):
    performance_metrics = {
        'sensitivity': calc_sensitivity,
        'specificity': calc_specificity,
        # 'recall': calc_recall,  # TODO: add this
        # 'precision': calc_precision,  # TODO: add this
    }
    assert performance_metric in performance_metrics, \
        f"Implementation of {performance_metric} not found!"

    return performance_metrics[performance_metric]


def perform_matched_permutation_test(
    y_true: IntDictOrArraylike,
    y_pred_ai: FloatDictOrArraylike,
    y_pred_readers: FloatDictOrArraylike,
    match: Union[str, Callable[[Any, Any, Optional[Any]], float]] = "sensitivity",
    conjugate_performance_metric: Optional[Union[str, Callable[[Any, Any, Optional[Any]], float]]] = None,
    sample_weight: Optional[FloatDictOrArraylike] = None,
    iterations: int = 1_000_000,
    verbose: bool = True
) -> float:
    """
    Perform a permutation test to estimate the probability that AI outperforms the panel of readers.

    Input:
    - y_true: ground truth containing 0 or 1 for each case, as dictionary (recommended) or list
    - y_pred_ai: continuous predictions for each case, as dictionary (recommended) or list
    - y_pred_readers: binarized predictions for each case, as dictionary (recommended) or list

    For the readers, the performance metric is obtained in two steps:
        1) for each reader we consider an operating point (PI-RADS 3+ or 4+) and match the trained
           AI models to the reader's sensitivity/specificity/recall/precision,
        2) we calculate the conjugate performance of the reader with respect to the average conjugate
           performance of the AI models. For example, for cased-based analysis, after matching the
           sensitivity, the reader's performance metric is calculated as:
           P_j=spec_R^j-1/N ∑_(i=1)^N〖spec_AI^i 〗, with spec_R^j the reader's specificity and spec_AI^i
           the specificity of trained AI model i at the same sensitivity as the reader.
    This is repeated for each reader, resulting in the performance with respect to the average AI
    performance for each reader.

    To account for variability in AI performance, the same method is applied to each trained AI model:
    we calculate the performance of trained AI models with respect to the average AI performance,
    averaged across the operating points of each reader. For case-based analysis, where sensitivity is
    matched, this evaluates to: P_k=1/M ∑_(j=1)^M(spec_AI^k-1/N ∑_(i=1)^N〖spec_AI^i 〗),  where spec_AI^k
    is the specificity of trained AI model k and spec_AI^i is the specificity of trained AI model i,
    both at the same sensitivity as reader j

    To account for deviation from the population encountered during clinical routine, perform inverse probability
    weighting by providing the weights per sample (see McKinney et al., 2020, Mansournia et al., 2016, Pinsky et al., 2012).

    Returns:
    - p: probability for Performance(AI) ≥ Performance(Panel of readers)
    """
    # validate and convert inputs
    if sample_weight is not None:
        y_true, y_pred_ai, y_pred_readers, sample_weight = validate_and_convert(
            y_true, y_pred_ai, y_pred_readers, sample_weight
        )
    else:
        y_true, y_pred_ai, y_pred_readers = validate_and_convert(
            y_true, y_pred_ai, y_pred_readers
        )

    assert len(y_true.shape) == 1, \
        f"Recieved unexpected shape for y_true (after conversion): {y_true.shape}"

    assert len(y_pred_ai.shape) == 2, f"Unexpected shape for y_pred_ai (after conversion): {y_pred_ai.shape}"
    assert len(y_pred_readers.shape) == 2, f"Unexpected shape for y_pred_readers (after conversion): {y_pred_readers.shape}"
    if y_pred_ai.shape[-1] != y_true.shape[-1]:
        y_pred_ai = y_pred_ai.transpose()
    if y_pred_readers.shape[-1] != y_true.shape[-1]:
        y_pred_readers = y_pred_readers.transpose()

    if conjugate_performance_metric is None:
        # determine conjugate performance metric
        conjugate_performance_metric = get_conjugate_performance_metric(match)

    if isinstance(conjugate_performance_metric, str):
        # determine conjugate performance metric function
        conjugate_performance_metric = get_performance_metric_func(conjugate_performance_metric)

    # keep track of conjugate AI scores, after matching to a reader
    all_scores_ai = []

    # collect performance metrics for readers w.r.t. averge AI
    scores_readers = []
    for y_pred_reader in y_pred_readers:
        # print(f"Sensitivity reader: {calc_sensitivity(y_true, y_pred_reader)}")
        # for each reader, match each AI instance to the reader's performance,
        # and collect the conjugate performance
        matched_conjugate_scores_ai = []
        for y_pred_ai_instance in y_pred_ai:
            y_pred_ai_matched = match_reader(
                y_true=y_true,
                y_pred_ai=y_pred_ai_instance,
                y_pred_reader=y_pred_reader,
                sample_weight=sample_weight,
                match=match
            )

            score_ai = conjugate_performance_metric(y_true, y_pred_ai_matched)
            matched_conjugate_scores_ai.append(score_ai)

        # calculate performance of reader w.r.t. AI and store results
        score_reader = conjugate_performance_metric(y_true, y_pred_reader)
        scores_readers.append(calc_order_auc_score(matched_conjugate_scores_ai, [score_reader]))

        # calculate performance of AI instance w.r.t. AI and store results
        scores_ai = []
        for i, score_ai in enumerate(matched_conjugate_scores_ai):
            other_scores = [score_reader]+matched_conjugate_scores_ai[:i] + matched_conjugate_scores_ai[i+1:]
            scores_ai.append(calc_order_auc_score(other_scores, [score_ai]))
        all_scores_ai.append(scores_ai)

    # collect performance metrics for AI (rank of conjugate performance w.r.t. average AI, averaged across readers)
    scores_ai = np.mean(all_scores_ai, axis=0)

    # perform bootstrapping, see https://github.com/mateuszbuda/ml-stat-util
    print(f"""
    Evaluating:
        {np.mean(scores_readers):.4f} ± {np.std(scores_readers):.4f} for panel of {len(scores_readers)} readers
        {np.mean(scores_ai):.4f} ± {np.std(scores_ai):.4f} for {len(scores_ai)} AI instances
        (with {iterations} iterations)""") if verbose else None

    p = perform_permutation_test(
        scores_baseline=list(scores_readers),
        scores_alternative=list(scores_ai),
        iterations=iterations,
    )

    if verbose:
        print(f"Probability for Performance(Panel of readers) ≥ Performance(AI): p = {p:.4e}")

    return float(p)
