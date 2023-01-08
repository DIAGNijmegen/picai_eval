from typing import List

import numpy as np
import pytest
from numpy.testing import assert_allclose

from picai_eval.statistical_helper import (calc_sensitivity, calc_specificity,
                                           match_reader, match_then_compare,
                                           multiple_permutation_tests,
                                           perform_matched_boostrapping)


def sample_predictions(n_pos=300, n_neg=700, kwargs_pos=None, kwargs_neg=None, seed=None):
    """Generate a set of predictions. With default parameters, these predictions have ~0.9 AUROC."""
    # set default values (which have ~0.9 AUC)
    if kwargs_pos is None:
        kwargs_pos = dict(a=3, b=2)
    if kwargs_neg is None:
        kwargs_neg = dict(a=2, b=6)

    # set random seed
    np.random.seed(seed)

    # sample predictions
    y_pred = np.concatenate((
        np.random.beta(size=n_pos, **kwargs_pos),
        np.random.beta(size=n_neg, **kwargs_neg),
    ))

    return y_pred


def sample_prediction_set(instances=10, n_pos=300, n_neg=700, kwargs_pos=None, kwargs_neg=None, seed=None):
    """Generate multiple sets of predictions and a set of labels."""
    # set random seed
    np.random.seed(seed)
    seeds = np.random.randint(int(1e5), size=instances)

    # create labels
    y_true = np.array([1] * n_pos + [0] * n_neg)

    y_pred = np.array([
        sample_predictions(
            n_pos=n_pos,
            n_neg=n_neg,
            kwargs_pos=kwargs_pos,
            kwargs_neg=kwargs_neg,
            seed=seed
        )
        for seed in seeds
    ])

    return y_true, y_pred


def test_multiple_permutation_tests_exact():
    # perform multiple permutation tests
    p_values = multiple_permutation_tests(
        all_scores={
            'a': {
                'scores_baseline': [0, 1, 2],
                'scores_alternative': [1, 2, 3]},
            'b': {
                'scores_baseline': [4, 1, 2],
                'scores_alternative': [1, 2, 3]
            },
            'c': {
                'scores_baseline': [0.92, 0.94, 0.95, 0.81, 0.82, 0.86],
                'scores_alternative': [0.96, 0.91, 0.90, 0.85, 0.81, 0.80]
            },
        }
    )

    # check p-values (are calculated exactly)
    for key in p_values:
        assert_allclose(p_values[key], {'a': 0.25, 'b': 0.7, 'c': 0.7218614718614719}[key])


# note: not all predictions set can be exactly matched, so always check for bias!
# the seed below, 1, generates predictions that can be matched.
@pytest.mark.parametrize("match", [
    'sensitivity',
    'specificity',
])
def test_operating_point_matching(match, iterations=1000, seed=576):
    # sample a prediction set, take one set of predictions as reader and another as AI predictions
    y_true, y_pred = sample_prediction_set(seed=seed)
    y_pred_reader = (y_pred[0] > 0.5).astype(int)
    y_pred_ai = y_pred[1]

    # sample indices, match performance and check performance difference (to discover bias in matching)
    np.random.seed(seed)
    z: List[float] = []
    for _ in range(iterations):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue

        y_true_it = y_true[indices]
        y_pred_ai_it = y_pred_ai[indices]
        y_pred_reader_it = y_pred_reader[indices]

        # match AI operating point to the reader
        y_pred_ai_matched_it = match_reader(
            y_true=y_true_it,
            y_pred_ai=y_pred_ai_it,
            y_pred_reader=y_pred_reader_it,
            match=match
        )

        # check performance difference between reader and matched AI
        if match == "sensitivity":
            performance_reader = calc_sensitivity(y_true=y_true_it, y_pred=y_pred_reader_it)
            performance_ai = calc_sensitivity(y_true=y_true_it, y_pred=y_pred_ai_matched_it)
        elif match == 'specificity':
            performance_reader = calc_specificity(y_true=y_true_it, y_pred=y_pred_reader_it)
            performance_ai = calc_specificity(y_true=y_true_it, y_pred=y_pred_ai_matched_it)
        else:
            raise NotImplementedError

        # collect performance difference
        z.append(performance_reader - performance_ai)

    # check if performance difference is zero for all subsets
    assert_allclose(np.mean(z), 0, atol=1e-3)


# The expected differences in Cohen's κ listed below are not based on manual calculations.
# Given that the two sets of predictions are sampled from the same distribution,
# the κ's should be close to each other (and therefore a difference close to zero).
@pytest.mark.parametrize("match,expected_z", [
    ('sensitivity', -0.025217179012471225),
    ('specificity', -0.025721235660373654),
])
def test_match_then_compare(match, expected_z, seed=576):
    # sample a prediction set, take one set of predictions as reader and another as AI predictions
    y_true, y_pred = sample_prediction_set(seed=seed)
    y_pred_reader = (y_pred[0] > 0.5).astype(int)
    y_pred_ai = y_pred[1]

    # perform matching and calculate difference in Cohen's κ
    z = match_then_compare(
        y_true=y_true,
        y_pred_ai=y_pred_ai,
        y_pred_reader=y_pred_reader,
        match=match,
    )

    # check if Cohen's κ difference (reader κ - AI κ) is as expected
    print("Observed z:", z)
    assert_allclose(z, expected_z)


# note: not all predictions set can be exactly matched, so always check for bias!
# the seed below, 1, generates predictions that can be matched.
def test_bootstrapping_sample_weights(match='sensitivity', flavour='match>sample>compare', seed=1):
    # sample a prediction set, take one set of predictions as reader and the others as AI predictions
    y_true, y_pred = sample_prediction_set(seed=seed)
    y_pred_reader = (y_pred[0] > 0.5).astype(int)
    y_pred_ai = y_pred[1:]

    kwargs = dict(
        y_true=y_true,
        y_pred_ai=y_pred_ai,
        y_pred_reader=y_pred_reader,
        match=match,
        flavour=flavour,
        iterations=100,
        seed=seed
    )

    # perform matched bootstrapping (twice)
    perform_matched_boostrapping(**kwargs)  # warmup. Should not be required, but it is.
    p1 = perform_matched_boostrapping(**kwargs)
    p2 = perform_matched_boostrapping(**kwargs)

    # perform matched bootstrapping with dummy sample weights
    p3 = perform_matched_boostrapping(
        sample_weight=np.random.uniform(size=len(y_true)),
        **kwargs
    )

    # compare p-values
    assert p1 == p2, "Random seed was not respected"
    assert p1 != p3, "Sample weight was not respected"
