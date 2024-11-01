import numpy as np

from picai_eval import Metrics


def test_calculate_counts():
    """
    Test the lesion TPR and FPR function
    """
    lesion_results = {
        "0": [(0, 0, 0.)],
        "1": [(0, 0, 0.)],
        "2": [(1, 1, 0.)],
        "3": [(1, 0, 0.), (1, 0, 0.)],
        "4": [(0, 0, 0.)],
        "5": [(1, 0, 0.), (1, 0, 0.)],
    }
    metrics = Metrics(lesion_results=lesion_results)
    np.testing.assert_allclose(metrics.lesion_TPR, [0.2, 0.2])
    np.testing.assert_allclose(metrics.lesion_FPR, [0.0, np.inf])
    assert metrics.AP == 0.2


def test_calculate_counts_empty():
    """
    Test the lesion TPR and FPR function
    """
    lesion_results = {
        "0": [(0, 0, 0.)],
        "1": [(0, 0, 0.)],
        "2": [(1, 0, 0.)],
        "3": [(1, 0, 0.), (1, 0, 0.)],
        "4": [(0, 0, 0.)],
        "5": [(1, 0, 0.), (1, 0, 0.)],
    }
    metrics = Metrics(lesion_results=lesion_results)
    np.testing.assert_allclose(metrics.lesion_TPR, [0., 0.0])
    np.testing.assert_allclose(metrics.lesion_FPR, [0.0, np.inf])
    assert metrics.AP == 0.0


if __name__ == "__main__":
    test_calculate_counts()
    test_calculate_counts_empty()
