import numpy as np

from picai_eval import Metrics


def test_lesion_tpr_at_fpr():
    """
    Test the lesion TPR at FPR function
    """
    lesion_results = {
        "0": [(0, 1, 1.)],
        "1": [(0, 2, 1.)],
        "2": [(1, 3, 1.)],
        "3": [(1, 1, 1.), (1, 1, 1.)],
        "4": [(0, 3, 1.)],
        "5": [(1, 2.5, 1.), (1, 1.5, 1.)],
    }
    metrics = Metrics(lesion_results=lesion_results)
    np.testing.assert_allclose(metrics.lesion_TPR, [0.2, 0.4, 0.4, 0.6, 0.6])
    np.testing.assert_allclose(metrics.lesion_FPR, [0.16666667, 0.16666667, 0.33333334, 0.33333334, np.inf])

    # test with FPR = 0.3
    np.testing.assert_almost_equal(metrics.lesion_TPR_at_FPR(0.3), 0.4)

    # test with too low FPR
    np.testing.assert_almost_equal(metrics.lesion_TPR_at_FPR(0.1), 0.0)


if __name__ == "__main__":
    test_lesion_tpr_at_fpr()
