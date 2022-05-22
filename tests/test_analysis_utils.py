import json

import numpy as np
from numpy.testing import assert_allclose
from picai_eval.analysis_utils import (calculate_pAUC_from_graph,
                                       calculate_statistics)


def test_calculate_statistics():
    """
    Unit Test: Calculating operating points
    """

    with open("tests/test-metrics.json") as fp:
        # read metrics with the following structure:
        # {
        #   'tpr': [[y1m1, y2m1, ...], [y1m2, y2m2, ...], ...]
        #   'fpr': [[x1m1, x2m1, ...], [x1m2, x2m2, ...], ...]
        # }
        metrics = json.load(fp)

    # restructure metrics
    metrics = list(zip(metrics['tpr'], metrics['fpr']))

    # calculate statistics
    op_match = {'y': 0.9166666666666666, 'x': 0.2205882353}  # x: 1-0.7794117647058824
    results = calculate_statistics(metrics=metrics, op_match=op_match)

    # Unit tests:
    assert_allclose(results['auroc_mean'], 0.8976800789760347), "Test 1 (AUROC) for calculating statistics failed!"
    assert_allclose(results['op_closest_y_y'], 0.9180555555555557), "Test 2 (operating point, sensitivity) for calculating statistics failed!"
    assert_allclose(results['op_closest_y_x'], 0.35294117647058826), "Test 3 (operating point, specificity) for calculating statistics failed!"  # 1 - 0.647
    assert_allclose(results['op_closest_y_y_std'], 0.0025983731852596724), "Test 4 (operating point, sensitivity std.) for calculating statistics failed!"
    assert_allclose(results['op_closest_y_x_std'], 0.06562039423796671), "Test 5 (operating point, specificity std.) for calculating statistics failed!"


def test_calculate_pAUC_from_graph():
    """
    Test normal operation for calculating the area under a curve
    """
    # test calculation of area under the curve
    # note: a typical ROC/FROC curve consists only of straight horizontal/vertical lines
    y = [0.00, 0.13, 0.13, 0.15, 0.15, 0.17, 0.17, 0.17, 0.50, 0.50, 0.89, 0.89, 1.00, 1.00]
    x = [0.00, 0.00, 0.10, 0.10, 0.20, 0.20, 0.30, 0.40, 0.40, 0.50, 0.50, 0.60, 0.60, 1.00]
    pAUC = calculate_pAUC_from_graph(
        x=x, y=y, pAUC_start=0.0, pAUC_end=1.0,
    )[0]
    assert_allclose(pAUC, (0.13 + 0.15 + 2*0.17 + 0.50 + 0.89 + 4*1.00) / 10)

    # test calculation of partial area under the curve
    y = [0.00, 0.13, 0.13, 0.15, 0.15, 0.17, 0.17, 0.17, 0.50, 0.50, 0.89, 0.89, 1.00, 1.00]
    x = [0.00, 0.00, 0.10, 0.10, 0.20, 0.20, 0.30, 0.40, 0.40, 0.50, 0.50, 0.60, 0.60, 1.00]
    pAUC = calculate_pAUC_from_graph(
        x=x, y=y, pAUC_start=0.20, pAUC_end=1.0,
    )[0]
    assert_allclose(pAUC, (2*0.17 + 0.50 + 0.89 + 4*1.00) / 10)

    # test partial AUC
    y = np.linspace(0, 1, num=101)
    x = np.linspace(0, 1, num=101)  # <- need value at 0.1 for pAUC to be exactly as calculated
    pAUC = calculate_pAUC_from_graph(
        x=x, y=y, pAUC_start=0.1, pAUC_end=1.0,
    )[0]
    assert_allclose(pAUC, 1/2 - 0.1*0.1/2)


def test_calculate_pAUC_from_graph_edge_cases():
    """
    Test edge cases for calculating the area under a curve
    """
    # ensure a warning is displayed when calculating statistics for a non-monotonically increasing curve
    # note: if the warning messages are not displayed, the coverage should not be 100%!
    y = [0, 1, 2, 3, 4, 3, 2, 3, 4, 5]
    x = [0, 1, 5, 6, 4, 3, 2, 3, 4, 5]
    calculate_pAUC_from_graph(
        x=x, y=y, pAUC_start=0.1, pAUC_end=3.1415,
    )

    # test unpacking of x and y
    y = [[np.linspace(0, 1)]]
    x = [[np.linspace(0, 1)]]
    pAUC = calculate_pAUC_from_graph(
        x=x, y=y, pAUC_start=0., pAUC_end=1.0,
    )[0]
    assert_allclose(pAUC, 1/2)

    # test return full = True
    y = np.linspace(0, 1)
    x = np.linspace(0, 1)
    results = calculate_pAUC_from_graph(
        x=x, y=y, pAUC_start=0., pAUC_end=1.0, full=True,
    )
    assert_allclose(results['auroc_all'], [1/2])
