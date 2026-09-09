import numpy as np

from analysis.critical_boundary import fit_boundary


def test_boundary_handles_degenerate_and_transition_data():
    x = np.array([1, 2, 4, 8], dtype=float)
    assert fit_boundary(x, np.ones(4)) == float("inf")
    assert fit_boundary(x, np.zeros(4)) == float("-inf")
    estimate = fit_boundary(x, np.array([1, 1, 0, 0]))
    assert 1 < estimate < 8

