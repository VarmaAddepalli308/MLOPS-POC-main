import numpy as np
from utils import calculate_returns, population_stability_index


def test_calculate_returns():
    values = [100.0, 110.0, 99.0]
    got = calculate_returns(values)
    assert np.isnan(got[0])
    assert np.isclose(got[1], 0.10)
    assert np.isclose(got[2], -0.10)


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(42)
    x = rng.normal(size=5000)
    assert population_stability_index(x, x) < 0.01


def test_psi_detects_shift():
    rng = np.random.default_rng(42)
    ref = rng.normal(size=5000)
    assert population_stability_index(ref, ref + 2.0) > 0.25
