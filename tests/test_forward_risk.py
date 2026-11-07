import unittest
import numpy as np
import pandas as pd
from src.forward_risk_core import (
    trailing,
    ew_cov,
    specific_variance,
    core_risk,
    empirical_supplement,
)


class ForwardRiskTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(914)

    def test_future_append_does_not_change_covariance_or_specific_risk(self):
        dates = pd.bdate_range("2020-01-01", periods=360)
        history = pd.DataFrame(self.rng.normal(0, 0.02, (360, 4)), index=dates)
        asof = dates[259]
        corrupted = history.copy()
        corrupted.loc[corrupted.index > asof] = 1e10
        one = trailing(history, asof)
        two = trailing(corrupted, asof)
        ages = np.arange(len(one) - 1, -1, -1)
        np.testing.assert_array_equal(ew_cov(one, ages), ew_cov(two, ages))
        first, _ = specific_variance(history, ["A", "A", "B", "B"], asof)
        second, _ = specific_variance(corrupted, ["A", "A", "B", "B"], asof)
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(one.index.max(), asof)

    def test_stock_marginal_risk_matches_finite_difference(self):
        x = self.rng.normal(size=(9, 3))
        q = self.rng.normal(size=(3, 3))
        f = q @ q.T * 0.0001
        d = self.rng.uniform(0.0001, 0.001, 9)
        a = self.rng.normal(0, 0.02, 9)
        answer = core_risk(x, f, d, a)
        sigma = x @ f @ x.T + np.diag(d)
        self.assertAlmostEqual(answer["variance"], float(a @ sigma @ a), places=15)
        step = 1e-6
        gradient = np.array(
            [
                (
                    (a + np.eye(9)[i] * step) @ sigma @ (a + np.eye(9)[i] * step)
                    - (a - np.eye(9)[i] * step) @ sigma @ (a - np.eye(9)[i] * step)
                )
                / (2 * step)
                for i in range(9)
            ]
        )
        np.testing.assert_allclose(answer["stock"], 0.5 * a * gradient, atol=1e-13)
        self.assertAlmostEqual(
            answer["factor"].sum() + answer["specific"].sum(),
            answer["stock"].sum(),
            places=15,
        )

    def test_hybrid_matches_independent_psd_full_matrix(self):
        x = self.rng.normal(size=(7, 3))
        q = self.rng.normal(size=(3, 3))
        f = q @ q.T * 0.0001
        d = np.full(7, 0.0002)
        a = self.rng.normal(0, 0.03, 7)
        u = self.rng.normal(0, 0.02, 4)
        q = self.rng.normal(size=(4, 4))
        cu = q @ q.T * 0.0001
        cm = x @ f @ x.T + np.diag(d)
        core = core_risk(x, f, d, a)
        for rho in [-0.9, 0.0, 0.9]:
            result = empirical_supplement(core, a, u, cu, rho)
            cross = (
                rho * np.outer(cm @ a, cu @ u) / np.sqrt((a @ cm @ a) * (u @ cu @ u))
            )
            full = np.block([[cm, cross], [cross.T, cu]])
            active = np.r_[a, u]
            self.assertGreaterEqual(np.linalg.eigvalsh(full).min(), -1e-12)
            self.assertAlmostEqual(
                result["variance"], float(active @ full @ active), places=15
            )
            np.testing.assert_allclose(
                np.r_[result["core_stock"], result["unknown_stock"]],
                active * (full @ active),
                atol=1e-14,
            )

    def test_short_specific_history_is_missing_not_zero(self):
        dates = pd.bdate_range("2020-01-01", periods=180)
        history = pd.DataFrame(self.rng.normal(0, 0.01, (180, 3)), index=dates)
        history.iloc[:100, 1] = np.nan
        history.iloc[:, 2] = np.nan
        d, n = specific_variance(history, ["A", "A", "B"], dates[-1])
        self.assertTrue(np.isfinite(d.iloc[0]))
        self.assertTrue(d.iloc[1:].isna().all())
        np.testing.assert_array_equal(n, [180, 80, 0])

    def test_rejects_invalid_core_covariance_and_unknown_exposure(self):
        with self.assertRaises(ValueError):
            core_risk([[1]], [[-0.001]], [0.001], [0.1])
        with self.assertRaises(ValueError):
            core_risk([[np.nan]], [[0.001]], [0.001], [0.1])
        with self.assertRaises(ValueError):
            trailing(
                pd.DataFrame(index=pd.to_datetime(["2020-01-02", "2020-01-01"])),
                "2020-01-03",
            )

    def test_negative_factor_contributions_are_retained(self):
        answer = core_risk(
            np.eye(2),
            np.array([[1.0, -0.8], [-0.8, 1.0]]),
            np.array([0.01, 0.01]),
            np.array([1.0, 0.1]),
        )
        self.assertLess(answer["factor"][1], 0.0)
        self.assertAlmostEqual(answer["stock"].sum(), answer["variance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
