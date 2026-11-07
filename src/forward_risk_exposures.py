"""Style exposures and cross-sectional factor fit for the forward risk study."""

import numpy as np
import pandas as pd
from scipy.linalg import null_space


STYLES = [
    "Size",
    "NonlinearSize",
    "Beta",
    "Momentum",
    "ResidualVolatility",
    "Liquidity",
    "BookToPrice",
    "EarningsYield",
    "Growth",
    "Leverage",
]


def standardized(raw, cap, fit):
    ref = raw[fit]
    med = np.median(ref)
    mad = np.median(abs(ref - med))
    clipped = (
        np.clip(raw, med - 5 * 1.4826 * mad, med + 5 * 1.4826 * mad)
        if mad > 0
        else raw.copy()
    )
    w = cap[fit] / cap[fit].sum()
    mean = w @ clipped[fit]
    scale = np.sqrt(w @ ((clipped[fit] - mean) ** 2))
    if scale < 1e-12:
        raise ValueError("Constant descriptor in historical fit universe")
    return (clipped - mean) / scale


def residual(raw, controls, cap, fit):
    x = np.column_stack([np.ones(len(raw)), controls])
    sw = np.sqrt(cap[fit] / cap[fit].mean())
    coef = np.linalg.lstsq(x[fit] * sw[:, None], raw[fit] * sw, rcond=None)[0]
    return raw - x @ coef


def style_matrix(descriptors, cap, fit):
    """Fit transforms ONLY on the dated estimation members; apply to other names."""
    x = np.empty((len(cap), 10))
    x[:, 0] = standardized(np.log(cap), cap, fit)
    x[:, 1] = standardized(residual(x[:, 0] ** 3, x[:, [0]], cap, fit), cap, fit)
    for j, name in enumerate(STYLES[2:], 2):
        x[:, j] = standardized(descriptors[name], cap, fit)
    x[:, 4] = standardized(residual(x[:, 4], x[:, [0, 2]], cap, fit), cap, fit)
    return x


def fit_factor_returns(design, returns, capitalization):
    """Constrained WLS with a cap-weighted industry-return identification constraint.

    Objective sum(sqrt(cap_i) * residual_i**2), not ordinary cap-weighted OLS.
    Sum(industry_cap_share_j * industry_return_j) = 0 separates country effect.
    """
    y = returns.reindex(design.index).to_numpy(dtype=float)
    cap = capitalization.reindex(design.index).to_numpy(dtype=float)
    x = design.to_numpy(dtype=float)
    if not np.isfinite(np.r_[y, cap, x.ravel()]).all() or (cap <= 0).any():
        raise ValueError("Incomplete cross-sectional fit input.")
    industries = np.array([c.startswith("Industry_") for c in design.columns])
    if not industries.any() or "Country" not in design:
        raise ValueError("Country and industry factors required.")
    if not np.allclose(x[:, industries].sum(axis=1), 1.0):
        raise ValueError("Exactly one industry required per stock.")
    c = np.zeros(x.shape[1])
    c[industries] = (cap[:, None] * x[:, industries]).sum(axis=0) / cap.sum()
    z = null_space(c[None, :])
    root_weight = np.power(cap / cap.mean(), 0.25)
    reduced = (x @ z) * root_weight[:, None]
    if len(y) < reduced.shape[1] + 20:
        raise ValueError("Insufficient stocks for industry and ten-style regression.")
    if np.linalg.matrix_rank(reduced) != reduced.shape[1]:
        raise ValueError("Unidentified factor design.")
    coefficients = z @ np.linalg.lstsq(reduced, y * root_weight, rcond=None)[0]
    residual = y - x @ coefficients
    if abs(c @ coefficients) > 1e-10:
        raise ArithmeticError("Identification constraint failed.")
    # Reduced normal equations, plus constrained direction, independently checked.
    if np.max(abs(reduced.T @ (residual * root_weight))) > 1e-8:
        raise ArithmeticError("Constrained WLS normal equations failed.")
    return {
        "factor_returns": pd.Series(coefficients, index=design.columns),
        "specific_returns": pd.Series(residual, index=design.index),
        "industry_constraint": pd.Series(c, index=design.columns),
        "condition_number": float(np.linalg.cond(reduced)),
    }
