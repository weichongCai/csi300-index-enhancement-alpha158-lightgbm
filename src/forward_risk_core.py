"""Past-only covariance, specific variance and Euler risk attribution.

Missing descriptors are not replaced by zero. A separately labelled empirical
block may supplement the fundamental block, using a PSD cross-block construction.
"""

import numpy as np
import pandas as pd


def trailing(frame, asof, window=252):
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Sorted unique history required")
    return frame.loc[frame.index <= pd.Timestamp(asof)].tail(window)


def ew_cov(values, ages, half_life=90, shrink=0.1):
    x = np.asarray(values, float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite complete covariance observations required")
    w = np.exp2(-np.asarray(ages, float) / half_life)
    w /= w.sum()
    centered = x - w @ x
    cov = (centered * w[:, None]).T @ centered / (1 - w @ w)
    cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
    ev, vec = np.linalg.eigh((cov + cov.T) / 2)
    floor = max(ev.max(), 1e-12) * 1e-8
    return (vec * np.maximum(ev, floor)) @ vec.T


def specific_variance(
    history, current_industry, asof, window=252, minimum=126, half_life=60, shrink=0.2
):
    data = trailing(history, asof, window)
    z = data.to_numpy(float)
    ok = np.isfinite(z)
    w = np.exp2(-np.arange(len(z) - 1, -1, -1) / half_life)[:, None] * ok
    mass = w.sum(axis=0)
    q = np.divide(w, mass, out=np.zeros_like(w), where=mass > 0)
    values = np.where(ok, z, 0.0)
    mean = (values * q).sum(axis=0)
    variance = ((np.where(ok, values - mean, 0.0) ** 2) * q).sum(axis=0) / np.maximum(
        1 - (q * q).sum(axis=0), 1e-12
    )
    ready = (ok.sum(axis=0) >= minimum) & np.isfinite(variance)
    result = pd.Series(
        np.where(ready, np.maximum(variance, 1e-12), np.nan), index=data.columns
    )
    groups = pd.Series(current_industry, index=data.columns)
    med = result.groupby(groups).transform("median")
    return (1 - shrink) * result + shrink * med, ok.sum(axis=0)


def core_risk(exposure, cov, specific, active):
    x = np.asarray(exposure, float)
    f = np.asarray(cov, float)
    d = np.asarray(specific, float)
    a = np.asarray(active, float)
    if not np.isfinite(np.r_[x.ravel(), f.ravel(), d, a]).all() or (d < 0).any():
        raise ValueError("Incomplete core risk inputs")
    if not np.allclose(f, f.T, atol=1e-12) or np.linalg.eigvalsh(f).min() < -1e-12:
        raise ValueError("Invalid factor covariance")
    b = x.T @ a
    fb = f @ b
    factor = b * fb
    specific_contribution = a * a * d
    stock = a * (x @ fb + a * d)
    var = float(factor.sum() + specific_contribution.sum())
    if var <= 0:
        raise ValueError("Core variance is not positive")
    np.testing.assert_allclose(stock.sum(), var, atol=1e-12)
    return {
        "variance": var,
        "factor": factor,
        "specific": specific_contribution,
        "stock": stock,
        "exposure": b,
    }


def empirical_supplement(core, active_core, active_missing, unknown_cov, rho):
    """PSD rank-one block linkage; expose this supplement as non-Barra.

    rho is estimated from past aggregate raw returns, then shrunk toward zero.
    Core/unknown stock variances are retained. This is an estimate, not an
    assumption that missing risk vanishes or its correlation is exactly zero.
    """
    if abs(rho) > 1:
        raise ValueError("Invalid block correlation")
    au = np.asarray(active_missing, float)
    vu = float(au @ unknown_cov @ au)
    vm = core["variance"]
    cross = float(rho * np.sqrt(vm * max(vu, 0)))
    total = vm + vu + 2 * cross
    if total <= 0:
        raise ValueError("Invalid combined variance")
    scale = 1 + cross / vm
    unknown_stock = (
        au * (unknown_cov @ au) * (1 + cross / vu) if vu > 0 else np.zeros(len(au))
    )
    result = {
        "variance": total,
        "factor": core["factor"] * scale,
        "specific": core["specific"] * scale,
        "core_stock": core["stock"] * scale,
        "unknown_stock": unknown_stock,
        "unknown_variance_contribution": float(unknown_stock.sum()),
        "unknown_variance": vu,
        "cross_covariance": cross,
        "correlation": rho,
        "volatility_lower_bound": abs(np.sqrt(vm) - np.sqrt(max(vu, 0))),
        "volatility_upper_bound": np.sqrt(vm) + np.sqrt(max(vu, 0)),
    }
    np.testing.assert_allclose(
        result["factor"].sum()
        + result["specific"].sum()
        + result["unknown_variance_contribution"],
        total,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result["core_stock"].sum() + result["unknown_stock"].sum(), total, atol=1e-12
    )
    return result
