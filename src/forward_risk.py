"""Self-built ex-ante fundamental risk attribution; no strategy rule changes."""

from pathlib import Path
import os

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
import argparse, hashlib, json
import numpy as np, pandas as pd
from .forward_risk_core import (
    trailing,
    ew_cov,
    specific_variance,
    core_risk,
    empirical_supplement,
)
from .forward_risk_exposures import STYLES, fit_factor_returns, style_matrix

REPO = Path(__file__).resolve().parents[1]


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def exposure_at(a, index, membership):
    cap = a["cap"][index]
    raw = a["descriptors"][index]
    sector = a["industry"][index]
    base = (
        np.isfinite(cap)
        & (cap > 0)
        & np.isfinite(raw).all(axis=1)
        & (sector != "UNKNOWN")
    )
    fit = base & membership
    if fit.sum() < 100:
        raise ValueError("Insufficient dated estimation members")
    labels = np.unique(sector[fit])
    valid = base & np.isin(sector, labels)
    ids = np.flatnonzero(valid)
    transformed = style_matrix(
        {name: raw[ids, j] for j, name in enumerate(STYLES[2:])}, cap[ids], fit[ids]
    )
    cols = ["Country", *STYLES, *["Industry_" + s for s in labels]]
    x = np.column_stack(
        [np.ones(len(ids)), transformed, sector[ids, None] == labels[None, :]]
    ).astype(float)
    return ids, x, cols, fit[ids]


def warmup(a, model, membership, out):
    path = out / "warmup_2015.npz"
    dates = pd.DatetimeIndex(a["dates"])
    codes = list(a["codes"])
    cols = list(model["columns"])
    hist = pd.read_csv(
        membership,
        sep="\t",
        header=None,
        names=["code", "start", "end"],
        parse_dates=["start", "end"],
    )
    intervals = {c: g for c, g in hist.groupby("code")}
    slots = np.flatnonzero((dates >= "2015-01-01") & (dates <= "2015-12-31"))
    f = np.full((len(slots), len(cols)), np.nan)
    u = np.full((len(slots), len(codes)), np.nan)
    audit = []
    for i, t in enumerate(slots):
        asof = dates[t - 1]
        member = np.zeros(len(codes), bool)
        for j, c in enumerate(codes):
            if c in intervals:
                member[j] = (
                    (intervals[c].start <= asof) & (intervals[c].end >= asof)
                ).any()
        try:
            ids, x, names, fit = exposure_at(a, t - 1, member)
            r = a["close"][t] / a["close"][t - 1] - 1
            usable = fit & np.isfinite(r[ids])
            ci = pd.Index(np.array(codes)[ids[usable]])
            fitted = fit_factor_returns(
                pd.DataFrame(x[usable], index=ci, columns=names),
                pd.Series(r[ids[usable]], index=ci),
                pd.Series(a["cap"][t - 1, ids[usable]], index=ci),
            )
            coeff = fitted["factor_returns"].to_numpy()
            dest = [cols.index(c) for c in names]
            f[i, dest] = coeff
            good = np.isfinite(r[ids])
            u[i, ids[good]] = r[ids[good]] - x[good] @ coeff
            audit.append(
                {
                    "date": dates[t],
                    "exposure_date": asof,
                    "fitted": True,
                    "members": int(usable.sum()),
                }
            )
        except (ValueError, ArithmeticError) as exc:
            audit.append(
                {
                    "date": dates[t],
                    "exposure_date": asof,
                    "fitted": False,
                    "reason": str(exc),
                }
            )
    pd.DataFrame(audit).to_csv(out / "warmup_fit_audit.csv", index=False)
    np.savez_compressed(
        path, dates=dates[slots].to_numpy(), factor_returns=f, specific_returns=u
    )
    print("WARMUP_COMPLETE", len(slots), sum(v["fitted"] for v in audit), flush=True)
    return load(path)


def factor_cov_at(history, asof, names):
    h = trailing(history, asof, 252)
    fixed = ["Country", *STYLES]
    industries = [c for c in names if c.startswith("Industry_")]
    excluded = []
    while True:
        required = fixed + industries
        good = h[required].notna().all(axis=1)
        if good.sum() >= 126:
            break
        if not industries:
            raise ValueError("Fewer than126 common-factor history days")
        worst = min(industries, key=lambda c: int(h[c].notna().sum()))
        industries.remove(worst)
        excluded.append(worst)
    keep = h.loc[good, required]
    positions = np.flatnonzero(good.to_numpy())
    ages = len(h) - 1 - positions
    f = ew_cov(keep.to_numpy(), ages, half_life=90, shrink=0.1)
    return (
        required,
        f,
        excluded,
        {
            "factor_history_start": str(keep.index[0].date()),
            "factor_history_end": str(keep.index[-1].date()),
            "factor_observations": len(keep),
        },
    )


def empirical_block(core, returns, asof, weights_core, weights_missing):
    missing = np.flatnonzero(abs(weights_missing) > 1e-12)
    if not len(missing):
        result = empirical_supplement(
            core, weights_core, np.array([]), np.zeros((0, 0)), 0.0
        )
        return (
            result,
            np.zeros((0, 0)),
            {
                "unknown_history_observations": 0,
                "rho_empirical": 0.0,
                "missing_ids": missing,
            },
        )
    coreids = np.flatnonzero(abs(weights_core) > 1e-12)
    h = trailing(returns, asof, 252)
    both = np.union1d(coreids, missing)
    complete = h.iloc[:, both].notna().all(axis=1)
    if complete.sum() < 126:
        raise ValueError(
            "Insufficient common raw-return history for missing-data supplement"
        )
    z = h.loc[complete].to_numpy()
    ages = len(h) - 1 - np.flatnonzero(complete)
    sigma = ew_cov(z[:, missing], ages, 90, 0.1)
    cm = z[:, coreids] @ weights_core[coreids]
    um = z[:, missing] @ weights_missing[missing]
    agg = ew_cov(np.column_stack([cm, um]), ages, 90, 0.0)
    empirical = float(agg[0, 1] / np.sqrt(agg[0, 0] * agg[1, 1]))
    rho = float(np.clip(empirical, -1, 1) * 0.9)
    result = empirical_supplement(
        core, weights_core, weights_missing[missing], sigma, rho
    )
    return (
        result,
        sigma,
        {
            "unknown_history_observations": int(complete.sum()),
            "rho_empirical": empirical,
            "missing_ids": missing,
        },
    )


def main(data_root, membership, out):
    source = data_root / "approx_barra_monthly_20261105"
    out.mkdir(parents=True, exist_ok=True)
    inputs = [
        source / "inputs_v2_paired_beta.npz",
        source / "fitted_model_v2_paired_beta.npz",
        data_root
        / "monthly_weight_reconstruction/estimated_close_weights_NOT_OBSERVED.parquet",
        REPO / "results/net_edge_improvement/account/holdings.parquet",
        REPO / "results/net_edge_improvement/account/daily_report.csv",
        REPO / "data/frozen_forecast.parquet",
        membership,
        Path(__file__),
        Path(__file__).with_name("forward_risk_core.py"),
        Path(__file__).with_name("forward_risk_exposures.py"),
    ]

    def sha256(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    identity = {path: sha256(path) for path in inputs}
    (out / "input_hashes.json").write_text(
        json.dumps({path.name: digest for path, digest in identity.items()}, indent=2),
        encoding="utf-8",
    )
    a = load(inputs[0])
    model = load(inputs[1])
    w = warmup(a, model, membership, out)
    dates = pd.DatetimeIndex(a["dates"])
    codes = list(a["codes"])
    factor_names = list(model["columns"])
    fd = pd.DatetimeIndex(np.r_[w["dates"], model["dates"]])
    fret = pd.DataFrame(
        np.concatenate([w["factor_returns"], model["factor_return"]]),
        index=fd,
        columns=factor_names,
    )
    specific = pd.DataFrame(
        np.concatenate([w["specific_returns"], model["specific"]]),
        index=fd,
        columns=codes,
    )
    fret.to_parquet(out / "factor_history_with_warmup.parquet")
    specific.to_parquet(out / "specific_history_with_warmup.parquet")
    returns = pd.DataFrame(a["close"], index=dates, columns=codes).pct_change(
        fill_method=None
    )
    bw = pd.read_parquet(inputs[2]).reindex(columns=codes)
    holdings = pd.read_parquet(inputs[3])
    holdings.date = pd.to_datetime(holdings.date)
    pw = (
        holdings.pivot(index="date", columns="instrument", values="weight")
        .reindex(index=bw.index[1:], columns=codes)
        .fillna(0.0)
    )
    records = []
    factor_records = []
    stock_records = []
    last_snapshot = None
    last_arrays = None
    for i, asof in enumerate(pw.index):
        t = dates.get_loc(asof)
        wp = pw.loc[asof].to_numpy()
        wb = bw.loc[asof].to_numpy()
        active = wp - wb
        row = {
            "asof": asof,
            "exposure_date": asof,
            "model": "self-built10style+historical industries",
            "benchmark": "monthly-reconstructed CSI300 holdings",
            "status": "unavailable",
        }
        try:
            ids, x, names, _ = exposure_at(a, t, wb > 0)
            allowed, f, excluded, meta = factor_cov_at(fret, asof, names)
            row.update(meta)
            d, counts = specific_variance(specific, a["industry"][t], asof)
            okay = np.isfinite(d.iloc[ids].to_numpy())
            for bad in excluded:
                okay &= a["industry"][t, ids] != bad.removeprefix("Industry_")
            coreids = ids[okay]
            dest = [names.index(c) for c in allowed]
            xcore = x[okay][:, dest]
            diagonal = d.iloc[coreids].to_numpy()
            ac = active[coreids]
            core = core_risk(xcore, f, diagonal, ac)
            core_weight = np.zeros(len(codes))
            core_weight[coreids] = active[coreids]
            unknown_weight = active - core_weight
            missing_mask = np.ones(len(codes), bool)
            missing_mask[coreids] = False
            row.update(
                status="core_available",
                core_predicted_TE=float(np.sqrt(core["variance"] * 252)),
                core_factor_variance=float(core["factor"].sum()),
                core_specific_variance=float(core["specific"].sum()),
                core_stock_count=len(coreids),
                minimum_specific_observations=int(counts[coreids].min()),
                unknown_portfolio_weight=float(wp[missing_mask].sum()),
                unknown_benchmark_weight=float(wb[missing_mask].sum()),
                absolute_unknown_active_weight=float(abs(unknown_weight).sum()),
                excluded_industries=";".join(excluded),
                covariance_min_eigenvalue=float(np.linalg.eigvalsh(f).min()),
            )
            snapshot = pd.DataFrame(f, index=allowed, columns=allowed)
            supplement = None
            unknown_cov = None
            sm = None
            try:
                supplement, unknown_cov, sm = empirical_block(
                    core, returns, asof, core_weight, unknown_weight
                )
                row.update(
                    status="core_and_empirical_supplement",
                    hybrid_predicted_TE=float(np.sqrt(supplement["variance"] * 252)),
                    unknown_block_correlation=supplement["correlation"],
                    unknown_block_predicted_sigma=float(
                        np.sqrt(supplement["unknown_variance"] * 252)
                    ),
                    correlation_uncertainty_lower=float(
                        supplement["volatility_lower_bound"] * np.sqrt(252)
                    ),
                    correlation_uncertainty_upper=float(
                        supplement["volatility_upper_bound"] * np.sqrt(252)
                    ),
                    unknown_history_observations=sm["unknown_history_observations"],
                )
            except ValueError as exc:
                row["supplement_status"] = str(exc)
            core_sigma = np.sqrt(core["variance"])
            sigma = np.sqrt(supplement["variance"]) if supplement else np.nan
            for j, name in enumerate(allowed):
                factor_records.append(
                    {
                        "asof": asof,
                        "factor": name,
                        "active_exposure": core["exposure"][j],
                        "core_variance_contribution": core["factor"][j],
                        "core_TE_contribution": core["factor"][j]
                        / core_sigma
                        * np.sqrt(252),
                        "hybrid_variance_contribution": supplement["factor"][j]
                        if supplement
                        else np.nan,
                        "hybrid_TE_contribution": supplement["factor"][j]
                        / sigma
                        * np.sqrt(252)
                        if supplement
                        else np.nan,
                    }
                )
            for j, code_id in enumerate(coreids):
                if abs(active[code_id]) < 1e-12:
                    continue
                stock_records.append(
                    {
                        "asof": asof,
                        "instrument": codes[code_id],
                        "block": "fundamental",
                        "active_weight": active[code_id],
                        "specific_variance": diagonal[j],
                        "core_specific_variance_contribution": core["specific"][j],
                        "core_total_variance_contribution": core["stock"][j],
                        "hybrid_total_variance_contribution": supplement["core_stock"][
                            j
                        ]
                        if supplement
                        else np.nan,
                    }
                )
            if supplement:
                for j, code_id in enumerate(sm["missing_ids"]):
                    stock_records.append(
                        {
                            "asof": asof,
                            "instrument": codes[code_id],
                            "block": "empirical_missing_data",
                            "active_weight": active[code_id],
                            "specific_variance": np.nan,
                            "core_specific_variance_contribution": 0.0,
                            "core_total_variance_contribution": 0.0,
                            "hybrid_total_variance_contribution": supplement[
                                "unknown_stock"
                            ][j],
                        }
                    )
            row["core_factor_TE_contribution"] = float(
                core["factor"].sum() / core_sigma * np.sqrt(252)
            )
            row["core_specific_TE_contribution"] = float(
                core["specific"].sum() / core_sigma * np.sqrt(252)
            )
            if supplement:
                row["hybrid_common_TE_contribution"] = float(
                    supplement["factor"].sum() / sigma * np.sqrt(252)
                )
                row["hybrid_specific_TE_contribution"] = float(
                    supplement["specific"].sum() / sigma * np.sqrt(252)
                )
                row["hybrid_missing_data_TE_contribution"] = float(
                    supplement["unknown_variance_contribution"] / sigma * np.sqrt(252)
                )
                row["hybrid_variance_reconciliation_error"] = float(
                    abs(
                        supplement["core_stock"].sum()
                        + supplement["unknown_stock"].sum()
                        - supplement["variance"]
                    )
                )
            # Save a model snapshot, not a future return. All inputs end at asof.
            last_snapshot = (
                asof,
                snapshot,
                pd.DataFrame(xcore, index=np.array(codes)[coreids], columns=allowed),
                pd.DataFrame(
                    {"specific_variance": diagonal, "active_weight": ac},
                    index=np.array(codes)[coreids],
                ),
            )
            last_arrays = {
                "coreids": coreids,
                "active_core": ac,
                "x": xcore,
                "f": f,
                "d": diagonal,
                "active_missing": unknown_weight,
                "core": core,
                "supplement": supplement,
                "unknown_cov": unknown_cov,
                "supplement_meta": sm,
            }
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            row["reason"] = str(exc)
        records.append(row)
        if (i + 1) % 250 == 0:
            print("EX_ANTE_RISK", i + 1, len(pw), flush=True)
    daily = pd.DataFrame(records).set_index("asof")
    daily.to_csv(out / "daily_predicted_risk.csv")
    fr = pd.DataFrame(factor_records)
    fr.to_parquet(out / "daily_factor_risk_contributions.parquet", index=False)
    sr = pd.DataFrame(stock_records)
    sr.to_parquet(out / "daily_stock_risk_contributions.parquet", index=False)
    if last_snapshot:
        asof, f, x, d = last_snapshot
        f.to_csv(out / "last_factor_covariance_daily.csv")
        x.to_csv(out / "last_stock_factor_exposures.csv")
        d.to_csv(out / "last_specific_variance_and_active_weights.csv")
        z = last_arrays
        cm = z["x"] @ z["f"] @ z["x"].T + np.diag(z["d"])
        active = z["active_core"]
        core_quad = float(active @ cm @ active)
        np.testing.assert_allclose(core_quad, z["core"]["variance"], atol=1e-12)
        snapshot_check = {
            "asof": str(asof.date()),
            "core_covariance_min_eigenvalue": float(np.linalg.eigvalsh(cm).min()),
            "core_full_stock_covariance_quadratic_form": core_quad,
            "core_component_variance": z["core"]["variance"],
        }
        if z["supplement"] is not None:
            ids = z["supplement_meta"]["missing_ids"]
            au = z["active_missing"][ids]
            cu = z["unknown_cov"]
            vu = z["supplement"]["unknown_variance"]
            if len(ids) and vu > 0:
                cross = (
                    z["supplement"]["correlation"]
                    * np.outer(cm @ active, cu @ au)
                    / np.sqrt(core_quad * vu)
                )
                full = np.block([[cm, cross], [cross.T, cu]])
                aa = np.r_[active, au]
            else:
                full = cm
                aa = active
            quad = float(aa @ full @ aa)
            np.testing.assert_allclose(quad, z["supplement"]["variance"], atol=1e-12)
            snapshot_check.update(
                hybrid_full_covariance_min_eigenvalue=float(
                    np.linalg.eigvalsh(full).min()
                ),
                hybrid_quadratic_form=quad,
                hybrid_component_variance=z["supplement"]["variance"],
            )
            if np.linalg.eigvalsh(full).min() < -1e-10:
                raise ValueError("Hybrid full covariance is not PSD")
        (out / "last_snapshot_matrix_checks.json").write_text(
            json.dumps(snapshot_check, indent=2), encoding="utf-8"
        )
    # Forward validation is a separate pass. No future returns enter F/D/weights.
    validation = []
    for asof, row in daily.iterrows():
        t = dates.get_loc(asof)
        if t + 1 >= len(dates) or row.status == "unavailable":
            continue
        next_day = dates[t + 1]
        active = pw.loc[asof].to_numpy() - bw.loc[asof].to_numpy()
        rr = returns.loc[next_day].to_numpy()
        required = abs(active) > 1e-12
        if not np.isfinite(rr[required]).all():
            continue
        realized = float(active[required] @ rr[required])
        te = row.get("hybrid_predicted_TE", np.nan)
        if not np.isfinite(te):
            continue
        validation.append(
            {
                "asof": asof,
                "validation_date": next_day,
                "predicted_TE": te,
                "frozen_inventory_next_active_return": realized,
                "standardized_next_return": realized / (te / np.sqrt(252)),
            }
        )
    valid = pd.DataFrame(validation)
    valid.to_csv(out / "forward_frozen_inventory_validation.csv", index=False)
    for p, h in identity.items():
        if sha256(p) != h:
            raise ValueError("Input changed during risk calculation")
    summary = {
        "asof_dates": len(daily),
        "statuses": daily.status.value_counts().to_dict(),
        "mean_core_TE": float(daily.core_predicted_TE.mean()),
        "mean_hybrid_TE": float(daily.hybrid_predicted_TE.mean()),
        "mean_unknown_portfolio_weight": float(daily.unknown_portfolio_weight.mean()),
        "max_unknown_portfolio_weight": float(daily.unknown_portfolio_weight.max()),
        "last_date": str(daily.index[-1].date()),
        "last_core_TE": float(daily.core_predicted_TE.iloc[-1]),
        "last_hybrid_TE": float(daily.hybrid_predicted_TE.iloc[-1]),
        "validation_days": len(valid),
        "standardized_forward_return_std": float(
            valid.standardized_next_return.std(ddof=1)
        ),
        "abs_z_over_1_96_fraction": float(
            valid.standardized_next_return.abs().gt(1.96).mean()
        ),
        "max_variance_sum_error": float(
            daily.hybrid_variance_reconciliation_error.max()
        ),
        "official_Barra": False,
        "original_strategy_inputs_unchanged": True,
        "risk_target": "frozen close inventory relative to estimated benchmark holdings, before future trading/costs",
        "missing_data_block": "separate empirical covariance with shrunk past correlation, NOT fundamental Barra",
        "F_policy": "trailing252, minimum126 complete factor rows, EW half-life90, diagonal shrink10%, PSD floor",
        "D_policy": "trailing252, minimum126 residuals, EW half-life60, 20% shrink to dated industry median",
        "risk_exposure_uses_next_day_return_filter": False,
        "forward_validation_used_for_fitting": False,
    }
    summary = {
        key: None if isinstance(value, float) and not np.isfinite(value) else value
        for key, value in summary.items()
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print("FORWARD_RISK_FINISHED", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the local risk inputs",
    )
    parser.add_argument(
        "--membership",
        type=Path,
        required=True,
        help="Historical Qlib CSI 300 membership TSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Separate directory for generated risk outputs",
    )
    args = parser.parse_args()
    if args.output_dir.resolve() == (REPO / "results/forward_risk").resolve():
        parser.error("Choose an output directory separate from the published results.")
    main(args.data_root, args.membership, args.output_dir)
