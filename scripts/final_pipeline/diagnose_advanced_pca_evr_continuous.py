from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "scripts" / "research_pca"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

import run_advanced_pca_strategy_comparison as research  # noqa: E402
import run_residual_comovement_penalized_pca as residual_pca  # noqa: E402


OUT_DIR = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "advanced_pca_v1_continuous_diagnostics"
FIG_DIR = PROJECT_ROOT / "reports" / "final_report" / "advanced_pca_v1" / "figures"
PCA_LAMBDA = 0.5


def log(message: str) -> None:
    print(f"[advanced_pca_continuous_evr] {message}", flush=True)


def orthonormalize(z_flat: np.ndarray, m: int, k: int) -> np.ndarray:
    z = z_flat.reshape(m, k)
    qr, r = np.linalg.qr(z)
    signs = np.sign(np.diag(r)[:k])
    signs[signs == 0] = 1.0
    return qr[:, :k] * signs


def canonical_a(m: int, k: int) -> np.ndarray:
    a = np.zeros((m, k))
    a[:k, :k] = np.eye(k)
    return a


def objective_for(y: np.ndarray, v_m: np.ndarray, lambda_penalty: float, m: int, k: int):
    def objective(z_flat: np.ndarray) -> float:
        a = orthonormalize(z_flat, m, k)
        q = v_m @ a
        residual = residual_pca.residual_from_loadings(y, q)
        rec = residual_pca.reconstruction_loss(y, residual)
        penalty = residual_pca.residual_corr_pc1_evr(residual)
        if not np.isfinite(rec) or not np.isfinite(penalty):
            return 1e6
        return float(rec + lambda_penalty * penalty)

    return objective


def optimize_with_previous(
    y: np.ndarray,
    basis: research.PCABasis,
    m: int,
    k: int,
    previous_a: np.ndarray | None,
    maxiter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    v_m = basis.eigvecs[:, :m]
    a0 = canonical_a(m, k)
    baseline_q = v_m @ a0
    baseline_metrics = research.factor_metrics(y, baseline_q, basis)
    baseline_objective = baseline_metrics["reconstruction_loss"] + PCA_LAMBDA * baseline_metrics["residual_pc1_evr_corr"]

    starts: list[np.ndarray] = []
    if previous_a is not None and previous_a.shape == (m, k):
        starts.append(previous_a + 1e-4 * rng.standard_normal((m, k)))
    starts.append(a0 + 1e-4 * rng.standard_normal((m, k)))

    objective = objective_for(y, v_m, PCA_LAMBDA, m, k)
    best_a = a0
    best_fun = float(baseline_objective)
    best_success = False
    best_nit = 0

    for start in starts:
        try:
            res = minimize(
                objective,
                start.ravel(),
                method="BFGS",
                options={"maxiter": maxiter, "gtol": 1e-5},
            )
        except Exception as exc:
            continue
        if np.isfinite(res.fun) and float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_a = orthonormalize(res.x, m, k)
            best_success = bool(res.success or float(res.fun) <= baseline_objective)
            best_nit = int(getattr(res, "nit", 0))

    return best_a, {
        "optimizer_success": best_success,
        "optimizer_objective_value": best_fun,
        "optimizer_n_iter": best_nit,
        "baseline_objective_value": float(baseline_objective),
    }


def compute_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = research.load_price_panel()
    universe, rank_cols = research.load_universe_panel()
    common_index = prices.index.intersection(universe.index)
    prices = prices.reindex(common_index)
    universe = universe.reindex(common_index)
    returns = np.log(prices.where(prices > 0)).diff()
    price_cols = set(prices.columns)

    base_start = research.WINDOW
    full_stop = len(common_index)
    start = base_start + args.start_offset
    if args.stop_offset is not None:
        stop = min(full_stop, base_start + args.stop_offset)
    else:
        stop = full_stop if args.max_hours is None else min(full_stop, base_start + args.max_hours)
    if stop <= start:
        raise ValueError(f"empty timestamp range: start={start}, stop={stop}")
    rows: list[dict[str, object]] = []
    skip_rows: list[dict[str, object]] = []
    previous_a: np.ndarray | None = None

    for count, pos in enumerate(range(start, stop), start=1):
        if count == 1 or count % args.log_every == 0:
            log(f"processed {count}/{stop - start}")
        ts = common_index[pos]
        candidates = research.active_universe_from_row(universe.loc[ts, rank_cols], price_cols)
        if len(candidates) < args.min_assets:
            skip_rows.append({"timestamp": ts, "skip_reason": "insufficient_candidates", "candidate_tickers": len(candidates), "eligible_tickers": 0})
            previous_a = None
            continue
        ret_window = returns.iloc[pos - research.WINDOW : pos][candidates]
        _y, eligible = research.standardize_window(ret_window)
        if len(eligible) < args.min_assets:
            skip_rows.append({"timestamp": ts, "skip_reason": "insufficient_complete_window", "candidate_tickers": len(candidates), "eligible_tickers": len(eligible)})
            previous_a = None
            continue
        current_ret = returns.iloc[pos][eligible]
        current_price = prices.iloc[pos][eligible]
        good_now = current_ret.notna() & current_price.gt(0)
        if int(good_now.sum()) < args.min_assets:
            skip_rows.append({"timestamp": ts, "skip_reason": "insufficient_current_valid", "candidate_tickers": len(candidates), "eligible_tickers": int(good_now.sum())})
            previous_a = None
            continue
        eligible_now = [c for c in eligible if bool(good_now.get(c, False))]
        ret_window_now = ret_window[eligible_now]
        y_fit, eligible_now_check = research.standardize_window(ret_window_now)
        if eligible_now_check != eligible_now:
            skip_rows.append({"timestamp": ts, "skip_reason": "eligible_recheck_mismatch", "candidate_tickers": len(candidates), "eligible_tickers": len(eligible_now)})
            previous_a = None
            continue
        basis = research.pca_basis(y_fit)
        m = min(args.m_components, len(eligible_now) - 1, len(basis.eigvals))
        if m < research.K_FACTORS:
            skip_rows.append({"timestamp": ts, "skip_reason": "insufficient_m_components", "candidate_tickers": len(candidates), "eligible_tickers": len(eligible_now)})
            previous_a = None
            continue

        rng = np.random.default_rng(args.seed + int(pos))
        prev = None if args.no_warm_start else previous_a
        a, opt = optimize_with_previous(y_fit, basis, m, research.K_FACTORS, prev, args.maxiter, rng)
        previous_a = None if args.no_warm_start else a
        q = basis.eigvecs[:, :m] @ a
        cov = np.cov(y_fit, rowvar=False)
        total_var = float(np.trace(cov))
        if total_var <= 0 or not np.isfinite(total_var):
            skip_rows.append({"timestamp": ts, "skip_reason": "invalid_total_variance", "candidate_tickers": len(candidates), "eligible_tickers": len(eligible_now)})
            previous_a = None
            continue
        evr = [float(q[:, i].T @ cov @ q[:, i] / total_var) for i in range(research.K_FACTORS)]
        rows.append(
            {
                "timestamp": ts,
                "method": "advanced_pca_v1_continuous_diagnostic",
                "lambda_pca_comovement": PCA_LAMBDA,
                "candidate_tickers": len(candidates),
                "eligible_tickers": len(eligible_now),
                "m_components": m,
                "PC1_explained_var_ratio": evr[0],
                "PC2_explained_var_ratio": evr[1],
                "PC3_explained_var_ratio": evr[2],
                "cumulative_explained_var_ratio_3": float(sum(evr)),
                **opt,
            }
        )

        if args.checkpoint_every and count % args.checkpoint_every == 0:
            pd.DataFrame(rows).to_csv(OUT_DIR / f"{args.output_prefix}_partial.csv", index=False)
            pd.DataFrame(skip_rows).to_csv(OUT_DIR / f"{args.output_prefix}_skips_partial.csv", index=False)

    return pd.DataFrame(rows), pd.DataFrame(skip_rows)


def plot_evr(diag: pd.DataFrame, figure_name: str) -> None:
    df = diag.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    hourly = pd.DataFrame({"timestamp": pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", tz="UTC")})
    df = hourly.merge(df, on="timestamp", how="left")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax2 = ax.twinx()
    pc1_line = ax.plot(df["timestamp"], df["PC1_explained_var_ratio"], label="Advanced PC1", linewidth=1.0, color="tab:blue")
    cum_line = ax.plot(df["timestamp"], df["cumulative_explained_var_ratio_3"], label="Advanced PC1-3 cumulative", linewidth=1.2, color="black")
    pc2_line = ax2.plot(df["timestamp"], df["PC2_explained_var_ratio"], label="Advanced PC2", linewidth=1.0, color="tab:orange")
    pc3_line = ax2.plot(df["timestamp"], df["PC3_explained_var_ratio"], label="Advanced PC3", linewidth=1.0, color="tab:green")
    ax.set_title("Advanced PCA explained variance ratio over time, PC1-PC3")
    ax.set_ylabel("PC1 and PC1-3 cumulative explained variance ratio")
    ax2.set_ylabel("PC2 and PC3 explained variance ratio")
    small = df[["PC2_explained_var_ratio", "PC3_explained_var_ratio"]].to_numpy(dtype=float)
    small = small[np.isfinite(small)]
    if small.size:
        lo = max(0.0, float(small.min()) * 0.85)
        hi = float(small.max()) * 1.15
        if hi > lo:
            ax2.set_ylim(lo, hi)
    lines = pc1_line + pc2_line + pc3_line + cum_line
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right", ncol=4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / figure_name, dpi=160)
    plt.close(fig)


def write_summary(diag: pd.DataFrame, skips: pd.DataFrame, figure_name: str) -> None:
    ts = pd.to_datetime(diag["timestamp"], utc=True).drop_duplicates().sort_values() if not diag.empty else pd.Series(dtype="datetime64[ns, UTC]")
    gaps = ts.diff().dropna() if len(ts) else pd.Series(dtype="timedelta64[ns]")
    expected = pd.date_range(ts.min(), ts.max(), freq="h", tz="UTC") if len(ts) else pd.DatetimeIndex([])
    summary = pd.DataFrame(
        [
            {
                "rows": len(diag),
                "skipped": len(skips),
                "unique_timestamps": len(ts),
                "expected_hourly_timestamps": len(expected),
                "missing_timestamps_inside_range": len(expected.difference(ts)) if len(ts) else 0,
                "gaps_gt_1h": int((gaps > pd.Timedelta(hours=1)).sum()) if len(gaps) else 0,
                "max_gap": str(gaps.max()) if len(gaps) else "",
                "figure": str(FIG_DIR / figure_name),
            }
        ]
    )
    summary.to_csv(OUT_DIR / "advanced_pca_v1_explained_variance_continuous_summary.csv", index=False)


def combine_chunks(args: argparse.Namespace) -> None:
    chunk_paths = sorted(
        path
        for path in OUT_DIR.glob("advanced_pca_v1_explained_variance_continuous_chunk_*.csv")
        if "_skips" not in path.name and "_partial" not in path.name
    )
    skip_paths = sorted(
        path
        for path in OUT_DIR.glob("advanced_pca_v1_explained_variance_continuous_chunk_*_skips.csv")
        if "_partial" not in path.name
    )
    if not chunk_paths:
        raise FileNotFoundError(f"no chunk CSVs found in {OUT_DIR}")
    diag = pd.concat([pd.read_csv(path) for path in chunk_paths], ignore_index=True)
    diag = diag.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    skip_frames = []
    for path in skip_paths:
        if path.stat().st_size > 2:
            skip_frames.append(pd.read_csv(path))
    skips = pd.concat(skip_frames, ignore_index=True) if skip_frames else pd.DataFrame()
    if not skips.empty:
        skips = skips.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    diag.to_csv(OUT_DIR / "advanced_pca_v1_explained_variance_continuous.csv", index=False)
    skips.to_csv(OUT_DIR / "advanced_pca_v1_explained_variance_continuous_skips.csv", index=False)
    plot_evr(diag, args.figure_name)
    write_summary(diag, skips, args.figure_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--max-hours", type=int, default=500)
    parser.add_argument("--min-assets", type=int, default=20)
    parser.add_argument("--m-components", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--stop-offset", type=int, default=None)
    parser.add_argument("--output-prefix", default="advanced_pca_v1_explained_variance_continuous")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--combine-chunks", action="store_true")
    parser.add_argument("--figure-name", default="advanced_pca_explained_variance_ratio_over_time_continuous_candidate.png")
    args = parser.parse_args()
    if args.full_run:
        args.max_hours = None
    return args


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if args.combine_chunks:
        combine_chunks(args)
        print(f"diagnostics: {OUT_DIR / 'advanced_pca_v1_explained_variance_continuous.csv'}")
        print(f"skips: {OUT_DIR / 'advanced_pca_v1_explained_variance_continuous_skips.csv'}")
        print(f"summary: {OUT_DIR / 'advanced_pca_v1_explained_variance_continuous_summary.csv'}")
        print(f"figure: {FIG_DIR / args.figure_name}")
        return
    diag, skips = compute_rows(args)
    diag_path = OUT_DIR / f"{args.output_prefix}.csv"
    skips_path = OUT_DIR / f"{args.output_prefix}_skips.csv"
    diag.to_csv(diag_path, index=False)
    skips.to_csv(skips_path, index=False)
    if not diag.empty:
        plot_evr(diag, args.figure_name)
    if args.output_prefix == "advanced_pca_v1_explained_variance_continuous":
        write_summary(diag, skips, args.figure_name)
    print(f"diagnostics: {diag_path}")
    print(f"skips: {skips_path}")
    print(f"summary: {OUT_DIR / 'advanced_pca_v1_explained_variance_continuous_summary.csv'}")
    print(f"figure: {FIG_DIR / args.figure_name}")


if __name__ == "__main__":
    main()
