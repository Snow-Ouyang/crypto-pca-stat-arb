from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "scripts" / "research_pca"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

import run_advanced_pca_strategy_comparison as research  # noqa: E402


OUT_DIR = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "advanced_pca_v1"
REPORT_DIR = PROJECT_ROOT / "reports" / "final_report" / "advanced_pca_v1"
FIG_DIR = REPORT_DIR / "figures"

BASELINE_EQUITY = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "final_strategy" / "soft_pc1_refined_hourly_equity.csv"

BASELINE_CONFIG = "soft_pc1_z_lambda_100_gross_cap_2p5"
ADVANCED_CONFIG = "advanced_pca_lambda0p5_soft_lambda2_gross_cap_2p5"

PCA_LAMBDA = 0.5
SOFT_LAMBDA = 2.0
GROSS_CAP = 2.5
FEE_BPS_LIST = [0, 5, 10]


def log(message: str) -> None:
    print(f"[advanced_pca_v1] {message}", flush=True)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def fixed_signal_thresholds() -> dict[tuple[str, str], tuple[float, float]]:
    buckets = ["hl_0_9", "hl_9_18", "hl_18_36", "hl_36_60", "hl_60_90"]
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for bucket in buckets:
        out[("long", bucket)] = (1.0, 0.5)
        out[("short", bucket)] = (1.0, 0.25)
    return out


def configure_research_module() -> None:
    research.OUT_DATA = OUT_DIR
    research.OUT_REPORT = REPORT_DIR
    research.METHODS = [("advanced_lambda_0p5", PCA_LAMBDA)]
    research.SOFT_PC1_LAMBDA = SOFT_LAMBDA
    research.GROSS_CAP = GROSS_CAP
    research.FEE_BPS_LIST = FEE_BPS_LIST
    research.signal_thresholds = fixed_signal_thresholds


def build_or_load_signal_panel(args: argparse.Namespace) -> pd.DataFrame:
    configure_research_module()
    residual_path = OUT_DIR / "advanced_residual_returns_long.csv"
    signal_path = OUT_DIR / "advanced_signal_panel.csv"
    if args.reuse and signal_path.exists():
        log(f"reusing signal panel: {signal_path}")
        return pd.read_csv(signal_path, parse_dates=["timestamp"], low_memory=False)
    if args.reuse and residual_path.exists():
        log(f"reusing residual panel: {residual_path}")
        residual = pd.read_csv(residual_path, parse_dates=["timestamp"], low_memory=False)
    else:
        log("building advanced PCA residual panel from raw price/universe inputs")
        residual = research.build_residual_panels(args)
    log("building fixed-threshold advanced signal panel")
    signal = research.build_signal_panel(residual)
    if signal.empty:
        raise RuntimeError("advanced signal panel is empty")
    return signal


def build_advanced_pca_diagnostics(args: argparse.Namespace) -> pd.DataFrame:
    configure_research_module()
    prices = research.load_price_panel()
    universe, rank_cols = research.load_universe_panel()
    common_index = prices.index.intersection(universe.index)
    prices = prices.reindex(common_index)
    universe = universe.reindex(common_index)
    returns = np.log(prices.where(prices > 0)).diff()
    price_cols = set(prices.columns)
    rng = np.random.default_rng(args.seed)

    start = research.WINDOW
    stop = len(common_index) if args.max_hours is None else min(len(common_index), research.WINDOW + args.max_hours)
    positions = list(range(start, stop))
    log(f"advanced PCA diagnostics timestamps={len(positions)} n_jobs={args.n_jobs}")
    results = Parallel(n_jobs=args.n_jobs, backend=args.parallel_backend, batch_size=args.batch_size, verbose=10)(
        delayed(compute_advanced_pca_diagnostic_timestamp)(
            pos,
            prices,
            returns,
            universe,
            rank_cols,
            price_cols,
            args,
        )
        for pos in positions
    )
    rows = [row for row, _reason in results if row is not None]
    skipped = sum(1 for row, _reason in results if row is None)
    skip_rows = [{"timestamp": common_index[pos], "skip_reason": reason} for pos, (_row, reason) in zip(positions, results) if _row is None]
    diag = pd.DataFrame(rows)
    diag.to_csv(OUT_DIR / "advanced_pca_v1_explained_variance.csv", index=False)
    pd.DataFrame(skip_rows).to_csv(OUT_DIR / "advanced_pca_v1_diagnostic_skips.csv", index=False)
    pd.DataFrame([{"skipped_pca_diagnostic_timestamps": skipped, "rows": len(diag)}]).to_csv(
        OUT_DIR / "advanced_pca_v1_diagnostics_summary.csv",
        index=False,
    )
    return diag


def compute_advanced_pca_diagnostic_timestamp(
    pos: int,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    rank_cols: list[str],
    price_cols: set[str],
    args: argparse.Namespace,
) -> tuple[dict[str, object] | None, str]:
    ts = prices.index[pos]
    candidates = research.active_universe_from_row(universe.loc[ts, rank_cols], price_cols)
    if len(candidates) < args.min_assets:
        return None, "insufficient_candidates"
    ret_window = returns.iloc[pos - research.WINDOW : pos][candidates]
    _y, eligible = research.standardize_window(ret_window)
    if len(eligible) < args.min_assets:
        return None, "insufficient_complete_window"
    current_ret = returns.iloc[pos][eligible]
    current_price = prices.iloc[pos][eligible]
    good_now = current_ret.notna() & current_price.gt(0)
    if int(good_now.sum()) < args.min_assets:
        return None, "insufficient_current_valid"
    eligible_now = [c for c in eligible if bool(good_now.get(c, False))]
    ret_window_now = ret_window[eligible_now]
    y_fit, eligible_now_check = research.standardize_window(ret_window_now)
    if eligible_now_check != eligible_now:
        return None, "eligible_recheck_mismatch"
    basis = research.pca_basis(y_fit)
    m = min(args.m_components, len(eligible_now) - 1, len(basis.eigvals))
    if m < research.K_FACTORS:
        return None, "insufficient_m_components"
    rng = np.random.default_rng(args.seed + int(pos))
    a, opt = research.optimize_advanced_pca(
        y=y_fit,
        basis=basis,
        m=m,
        k=research.K_FACTORS,
        lambda_penalty=PCA_LAMBDA,
        rng=rng,
        random_starts=args.random_starts,
        maxiter=args.maxiter,
    )
    q = basis.eigvecs[:, :m] @ a
    cov = np.cov(y_fit, rowvar=False)
    total_var = float(np.trace(cov))
    if total_var <= 0 or not np.isfinite(total_var):
        return None, "invalid_total_variance"
    evr = [float(q[:, i].T @ cov @ q[:, i] / total_var) for i in range(research.K_FACTORS)]
    return (
        {
            "timestamp": ts,
            "method": "advanced_pca_v1",
            "lambda_pca": PCA_LAMBDA,
            "eligible_tickers": len(eligible_now),
            "PC1_explained_var_ratio": evr[0],
            "PC2_explained_var_ratio": evr[1],
            "PC3_explained_var_ratio": evr[2],
            "cumulative_explained_var_ratio_3": float(sum(evr)),
            "optimizer_success": bool(opt.get("optimizer_success", False)),
            "optimizer_objective_value": opt.get("optimizer_objective_value", np.nan),
        },
        "",
    )


def plot_advanced_explained_variance(diag: pd.DataFrame) -> None:
    if diag.empty:
        return
    df = diag.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    hourly = pd.DataFrame({"timestamp": pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", tz="UTC")})
    df = hourly.merge(df, on="timestamp", how="left")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax2 = ax.twinx()
    pc1_line = ax.plot(df["timestamp"], df["PC1_explained_var_ratio"], label="Advanced PC1", linewidth=1.0, color="tab:blue")
    cum_line = ax.plot(
        df["timestamp"],
        df["cumulative_explained_var_ratio_3"],
        label="Advanced PC1-3 cumulative",
        linewidth=1.2,
        color="black",
    )
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
    fig.savefig(FIG_DIR / "advanced_pca_explained_variance_ratio_over_time.png", dpi=160)
    plt.close(fig)


def write_diagnostics_report(diag: pd.DataFrame, args: argparse.Namespace) -> None:
    if diag.empty:
        return
    numeric = diag[
        [
            "PC1_explained_var_ratio",
            "PC2_explained_var_ratio",
            "PC3_explained_var_ratio",
            "cumulative_explained_var_ratio_3",
        ]
    ].mean()
    lines = [
        "# Advanced PCA v1 Diagnostics",
        "",
        f"- PCA residual-comovement penalty: `{PCA_LAMBDA}`.",
        f"- Diagnostic rows: `{len(diag)}`.",
        f"- PCA max hours: `{args.max_hours if args.max_hours is not None else 'full'}`.",
        "",
        "## Mean Explained Variance",
        "",
        pd.DataFrame([numeric.to_dict()]).to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Figure",
        "",
        "![Advanced PCA explained variance](figures/advanced_pca_explained_variance_ratio_over_time.png)",
    ]
    (REPORT_DIR / "advanced_pca_v1_diagnostics_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_advanced_strategy(signal_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configure_research_module()
    all_equity = []
    all_positions = []
    for fee_bps in FEE_BPS_LIST:
        log(f"backtesting advanced soft sleeve fee={fee_bps}bps")
        equity, positions = research.run_sleeve(signal_panel, "soft_pc1_lambda2", fee_bps)
        all_equity.append(equity)
        all_positions.append(positions)
    equity_long = pd.concat(all_equity, ignore_index=True)
    positions = pd.concat([p for p in all_positions if not p.empty], ignore_index=True) if any(not p.empty for p in all_positions) else pd.DataFrame()
    summary = research.summarize(equity_long, positions)
    summary["strategy"] = "advanced_pca_v1"
    summary["config"] = ADVANCED_CONFIG
    summary["pca_lambda"] = PCA_LAMBDA
    summary["soft_lambda"] = SOFT_LAMBDA
    summary["long_entry"] = 1.0
    summary["long_exit"] = 0.5
    summary["short_entry"] = 1.0
    summary["short_exit"] = 0.25
    return equity_long, positions, summary


def to_final_hourly(equity_long: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "active_gross_exposure",
        "active_net_exposure",
        "active_pc1_exposure",
        "n_active_positions",
    ]
    wide = equity_long.pivot_table(index="timestamp", columns="fee_bps", values="net_equity", aggfunc="last").reset_index()
    wide = wide.rename(columns={fee: f"net_equity_{int(fee)}bps" for fee in wide.columns if fee != "timestamp"})
    first_fee = equity_long[equity_long["fee_bps"] == FEE_BPS_LIST[0]].copy()
    exposure_cols = [c for c in metric_cols if c in first_fee.columns]
    hourly = first_fee[["timestamp", *exposure_cols]].merge(
        wide[["timestamp", *[f"net_equity_{fee}bps" for fee in FEE_BPS_LIST]]],
        on="timestamp",
        how="left",
    )
    hourly.insert(1, "config", ADVANCED_CONFIG)
    hourly["gross_equity"] = hourly["net_equity_0bps"]
    return hourly.sort_values("timestamp")


def max_drawdown(series: pd.Series) -> float:
    return float((series - series.cummax()).min()) if len(series) else 0.0


def sharpe_like_hourly(pnl: pd.Series) -> float:
    std = pnl.std()
    return float(pnl.mean() / std * np.sqrt(24 * 365)) if std and np.isfinite(std) and std > 0 else np.nan


def summarize_existing_equity(path: Path, config: str, strategy: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    eq = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    eq = eq[eq["config"] == config].sort_values("timestamp")
    eq = eq[(eq["timestamp"] >= start_ts) & (eq["timestamp"] <= end_ts)].copy()
    if eq.empty:
        raise ValueError(f"No rows for {config} in {path} between {start_ts} and {end_ts}")
    rows = []
    for fee_bps in FEE_BPS_LIST:
        col = f"net_equity_{fee_bps}bps"
        pnl = eq[col].diff().fillna(eq[col])
        pc1_col = "active_pc1_exposure_used_beta" if "active_pc1_exposure_used_beta" in eq.columns else "active_pc1_exposure"
        rows.append(
            {
                "strategy": strategy,
                "config": config,
                "fee_bps": fee_bps,
                "final_net_equity": float(eq[col].iloc[-1]),
                "max_drawdown_net": max_drawdown(eq[col]),
                "sharpe_like_net": sharpe_like_hourly(pnl),
                "avg_active_gross_exposure": float(eq["active_gross_exposure"].mean()) if "active_gross_exposure" in eq else np.nan,
                "avg_abs_active_net_exposure": float(eq["active_net_exposure"].abs().mean()) if "active_net_exposure" in eq else np.nan,
                "avg_abs_active_pc1_exposure": float(eq[pc1_col].abs().mean()) if pc1_col in eq else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(comparison: pd.DataFrame, signal_panel: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Advanced PCA v1 Mainline Branch",
        "",
        f"- Advanced PCA residual-comovement penalty: `{PCA_LAMBDA}`.",
        f"- Soft factor constraint lambda: `{SOFT_LAMBDA}`.",
        f"- Filter: `0 < b < 1`, `half_life < 90h`.",
        "- Thresholds: long `entry=1 / exit=0.5`; short `entry=1 / exit=0.25`.",
        "- No sigma percentile or R2 entry filters are applied.",
        "",
        f"Signal rows: `{len(signal_panel)}`. Timestamp range: `{signal_panel['timestamp'].min()}` to `{signal_panel['timestamp'].max()}`.",
        f"PCA max hours: `{args.max_hours if args.max_hours is not None else 'full'}`.",
        "",
        "## Three-Line Summary",
        "",
        comparison.sort_values(["fee_bps", "strategy"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## PCA Diagnostics",
        "",
        "![Advanced PCA explained variance](figures/advanced_pca_explained_variance_ratio_over_time.png)",
    ]
    (REPORT_DIR / "advanced_pca_v1_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    if args.diagnostics_only:
        diag_path = OUT_DIR / "advanced_pca_v1_explained_variance.csv"
        if args.reuse and diag_path.exists():
            log(f"reusing advanced PCA diagnostics: {diag_path}")
            diagnostics = pd.read_csv(diag_path, parse_dates=["timestamp"], low_memory=False)
        else:
            diagnostics = build_advanced_pca_diagnostics(args)
        plot_advanced_explained_variance(diagnostics)
        write_diagnostics_report(diagnostics, args)
        print(f"diagnostics: {OUT_DIR / 'advanced_pca_v1_explained_variance.csv'}")
        print(f"figure: {FIG_DIR / 'advanced_pca_explained_variance_ratio_over_time.png'}")
        return

    signal_panel = build_or_load_signal_panel(args)
    diag_path = OUT_DIR / "advanced_pca_v1_explained_variance.csv"
    if args.reuse and diag_path.exists():
        log(f"reusing advanced PCA diagnostics: {diag_path}")
        diagnostics = pd.read_csv(diag_path, parse_dates=["timestamp"], low_memory=False)
    else:
        diagnostics = build_advanced_pca_diagnostics(args)
    plot_advanced_explained_variance(diagnostics)
    write_diagnostics_report(diagnostics, args)
    signal_panel = signal_panel[signal_panel["method"] == "advanced_lambda_0p5"].copy()
    equity_long, positions, advanced_summary = run_advanced_strategy(signal_panel)
    hourly = to_final_hourly(equity_long)
    hourly.to_csv(OUT_DIR / "advanced_pca_v1_hourly_equity.csv", index=False)
    positions.to_csv(OUT_DIR / "advanced_pca_v1_positions.csv", index=False)
    advanced_summary.to_csv(REPORT_DIR / "advanced_pca_v1_summary.csv", index=False)
    start_ts = pd.to_datetime(hourly["timestamp"].min())
    end_ts = pd.to_datetime(hourly["timestamp"].max())
    pd.DataFrame(
        [
            {
                "max_hours": args.max_hours if args.max_hours is not None else "full",
                "pca_lambda": PCA_LAMBDA,
                "soft_lambda": SOFT_LAMBDA,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "signal_rows": len(signal_panel),
                "diagnostic_rows": len(diagnostics),
            }
        ]
    ).to_csv(REPORT_DIR / "advanced_pca_v1_run_metadata.csv", index=False)

    comparison = pd.concat(
        [
            summarize_existing_equity(BASELINE_EQUITY, BASELINE_CONFIG, "baseline", start_ts, end_ts),
            advanced_summary[
                [
                    "strategy",
                    "config",
                    "fee_bps",
                    "final_net_equity",
                    "max_drawdown_net",
                    "sharpe_like_net",
                    "avg_active_gross_exposure",
                    "avg_abs_active_net_exposure",
                    "avg_abs_active_pc1_exposure",
                ]
            ],
        ],
        ignore_index=True,
        sort=False,
    )
    comparison.to_csv(REPORT_DIR / "three_line_strategy_summary.csv", index=False)
    write_report(comparison, signal_panel, args)
    print(comparison.sort_values(["fee_bps", "strategy"]).to_string(index=False))
    print(f"hourly: {OUT_DIR / 'advanced_pca_v1_hourly_equity.csv'}")
    print(f"positions: {OUT_DIR / 'advanced_pca_v1_positions.csv'}")
    print(f"summary: {REPORT_DIR / 'three_line_strategy_summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final-pipeline advanced PCA v1 branch.")
    parser.add_argument("--max-hours", type=int, default=None, help="Cap PCA generation hours for smoke runs.")
    parser.add_argument("--full-run", action="store_true", help="Run all available hours.")
    parser.add_argument("--m-components", type=int, default=8)
    parser.add_argument("--min-assets", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--random-starts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--parallel-backend", choices=["threading", "loky"], default="threading")
    parser.add_argument("--reuse", action="store_true", help="Reuse final_pipeline/advanced_pca_v1 residual or signal panel if present.")
    parser.add_argument("--diagnostics-only", action="store_true", help="Only build advanced PCA diagnostics and figure.")
    args = parser.parse_args()
    if args.full_run:
        args.max_hours = None
    return args


if __name__ == "__main__":
    run(parse_args())
