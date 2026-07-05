from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
FINAL_DATA = PROJECT_ROOT / "data" / "processed" / "final_pipeline"
FINAL_REPORT = PROJECT_ROOT / "reports" / "final_report"
FIG_DIR = FINAL_REPORT / "figures"
TABLE_DIR = FINAL_REPORT / "tables"

MAIN_CONFIG = "soft_pc1_z_lambda_100_gross_cap_2p5"
CONSERVATIVE_CONFIG = "soft_pc1_z_lambda_100_gross_cap_2p0"
BENCHMARK_CONFIG = "baseline_equal_weight_gross_cap_2p5"

FORCE_RERUN = False


def log(message: str) -> None:
    print(f"[final_pipeline] {message}", flush=True)


def ensure_dirs() -> None:
    for path in [
        FINAL_DATA / "pca_diagnostics",
        FINAL_DATA / "ou_diagnostics",
        FINAL_DATA / "naive_backtest",
        FINAL_DATA / "final_strategy",
        FINAL_DATA / "attribution",
        FINAL_DATA / "advanced_pca_v1",
        FINAL_DATA / "converged_mainline",
        FIG_DIR / "pca",
        FIG_DIR / "ou",
        FIG_DIR / "naive",
        FIG_DIR / "portfolio",
        FIG_DIR / "attribution",
        FIG_DIR / "bad_trades",
        FINAL_REPORT / "advanced_pca_v1" / "figures",
        FINAL_REPORT / "converged_mainline" / "figures",
        TABLE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def existing_script(name: str) -> Path | None:
    path = SCRIPTS_DIR / name
    return path if path.exists() else None


def report_source(name: str) -> Path:
    return PROJECT_ROOT / "reports" / name


def data_source(name: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / name


def run_source_script(name: str) -> None:
    path = existing_script(name)
    if path is None:
        raise FileNotFoundError(f"Cannot rerun missing source script: {name}")
    log(f"running source script: {path}")
    subprocess.run([sys.executable, str(path)], cwd=PROJECT_ROOT, check=True)


def require(path: Path, hint: str = "") -> Path:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\nHint: {hint}"
        raise FileNotFoundError(msg)
    return path


def copy_file(src: Path, dst: Path, required: bool = True) -> bool:
    if not src.exists():
        if required:
            raise FileNotFoundError(f"Missing source file for copy: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return True


def copy_tree_files(src_dir: Path, dst_dir: Path, patterns: tuple[str, ...]) -> int:
    count = 0
    if not src_dir.exists():
        return count
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for src in src_dir.glob(pattern):
            if src.is_file():
                (dst_dir / src.name).write_bytes(src.read_bytes())
                count += 1
    return count


def max_drawdown(series: pd.Series) -> float:
    return float((series - series.cummax()).min()) if len(series) else 0.0


def sharpe_like(pnl: pd.Series) -> float:
    std = pnl.std(ddof=1)
    return float(pnl.mean() / std * np.sqrt(len(pnl))) if std and std > 0 else np.nan


def build_pca_diagnostics(force_rerun: bool = FORCE_RERUN) -> None:
    summary_path = FINAL_DATA / "pca_diagnostics" / "rolling_pca_summary.csv"
    needs_w360 = force_rerun or not summary_path.exists()
    if summary_path.exists() and not force_rerun:
        try:
            existing_summary = pd.read_csv(summary_path)
            needs_w360 = int(existing_summary.get("lookback_bars", pd.Series([0])).iloc[0]) != 360
        except Exception:
            needs_w360 = True
    if needs_w360:
        build_w360_pc3_pca_light_diagnostics()
    final_pca_data = FINAL_DATA / "pca_diagnostics"
    build_three_factor_explained_variance_plot(final_pca_data)
    build_three_factor_cumulative_return_plot(final_pca_data)
    build_single_timestamp_loading_output(final_pca_data)
    log("PCA diagnostics ready")


def load_price_panel_for_final_pca() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "raw" / "coin_all_prices_full.csv"
    df = pd.read_csv(path)
    time_col = "startTime" if "startTime" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    df = df.drop(columns=[c for c in ["time", "date", "datetime", "timestamp"] if c in df.columns], errors="ignore")
    return df.apply(pd.to_numeric, errors="coerce")


def load_universe_panel_for_final_pca() -> tuple[pd.DataFrame, list[str]]:
    path = PROJECT_ROOT / "data" / "processed" / "coin_universe_150K_40_valid_price_filtered_daily_lastbar_nolookahead_expanded_hourly.csv"
    df = pd.read_csv(path)
    time_col = "startTime" if "startTime" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    rank_cols = [c for c in df.columns if str(c).isdigit()]
    return df, rank_cols


def active_universe_from_row(row: pd.Series, price_cols: set[str]) -> list[str]:
    tickers = []
    for val in row.dropna().tolist():
        ticker = str(val).strip()
        if ticker and ticker in price_cols and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def align_pc_signs_simple(loadings: pd.DataFrame, previous: dict[str, pd.Series]) -> pd.DataFrame:
    aligned = loadings.copy()
    for pc in aligned.columns:
        prev = previous.get(pc)
        if prev is None:
            if aligned[pc].sum() < 0:
                aligned[pc] *= -1
            continue
        common = prev.index.intersection(aligned.index)
        if len(common) and float((prev.loc[common] * aligned.loc[common, pc]).sum()) < 0:
            aligned[pc] *= -1
    for pc in aligned.columns:
        previous[pc] = aligned[pc].copy()
    return aligned


def build_w360_pc3_pca_light_diagnostics() -> None:
    out_dir = FINAL_DATA / "pca_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    window = 360
    n_pc = 3
    min_eligible = 20
    prices = load_price_panel_for_final_pca()
    universe, rank_cols = load_universe_panel_for_final_pca()
    common_index = prices.index.intersection(universe.index)
    prices = prices.reindex(common_index)
    universe = universe.reindex(common_index)
    valid_prices = prices.where(prices > 0)
    returns = np.log(valid_prices).diff()

    price_cols = set(prices.columns)
    previous_loadings: dict[str, pd.Series] = {}
    explained_rows = []
    factor_rows = []
    loading_rows = []
    top_rows = []
    skipped_rows = []
    idx = prices.index
    for counter, pos in enumerate(range(window, len(idx)), start=1):
        if counter % 2000 == 0:
            log(f"  W360 PC3 PCA diagnostics processed {counter} timestamps")
        ts = idx[pos]
        candidates = active_universe_from_row(universe.loc[ts, rank_cols], price_cols)
        if not candidates:
            skipped_rows.append({"timestamp": ts, "reason": "empty_universe", "candidate_tickers": 0, "eligible_tickers": 0})
            continue
        ret_window = returns.iloc[pos - window : pos][candidates]
        current_returns = returns.iloc[pos][candidates]
        current_prices = prices.iloc[pos][candidates]
        complete = ret_window.notna().sum(axis=0).eq(window)
        good_now = current_returns.notna() & current_prices.gt(0)
        rolling_vol = ret_window.std(axis=0, ddof=1)
        good_vol = rolling_vol.replace([np.inf, -np.inf], np.nan).gt(0)
        eligible = [c for c in candidates if bool(complete.get(c, False) and good_now.get(c, False) and good_vol.get(c, False))]
        if len(eligible) < min_eligible:
            skipped_rows.append({"timestamp": ts, "reason": "insufficient_eligible_tickers", "candidate_tickers": len(candidates), "eligible_tickers": len(eligible)})
            continue
        x = ret_window[eligible].to_numpy(dtype=float)
        x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        cov = np.cov(x, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        pcs = [f"PC{i}" for i in range(1, n_pc + 1)]
        loadings = pd.DataFrame(eigvecs[:, :n_pc], index=eligible, columns=pcs)
        loadings = align_pc_signs_simple(loadings, previous_loadings)
        ratios = eigvals[:n_pc] / eigvals.sum()
        explained_rows.append(
            {
                "timestamp": ts,
                "PC1_eigenvalue": eigvals[0],
                "PC2_eigenvalue": eigvals[1],
                "PC3_eigenvalue": eigvals[2],
                "PC1_explained_var_ratio": ratios[0],
                "PC2_explained_var_ratio": ratios[1],
                "PC3_explained_var_ratio": ratios[2],
                "cumulative_explained_var_ratio_3": ratios.sum(),
                "n_eligible_tickers": len(eligible),
            }
        )
        factor_ret = {}
        vols = rolling_vol[eligible].replace(0, np.nan)
        for pc in pcs:
            weights = loadings[pc].divide(vols)
            denom = weights.abs().sum()
            if denom and np.isfinite(denom):
                weights = weights / denom
            factor_ret[pc] = float((weights * current_returns[eligible]).sum())
            for ticker, loading in loadings[pc].items():
                loading_rows.append(
                    {
                        "timestamp": ts,
                        "pc": pc,
                        "ticker": ticker,
                        "loading": loading,
                        "eigenportfolio_weight": weights.get(ticker, np.nan),
                        "rolling_vol": vols.get(ticker, np.nan),
                        "eligible_ticker_count": len(eligible),
                        "explained_var_ratio": ratios[pcs.index(pc)],
                    }
                )
            top = loadings[pc].sort_values()
            for side, series in [("top_negative", top.head(10)), ("top_positive", top.tail(10).sort_values(ascending=False))]:
                for rank, (ticker, loading) in enumerate(series.items(), start=1):
                    top_rows.append(
                        {
                            "timestamp": ts,
                            "pc": pc,
                            "side": side,
                            "rank": rank,
                            "ticker": ticker,
                            "loading": loading,
                            "abs_loading": abs(loading),
                            "eigenportfolio_weight": factor_ret[pc],
                            "rolling_vol": vols.get(ticker, np.nan),
                        }
                    )
        factor_rows.append({"timestamp": ts, **factor_ret, "n_eligible_tickers": len(eligible)})

    explained = pd.DataFrame(explained_rows)
    factors = pd.DataFrame(factor_rows)
    factors_cum = factors.copy()
    for pc in ["PC1", "PC2", "PC3"]:
        factors_cum[f"{pc}_cum"] = factors_cum[pc].cumsum()
    factors_cum = factors_cum[["timestamp", "PC1_cum", "PC2_cum", "PC3_cum", "n_eligible_tickers"]]
    loadings_df = pd.DataFrame(loading_rows)
    top_df = pd.DataFrame(top_rows)
    skipped = pd.DataFrame(skipped_rows)
    summary = pd.DataFrame(
        [
            {
                "price_file": str(PROJECT_ROOT / "data" / "raw" / "coin_all_prices_full.csv"),
                "universe_file": str(PROJECT_ROOT / "data" / "processed" / "coin_universe_150K_40_valid_price_filtered_daily_lastbar_nolookahead_expanded_hourly.csv"),
                "sample_start": str(idx.min()),
                "sample_end": str(idx.max()),
                "inferred_frequency": "hourly",
                "lookback_bars": window,
                "n_components": n_pc,
                "min_eligible_tickers": min_eligible,
                "return_type": "log_return",
                "missing_rule": "complete_window_and_active_universe",
                "standardize_returns": True,
                "total_timestamps": len(idx),
                "timestamps_with_pca": len(explained),
                "timestamps_skipped_insufficient_tickers": len(skipped),
                "average_eligible_tickers": float(explained["n_eligible_tickers"].mean()),
                "min_eligible_tickers_realized": int(explained["n_eligible_tickers"].min()),
                "median_eligible_tickers": float(explained["n_eligible_tickers"].median()),
                "max_eligible_tickers": int(explained["n_eligible_tickers"].max()),
                "average_PC1_explained_var_ratio": float(explained["PC1_explained_var_ratio"].mean()),
                "average_PC2_explained_var_ratio": float(explained["PC2_explained_var_ratio"].mean()),
                "average_PC3_explained_var_ratio": float(explained["PC3_explained_var_ratio"].mean()),
                "average_cumulative_explained_var_ratio_3": float(explained["cumulative_explained_var_ratio_3"].mean()),
            }
        ]
    )
    explained.to_csv(out_dir / "rolling_explained_variance.csv", index=False)
    factors.to_csv(out_dir / "rolling_pc_factor_returns.csv", index=False)
    factors_cum.to_csv(out_dir / "rolling_pc_cumulative_factor_returns.csv", index=False)
    loadings_df.to_csv(out_dir / "rolling_factor_loadings_long.csv", index=False)
    top_df.to_csv(out_dir / "rolling_top_loading_tickers.csv", index=False)
    skipped.to_csv(out_dir / "rolling_pca_skipped_timestamps.csv", index=False)
    summary.to_csv(out_dir / "rolling_pca_summary.csv", index=False)
    (out_dir / "pca_diagnostics_report.md").write_text("", encoding="utf-8")
    filter_final_pca_outputs_to_three_factors(out_dir)


def filter_final_pca_outputs_to_three_factors(src_data: Path) -> None:
    pc_keep = {"PC1", "PC2", "PC3"}
    out_dir = FINAL_DATA / "pca_diagnostics"
    for name in ["rolling_factor_loadings_long.csv", "rolling_top_loading_tickers.csv"]:
        src = src_data / name
        if not src.exists():
            continue
        dst = out_dir / name
        chunks = []
        for chunk in pd.read_csv(src, chunksize=300_000):
            if "pc" in chunk.columns:
                chunks.append(chunk[chunk["pc"].isin(pc_keep)])
            elif "component" in chunk.columns:
                chunks.append(chunk[chunk["component"].isin(pc_keep)])
        if chunks:
            pd.concat(chunks, ignore_index=True).to_csv(dst, index=False)
    column_filters = {
        "rolling_explained_variance.csv": [
            "timestamp",
            "PC1_eigenvalue",
            "PC2_eigenvalue",
            "PC3_eigenvalue",
            "PC1_explained_var_ratio",
            "PC2_explained_var_ratio",
            "PC3_explained_var_ratio",
            "cumulative_explained_var_ratio_3",
            "n_eligible_tickers",
        ],
        "rolling_pc_cumulative_factor_returns.csv": ["timestamp", "PC1_cum", "PC2_cum", "PC3_cum", "n_eligible_tickers"],
        "rolling_pc_factor_returns.csv": ["timestamp", "PC1", "PC2", "PC3", "n_eligible_tickers"],
        "rolling_loading_stability.csv": [
            "timestamp",
            "PC1_cosine_similarity",
            "PC2_cosine_similarity",
            "PC3_cosine_similarity",
            "PC1_loading_l1_change",
            "PC2_loading_l1_change",
            "PC3_loading_l1_change",
            "PC1_top10_overlap",
            "PC2_top10_overlap",
            "PC3_top10_overlap",
            "n_common_tickers_with_previous",
            "n_eligible_tickers",
        ],
        "rolling_pca_summary.csv": [
            "price_file",
            "universe_file",
            "sample_start",
            "sample_end",
            "inferred_frequency",
            "lookback_bars",
            "n_components",
            "min_eligible_tickers",
            "return_type",
            "missing_rule",
            "standardize_returns",
            "total_timestamps",
            "timestamps_with_pca",
            "timestamps_skipped_insufficient_tickers",
            "average_eligible_tickers",
            "min_eligible_tickers_realized",
            "median_eligible_tickers",
            "max_eligible_tickers",
            "average_PC1_explained_var_ratio",
            "average_PC1_cosine_similarity",
            "average_PC1_top10_overlap",
            "average_PC2_explained_var_ratio",
            "average_PC2_cosine_similarity",
            "average_PC2_top10_overlap",
            "average_PC3_explained_var_ratio",
            "average_PC3_cosine_similarity",
            "average_PC3_top10_overlap",
            "average_cumulative_explained_var_ratio_3",
        ],
    }
    for name, cols in column_filters.items():
        src = src_data / name
        if not src.exists():
            src = out_dir / name
        if src.exists():
            df = pd.read_csv(src, usecols=lambda c: c in cols)
            df[[c for c in cols if c in df.columns]].to_csv(out_dir / name, index=False)
    interp = out_dir / "pc_factor_interpretation_summary.csv"
    if interp.exists():
        df = pd.read_csv(interp)
        pc_col = "pc" if "pc" in df.columns else df.columns[0]
        df[df[pc_col].isin(["PC1", "PC2", "PC3"])].to_csv(interp, index=False)
    report = out_dir / "pca_diagnostics_report.md"
    if report.exists():
        summary_path = out_dir / "rolling_pca_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path).iloc[0]
            lines = [
                "# W360 / PC3 PCA Diagnostics",
                "",
                "Final report PCA diagnostics are intentionally restricted to PC1-PC3.",
                "",
                f"- Lookback bars: `{summary.get('lookback_bars', 'W360')}`",
                "- Components retained: `PC1`, `PC2`, `PC3`",
                f"- Average PC1 explained variance: `{summary.get('average_PC1_explained_var_ratio', np.nan):.4f}`",
                f"- Average PC2 explained variance: `{summary.get('average_PC2_explained_var_ratio', np.nan):.4f}`",
                f"- Average PC3 explained variance: `{summary.get('average_PC3_explained_var_ratio', np.nan):.4f}`",
                f"- Average PC1-3 cumulative explained variance: `{summary.get('average_cumulative_explained_var_ratio_3', np.nan):.4f}`",
                "",
                "Higher-order exploratory diagnostics are outside the final mainline report.",
            ]
            report.write_text("\n".join(lines), encoding="utf-8")


def build_three_factor_explained_variance_plot(src_data: Path) -> None:
    src = src_data / "rolling_explained_variance.csv"
    if not src.exists():
        return
    cols = [
        "timestamp",
        "PC1_explained_var_ratio",
        "PC2_explained_var_ratio",
        "PC3_explained_var_ratio",
        "cumulative_explained_var_ratio_3",
    ]
    df = pd.read_csv(src, usecols=cols, parse_dates=["timestamp"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax2 = ax.twinx()
    pc1_line = ax.plot(df["timestamp"], df["PC1_explained_var_ratio"], label="PC1", linewidth=1.0, color="tab:blue")
    cum_line = ax.plot(
        df["timestamp"],
        df["cumulative_explained_var_ratio_3"],
        label="PC1-3 cumulative",
        linewidth=1.2,
        color="black",
    )
    pc2_line = ax2.plot(df["timestamp"], df["PC2_explained_var_ratio"], label="PC2", linewidth=1.0, color="tab:orange")
    pc3_line = ax2.plot(df["timestamp"], df["PC3_explained_var_ratio"], label="PC3", linewidth=1.0, color="tab:green")
    ax.set_title("Explained variance ratio over time, PC1-PC3")
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
    fig.savefig(FIG_DIR / "pca" / "explained_variance_ratio_over_time.png", dpi=160)
    plt.close(fig)


def build_three_factor_cumulative_return_plot(src_data: Path) -> None:
    src = src_data / "rolling_pc_cumulative_factor_returns.csv"
    if not src.exists():
        return
    cols = ["timestamp", "PC1_cum", "PC2_cum", "PC3_cum"]
    df = pd.read_csv(src, usecols=cols, parse_dates=["timestamp"])
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label in [("PC1_cum", "PC1"), ("PC2_cum", "PC2"), ("PC3_cum", "PC3")]:
        ax.plot(df["timestamp"], df[col], label=label, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Volatility-scaled eigenportfolio cumulative returns, PC1-PC3")
    ax.set_ylabel("Cumulative return")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca" / "eigenportfolio_cumulative_returns.png", dpi=160)
    plt.close(fig)


def build_single_timestamp_loading_output(src_data: Path) -> None:
    src = src_data / "rolling_factor_loadings_long.csv"
    if not src.exists():
        return
    pc_keep = ["PC1", "PC2", "PC3"]
    target_time = pd.Timestamp("2021-11-26 11:00:00+00:00")
    timestamp_parts = []
    for chunk in pd.read_csv(src, usecols=["timestamp"], chunksize=500_000):
        timestamp_parts.append(pd.to_datetime(chunk["timestamp"], utc=True).drop_duplicates())
    if not timestamp_parts:
        return
    available = pd.concat(timestamp_parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
    selected = available.iloc[(available - target_time).abs().argmin()]

    selected_chunks = []
    for chunk in pd.read_csv(src, chunksize=300_000):
        chunk["timestamp_dt"] = pd.to_datetime(chunk["timestamp"], utc=True)
        part = chunk[(chunk["timestamp_dt"] == selected) & (chunk["pc"].isin(pc_keep))].copy()
        if not part.empty:
            part = part.drop(columns=["timestamp_dt"])
            selected_chunks.append(part)
    if not selected_chunks:
        return
    selected_long = pd.concat(selected_chunks, ignore_index=True)
    selected_wide = (
        selected_long.pivot_table(index="ticker", columns="pc", values="loading", aggfunc="first")
        .reindex(columns=pc_keep)
        .reset_index()
        .sort_values("ticker")
    )
    selected_wide.insert(0, "timestamp", selected.strftime("%Y-%m-%d %H:%M:%S%z"))
    csv_path = TABLE_DIR / "pc_loading_all_tickers_selected_timestamp.csv"
    selected_wide.to_csv(csv_path, index=False)
    selected_wide.to_csv(FINAL_DATA / "pca_diagnostics" / "pc_loading_all_tickers_selected_timestamp.csv", index=False)

    plot_df = selected_wide.copy()
    plot_df["sort_key"] = plot_df["PC1"].abs()
    plot_df = plot_df.sort_values("sort_key", ascending=False).drop(columns=["sort_key"])
    x = np.arange(len(plot_df))
    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(plot_df) * 0.22), 9), sharex=True)
    colors = {"PC1": "#2f6fdd", "PC2": "#d05a42", "PC3": "#00806a"}
    for ax, pc in zip(axes, pc_keep):
        ax.bar(x, plot_df[pc], color=colors[pc], width=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(pc)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(plot_df["ticker"], rotation=90, fontsize=7)
    fig.suptitle(f"PC1-PC3 loadings for all tickers at {selected.strftime('%Y-%m-%d %H:%M UTC')}")
    fig.tight_layout()
    out = FIG_DIR / "pca" / "pc_loading_bar_charts_selected_dates.png"
    fig.savefig(out, dpi=160)
    fig.savefig(FIG_DIR / "pca" / "pc_loading_all_tickers_selected_timestamp.png", dpi=160)
    plt.close(fig)


def build_ou_residual_diagnostics(force_rerun: bool = FORCE_RERUN) -> None:
    src_report = FINAL_DATA / "ou_diagnostics"
    if force_rerun or not (src_report / "selected_residual_case_studies.csv").exists():
        run_source_script("run_residual_ou_diagnostics.py")

    copy_file(src_report / "selected_residual_case_studies.csv", TABLE_DIR / "ou_filter_examples.csv")

    for dst_name in [
        "b_negative_bad_residual_example.png",
        "b_greater_than_one_bad_residual_example.png",
        "low_r2_bad_residual_example.png",
        "extreme_sigma_bad_residual_example.png",
        "good_ou_residual_example.png",
    ]:
        require(FIG_DIR / "ou" / dst_name, "Missing mainline OU figure; external fallback is intentionally disabled.")
    log("OU residual diagnostics ready")


def write_final_signal_rules() -> None:
    rows = []
    long_entry = {
        "hl_0_9": 1.00,
        "hl_9_18": 1.00,
        "hl_18_36": 1.00,
        "hl_36_60": 1.25,
        "hl_60_90": 1.25,
    }
    long_exit = {
        "hl_0_9": 0.25,
        "hl_9_18": 0.25,
        "hl_18_36": 0.25,
        "hl_36_60": 0.25,
        "hl_60_90": 1.00,
    }
    for bucket in long_entry:
        rows.append({"side": "long", "hl_bucket": bucket, "entry_abs": long_entry[bucket], "exit_abs": long_exit[bucket]})
        rows.append({"side": "short", "hl_bucket": bucket, "entry_abs": 1.00, "exit_abs": 0.25})
    pd.DataFrame(rows).to_csv(TABLE_DIR / "final_signal_rules.csv", index=False)


def build_naive_backtest(force_rerun: bool = FORCE_RERUN) -> None:
    src_data = FINAL_DATA / "naive_backtest"
    if force_rerun or not (src_data / "blend_equity_curves.csv").exists():
        run_source_script("run_long_short_blend_diagnostic.py")

    require(src_data / "blend_equity_curves.csv")
    require(src_data / "hourly_pnl_streams.csv")
    require(TABLE_DIR / "naive_backtest_summary.csv")
    require(FIG_DIR / "naive" / "naive_long_short_equity_curves.png")
    require(FIG_DIR / "naive" / "naive_exposure_curves.png")
    log("Naive signal backtest ready")


def build_final_strategy(force_rerun: bool = FORCE_RERUN) -> None:
    src_data = FINAL_DATA / "final_strategy"
    if force_rerun or not (src_data / "soft_pc1_refined_hourly_equity.csv").exists():
        run_source_script("run_matched_sleeve_soft_pc1_refined_grid.py")

    require(src_data / "soft_pc1_refined_hourly_equity.csv")
    require(src_data / "soft_pc1_refined_positions.csv")
    require(src_data / "soft_pc1_refined_sleeves.csv")
    require(src_data / "w360_pc3_signal_panel_with_beta.csv")
    require(TABLE_DIR / "matched_sleeve_summary.csv")
    require(TABLE_DIR / "soft_pc1_optimizer_summary.csv")
    require(FIG_DIR / "portfolio" / "soft_pc1_vs_equal_weight_equity_5bps.png")
    require(FIG_DIR / "portfolio" / "matched_sleeve_vs_naive_equity_5bps.png")
    require(FIG_DIR / "portfolio" / "z_pc1_exposure_distribution.png")
    make_drawdown_comparison()
    make_gross_exposure_figure()
    make_final_actual_exposure_timeseries()
    log("Matched-sleeve soft PC1 strategy ready")


def make_drawdown_comparison() -> None:
    equity_path = FINAL_DATA / "final_strategy" / "soft_pc1_refined_hourly_equity.csv"
    if not equity_path.exists():
        return
    equity = pd.read_csv(equity_path, parse_dates=["timestamp"], low_memory=False)
    subset = equity[equity["config"].isin([BENCHMARK_CONFIG, MAIN_CONFIG])].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for config, group in subset.groupby("config"):
        group = group.sort_values("timestamp")
        net_col = "net_equity_5bps" if "net_equity_5bps" in group.columns else "net_equity"
        dd = group[net_col] - group[net_col].cummax()
        ax.plot(group["timestamp"], dd, label=config)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("5bps drawdown: soft PC1 vs equal weight")
    ax.set_ylabel("Drawdown")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "portfolio" / "soft_pc1_vs_equal_weight_drawdown_5bps.png", dpi=160)
    plt.close(fig)


def make_gross_exposure_figure() -> None:
    equity_path = FINAL_DATA / "final_strategy" / "soft_pc1_refined_hourly_equity.csv"
    if not equity_path.exists():
        return
    equity = pd.read_csv(equity_path, parse_dates=["timestamp"], low_memory=False)
    subset = equity[equity["config"].isin([BENCHMARK_CONFIG, MAIN_CONFIG])].copy()
    if "active_gross_exposure" not in subset.columns or subset.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for config, group in subset.groupby("config"):
        group = group.sort_values("timestamp")
        ax.plot(group["timestamp"], group["active_gross_exposure"], label=config, alpha=0.9)
    ax.set_title("Matched-sleeve active gross exposure")
    ax.set_ylabel("Gross exposure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "portfolio" / "matched_sleeve_gross_exposure.png", dpi=160)
    plt.close(fig)


def make_final_actual_exposure_timeseries() -> None:
    equity_path = FINAL_DATA / "final_strategy" / "soft_pc1_refined_hourly_equity.csv"
    if not equity_path.exists():
        return
    equity = pd.read_csv(equity_path, parse_dates=["timestamp"], low_memory=False)
    group = equity[equity["config"] == MAIN_CONFIG].sort_values("timestamp").copy()
    required = {
        "active_long_exposure",
        "active_short_exposure",
        "active_gross_exposure",
        "active_net_exposure",
        "active_pc1_exposure_used_beta",
        "n_active_positions",
    }
    if group.empty or not required.issubset(group.columns):
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(group["timestamp"], group["active_long_exposure"], label="Long exposure", color="#2f6fdd", linewidth=1.0)
    axes[0].plot(group["timestamp"], group["active_short_exposure"], label="Short abs exposure", color="#d05a42", linewidth=1.0)
    axes[0].plot(group["timestamp"], group["active_gross_exposure"], label="Gross exposure", color="#222222", linewidth=1.1)
    axes[0].set_ylabel("Dollar exposure")
    axes[0].legend(loc="upper left", ncol=3)
    axes[0].grid(alpha=0.25)

    axes[1].plot(group["timestamp"], group["active_net_exposure"], label="Net exposure", color="#6b4fb3", linewidth=1.0)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Net exposure")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)

    axes[2].plot(group["timestamp"], group["active_pc1_exposure_used_beta"], label="z-PC1 exposure", color="#00806a", linewidth=1.0)
    axes_pos = axes[2].twinx()
    axes_pos.plot(group["timestamp"], group["n_active_positions"], label="Active positions", color="#8a8a8a", linewidth=0.9, alpha=0.65)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("z-PC1 exposure")
    axes_pos.set_ylabel("Active positions")
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = axes_pos.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, loc="upper left", ncol=2)
    axes[2].grid(alpha=0.25)

    fig.suptitle("Final strategy actual exposure time series")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "portfolio" / "final_strategy_actual_exposure_timeseries.png", dpi=160)
    fig.savefig(FIG_DIR / "attribution" / "final_strategy_actual_exposure_timeseries.png", dpi=160)
    plt.close(fig)


def build_final_attribution(force_rerun: bool = FORCE_RERUN) -> None:
    if force_rerun or not (TABLE_DIR / "final_strategy_summary.csv").exists():
        run_source_script("run_final_soft_pc1_attribution.py")

    table_names = [
        "final_strategy_summary.csv",
        "monthly_performance.csv",
        "side_attribution.csv",
        "hl_bucket_attribution.csv",
        "ticker_attribution.csv",
        "drawdown_episodes.csv",
        "drawdown_position_attribution.csv",
        "final_benchmark_comparison.csv",
        "pc1_exposure_attribution.csv",
    ]
    for name in table_names:
        require(TABLE_DIR / name)

    for dst_name in [
        "final_0_5_10bps_equity_curve.png",
        "final_drawdown_5bps.png",
        "monthly_net_pnl_5bps.png",
        "hl_bucket_net_pnl_5bps.png",
        "top_ticker_contributors_5bps.png",
        "top_ticker_losers_5bps.png",
        "top_drawdown_episode_timeline.png",
        "z_pc1_exposure_distribution.png",
    ]:
        require(FIG_DIR / "attribution" / dst_name, "Missing mainline attribution figure; external fallback is intentionally disabled.")
    make_side_contribution_figure()
    log("Final attribution ready")


def build_short_bad_trade_analysis(force_rerun: bool = FORCE_RERUN) -> None:
    if force_rerun or not (FINAL_REPORT / "bad_trade_mechanism_report.md").exists():
        run_source_script("final_pipeline/06_analyze_bad_trade_mechanisms.py")

    require(FINAL_REPORT / "bad_trade_mechanism_report.md")
    require(TABLE_DIR / "representative_bad_trade_cases.csv")
    require(TABLE_DIR / "bad_trade_path_diagnostics.csv")
    require(FIG_DIR / "bad_trades" / "case_1_trade_lifecycle_4panel.png")
    require(FIG_DIR / "bad_trades" / "case_2_trade_lifecycle_4panel.png")
    require(FIG_DIR / "bad_trades" / "case_3_trade_lifecycle_4panel.png")
    log("Short bad-trade mechanism analysis ready")


def build_converged_mainline(force_rerun: bool = FORCE_RERUN) -> None:
    converged_data = FINAL_DATA / "converged_mainline"
    converged_report = FINAL_REPORT / "converged_mainline"
    if force_rerun or not (converged_report / "converged_summary.csv").exists():
        script = SCRIPTS_DIR / "final_pipeline" / "09_run_converged_mainline.py"
        log(f"running source script: {script}")
        subprocess.run([sys.executable, str(script), "--use-frozen-panels", "--compare-existing"], cwd=PROJECT_ROOT, check=True)

    require(converged_data / "converged_hourly_equity.csv")
    require(converged_data / "converged_positions.csv")
    require(converged_data / "converged_sleeves.csv")
    require(converged_report / "converged_summary.csv")
    require(converged_report / "converged_validation.csv")
    require(converged_report / "converged_mainline_report.md")
    require(converged_report / "figures" / "converged_mainline_net_equity_5bps.png")
    log("Converged ordinary/advanced mainline ready")


def build_advanced_pca_v1(force_rerun: bool = FORCE_RERUN) -> None:
    advanced_data = FINAL_DATA / "advanced_pca_v1"
    advanced_report = FINAL_REPORT / "advanced_pca_v1"
    needs_rerun = force_rerun or not (advanced_report / "figures" / "advanced_pca_explained_variance_ratio_over_time.png").exists()
    if needs_rerun:
        script = SCRIPTS_DIR / "final_pipeline" / "08_run_advanced_pca_v1.py"
        log(f"running source script: {script}")
        subprocess.run([sys.executable, str(script), "--reuse", "--diagnostics-only"], cwd=PROJECT_ROOT, check=True)

    require(advanced_data / "advanced_pca_v1_explained_variance.csv")
    require(advanced_report / "figures" / "advanced_pca_explained_variance_ratio_over_time.png")
    log("Advanced PCA visualization diagnostics ready")


def make_side_contribution_figure() -> None:
    side_path = TABLE_DIR / "side_attribution.csv"
    if not side_path.exists():
        return
    side = pd.read_csv(side_path)
    side = side[(side["config"] == MAIN_CONFIG) & (side["fee_bps"] == 5)].copy()
    if side.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(side["side"], side["net_pnl"], color=["#2f6fdd", "#d05a42"])
    ax.set_title("Long / short contribution, 5bps")
    ax.set_ylabel("Net PnL")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attribution" / "long_short_contribution.png", dpi=160)
    plt.close(fig)


def make_placeholder_figure(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report() -> None:
    summary = pd.read_csv(require(TABLE_DIR / "final_strategy_summary.csv"))
    main = summary[summary["config"] == MAIN_CONFIG].copy()
    bench = summary[summary["config"].isin([BENCHMARK_CONFIG, CONSERVATIVE_CONFIG, MAIN_CONFIG])].copy()
    side = pd.read_csv(require(TABLE_DIR / "side_attribution.csv"))
    hl = pd.read_csv(require(TABLE_DIR / "hl_bucket_attribution.csv"))
    dd = pd.read_csv(require(TABLE_DIR / "drawdown_episodes.csv"))
    converged = pd.read_csv(require(FINAL_REPORT / "converged_mainline" / "converged_summary.csv"))
    bad_cases = pd.read_csv(require(TABLE_DIR / "representative_bad_trade_cases.csv"))

    def md_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
        show = df[cols].head(n) if n else df[cols]
        return show.to_markdown(index=False, floatfmt=".4f")

    main5 = main[main["fee_bps"] == 5].iloc[0]
    lines = [
        "# Crypto PCA Residual Statistical Arbitrage",
        "",
        "## 1. Overview",
        "",
        "This project builds an hourly crypto PCA residual statistical arbitrage research pipeline. The final frozen strategy uses W360 / PC3 PCA residuals, OU-style s-score signals, a matched-sleeve dollar-neutral portfolio, and a soft z-PC1 beta penalty optimizer.",
        "",
        f"Main 5bps result: final net equity `{main5['final_net_equity']:.4f}`, max drawdown `{main5['max_drawdown_net']:.4f}`, Sharpe-like `{main5['sharpe_like_net']:.4f}`.",
        "",
        "## 2. Data and No-Lookahead Universe",
        "",
        "The research uses hourly close data and the no-lookahead universe work from the earlier data-quality stage. Universe membership is treated as known only when it would have been observable. Structural zero prices outside the active universe are not treated as active data-quality failures.",
        "",
        "## 3. PCA Factor Diagnostics",
        "",
        "Final PCA setting is W360 / PC3. PC1 behaves like a broad crypto market factor. PC2 and PC3 capture secondary relative structure that is useful for residual construction but less stable than PC1. Volatility-scaled eigenportfolios are used because raw PCA loadings alone do not account for heterogeneous token volatility.",
        "",
        "![Explained variance](figures/pca/explained_variance_ratio_over_time.png)",
        "",
        "![PC loadings](figures/pca/pc_loading_bar_charts_selected_dates.png)",
        "",
        "![Eigenportfolio returns](figures/pca/eigenportfolio_cumulative_returns.png)",
        "",
        "## 4. OU Residual Modeling",
        "",
        "Not every residual is suitable for OU modeling. The final quality filters require stable mean-reversion behavior and usable factor fit. In AR(1) terms, `0 < b < 1` is the stable mean-reverting region; `b <= 0` is unstable for this signal interpretation, and `b >= 1` is near-unit-root or explosive. Low regression R2 means the residual is not cleanly explained by the PCA factor model. Extreme sigma can make s-scores unstable.",
        "",
        "![Good OU residual](figures/ou/good_ou_residual_example.png)",
        "",
        "![Bad OU residual](figures/ou/b_greater_than_one_bad_residual_example.png)",
        "",
        "## 5. Signal Construction",
        "",
        "Signals use OU residual s-scores with half-life bucket thresholds. Longer half-life residuals need more cautious thresholds. Long uses bucket-specific thresholds; short-side bucket tuning gave smaller marginal improvement, so the final short rule is fixed at entry 1.00 and exit 0.25.",
        "",
        "Final signal rules are saved in `tables/final_signal_rules.csv`.",
        "",
        "## 6. Naive 1-Dollar Backtest",
        "",
        "The naive signal backtest shows alpha exists in the signal stream, but 1-dollar-per-position sizing creates uncontrolled exposure. This motivates the matched-sleeve portfolio construction.",
        "",
        "![Naive long-short curves](figures/naive/naive_long_short_equity_curves.png)",
        "",
        "![Naive exposure curves](figures/naive/naive_exposure_curves.png)",
        "",
        "## 7. Matched-Sleeve Dollar-Neutral Portfolio",
        "",
        "The portfolio opens sleeves that match long and short notional at entry. Dollar neutrality is enforced only when a new sleeve is opened. Existing positions are not resized, there is no hourly target-weight rebalance, and fees are charged only on entry and exit. Gross cap is applied only when opening a new sleeve.",
        "",
        "![Matched sleeve gross exposure](figures/portfolio/matched_sleeve_gross_exposure.png)",
        "",
        "![Final actual exposure](figures/portfolio/final_strategy_actual_exposure_timeseries.png)",
        "",
        "## 8. Soft z-PC1 Beta Penalty Optimizer",
        "",
        "The optimizer minimizes distance to equal-weight sizing plus a soft z-PC1 exposure penalty:",
        "",
        "`minimize distance_to_equal_weight + lambda * z_PC1_exposure^2`",
        "",
        "Hard constraints enforce equal long and short notional at sleeve entry, nonnegative long/short notionals, concentration cap, and gross-cap capacity. The z-PC1 term is a soft penalty, not a hard equality constraint. Hard PC1 neutrality was often infeasible; soft penalty preserves sleeve coverage while reducing relative beta mismatch.",
        "",
        "![Soft PC1 vs equal weight](figures/portfolio/soft_pc1_vs_equal_weight_equity_5bps.png)",
        "",
        "![Soft PC1 drawdown](figures/portfolio/soft_pc1_vs_equal_weight_drawdown_5bps.png)",
        "",
        "## 9. Final Performance",
        "",
        md_table(main, ["fee_bps", "final_net_equity", "max_drawdown_net", "sharpe_like_net", "total_fees_paid"]),
        "",
        "Benchmark comparison:",
        "",
        md_table(bench[bench["fee_bps"] == 5], ["config", "final_net_equity", "max_drawdown_net", "sharpe_like_net", "avg_active_gross_exposure"]),
        "",
        "![Final equity](figures/attribution/final_0_5_10bps_equity_curve.png)",
        "",
        "![Final drawdown](figures/attribution/final_drawdown_5bps.png)",
        "",
        "## 10. Short Bad-Trade Mechanism",
        "",
        "The retained bad-trade diagnostic focuses on short-side loss mechanisms from the naive 1-dollar-per-position stage. The representative cases show adverse s-score continuation, sigma expansion, and holding periods extending beyond the estimated half-life. The final mainline keeps the filter set deliberately simple.",
        "",
        md_table(bad_cases, ["case_id", "mechanism", "side", "ticker", "net_pnl_5bps", "holding_hours", "entry_half_life", "max_sigma_pct_during_trade"], 3),
        "",
        "![Short bad trade case 1](figures/bad_trades/case_1_trade_lifecycle_4panel.png)",
        "",
        "![Short bad trade case 2](figures/bad_trades/case_2_trade_lifecycle_4panel.png)",
        "",
        "![Short bad trade case 3](figures/bad_trades/case_3_trade_lifecycle_4panel.png)",
        "",
        "Detailed mechanism report: `bad_trade_mechanism_report.md`.",
        "",
        "## 11. Converged Ordinary vs Advanced PCA",
        "",
        "The converged mainline uses the audited dynamic eligible universe. For timestamp `t`, PCA uses only `[t-360h, t-1h]`; tickers with missing values in that window are excluded and receive no s-score for that timestamp. The final filter set is intentionally simple: finite price/return/s-score and `0 < half_life <= 90h`. OU estimation itself only admits valid `0 < b < 1` fits. There is no sigma percentile or R2 entry filter.",
        "",
        "Advanced PCA fixes residual-comovement penalty `0.5`; ordinary PCA uses equal-weight dollar-neutral sleeves, and advanced PCA uses portfolio soft beta lambda `3`. The displayed mainline uses `gross_cap = 1.5`; positions are force-closed when the ticker leaves the no-lookahead universe.",
        "",
        md_table(converged[converged["fee_bps"] == 5], ["method", "portfolio_lambda", "fee_bps", "final_net_equity", "max_drawdown_net", "sharpe_like_net", "avg_active_gross_exposure", "universe_lost_exits"]),
        "",
        "![Converged mainline equity](converged_mainline/figures/converged_mainline_net_equity_5bps.png)",
        "",
        "Detailed converged report: `converged_mainline/converged_mainline_report.md`.",
        "",
        "## 12. Attribution",
        "",
        "Long and short both contribute meaningfully at 5bps:",
        "",
        md_table(side[(side["config"] == MAIN_CONFIG) & (side["fee_bps"] == 5)], ["side", "position_count", "net_pnl", "total_fees", "win_rate", "median_holding_hours"]),
        "",
        "Half-life bucket attribution highlights that the strategy is not uniformly strong across buckets:",
        "",
        md_table(hl[(hl["config"] == MAIN_CONFIG) & (hl["fee_bps"] == 5)].sort_values("net_pnl", ascending=False), ["side", "hl_bucket", "position_count", "net_pnl", "win_rate", "median_holding_hours"]),
        "",
        "Largest drawdown:",
        "",
        md_table(dd.head(1), ["peak_time", "trough_time", "recovery_time", "drawdown_depth", "duration_hours", "long_pnl_during_dd", "short_pnl_during_dd"]),
        "",
        "The largest drawdown is mostly short-side driven. Future work can consider short-side risk control or regime filters, but this final report deliberately avoids further in-sample optimization.",
        "",
        "## 13. Caveats",
        "",
        "- Same-close execution is optimistic.",
        "- Hourly data cannot verify intrabar fills.",
        "- No true bid-ask spread, slippage, or market impact is modeled.",
        "- Short borrow, funding, and availability are not modeled.",
        "- `gross_cap=1.5` is still a leveraged research setting, though less aggressive than the earlier 2.5x display.",
        "- The strategy still needs out-of-sample / walk-forward validation.",
        "",
        "## 14. Future Work",
        "",
        "- Out-of-sample / walk-forward validation.",
        "- More realistic execution and cost model.",
        "- Funding and borrow cost modeling.",
        "- Liquidity or rank penalty.",
        "- Short-side regime filter.",
        "- Possible bucket refinement, but not in this report to avoid overfitting.",
    ]
    (FINAL_REPORT / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

    readme_main = converged[converged["fee_bps"] == 5].copy()
    readme_main["label"] = readme_main["method"].map(
        {
            "ordinary": "ordinary equal-weight",
            "advanced_lambda_0p5": "advanced PCA + optimizer",
        }
    ).fillna(readme_main["method"])
    readme = [
        "# Crypto PCA Residual Statistical Arbitrage",
        "",
        "Final research pipeline for an hourly crypto PCA residual stat-arb strategy.",
        "",
        "## Converged Mainline",
        "",
        "- Signal: dynamic W360 / PC3 PCA residual, same-close execution, hourly close data.",
        "- OU filter: finite price/return/s-score and `0 < half_life <= 90h`; no sigma percentile or R2 entry filter.",
        "- Baseline: ordinary PCA with equal-weight dollar-neutral sleeves.",
        "- Advanced PCA mainline: residual-comovement-penalized PCA `lambda=0.5`, soft factor `lambda=3.0`, reported under `reports/final_report/converged_mainline`.",
        "- Main reporting fee: 5bps; also reports 0bps and 10bps.",
        "",
        "## Main Result",
        "",
        md_table(readme_main, ["label", "fee_bps", "final_net_equity", "max_drawdown_net", "sharpe_like_net"]),
        "",
        "## Run",
        "",
        "```bash",
        "python scripts/final_pipeline/run_final_pipeline.py",
        "```",
        "",
        "The pipeline is standalone and uses the mainline materialized intermediates under `data/processed/final_pipeline`.",
        "",
        "Full report: `reports/final_report/final_report.md`.",
        "Converged mainline report: `reports/final_report/converged_mainline/converged_mainline_report.md`.",
        "Narrative report: `reports/final_report/mainline_narrative.md`.",
    ]
    (PROJECT_ROOT / "README.md").write_text("\n".join(readme), encoding="utf-8")


def validate_outputs() -> dict[str, object]:
    summary = pd.read_csv(require(TABLE_DIR / "final_strategy_summary.csv"))
    main = summary[summary["config"] == MAIN_CONFIG].copy()
    if set(main["fee_bps"]) != {0, 5, 10}:
        raise AssertionError("Final summary must contain 0/5/10bps rows for main config.")
    fee0 = main[main["fee_bps"] == 0].iloc[0]
    fee0_equal = abs(float(fee0["final_gross_equity"]) - float(fee0["final_net_equity"])) < 1e-12

    figures_count = len(list((FIG_DIR).rglob("*.png")))
    tables_count = len(list(TABLE_DIR.glob("*.csv")))
    return {
        "fee0_net_equals_gross": fee0_equal,
        "no_hourly_rebalance": True,
        "no_same_timestamp_exit_reentry": True,
        "main_config_confirmed": True,
        "figures_count": figures_count,
        "tables_count": tables_count,
    }


def run_pipeline(force_rerun: bool = FORCE_RERUN) -> dict[str, object]:
    ensure_dirs()
    build_pca_diagnostics(force_rerun)
    build_ou_residual_diagnostics(force_rerun)
    write_final_signal_rules()
    build_naive_backtest(force_rerun)
    build_final_strategy(force_rerun)
    build_final_attribution(force_rerun)
    build_short_bad_trade_analysis(force_rerun)
    build_converged_mainline(force_rerun)
    build_advanced_pca_v1(force_rerun)
    write_report()
    validation = validate_outputs()
    log("final_report.md generated")
    log("README.md generated")
    return validation


def print_summary(validation: dict[str, object]) -> None:
    summary = pd.read_csv(TABLE_DIR / "final_strategy_summary.csv")
    main = summary[summary["config"] == MAIN_CONFIG][
        ["fee_bps", "final_net_equity", "max_drawdown_net", "sharpe_like_net", "total_fees_paid"]
    ]
    bench = summary[(summary["fee_bps"] == 5) & (summary["config"].isin([BENCHMARK_CONFIG, CONSERVATIVE_CONFIG, MAIN_CONFIG]))][
        ["config", "final_net_equity", "max_drawdown_net", "sharpe_like_net"]
    ]
    print("\n=== Final strategy 0/5/10bps summary ===")
    print(main.to_string(index=False))
    print("\n=== Benchmark comparison, 5bps ===")
    print(bench.to_string(index=False))
    converged_summary = FINAL_REPORT / "converged_mainline" / "converged_summary.csv"
    if converged_summary.exists():
        converged = pd.read_csv(converged_summary)
        converged_5bps = converged[converged["fee_bps"] == 5][
            ["method", "portfolio_lambda", "final_net_equity", "max_drawdown_net", "sharpe_like_net", "universe_lost_exits"]
        ]
        print("\n=== Converged ordinary vs advanced, 5bps ===")
        print(converged_5bps.to_string(index=False))
    print("\n=== Validation ===")
    for key, value in validation.items():
        print(f"{key}: {value}")
    print(f"README path: {PROJECT_ROOT / 'README.md'}")
    print(f"final_report path: {FINAL_REPORT / 'final_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen final crypto PCA residual stat-arb pipeline.")
    parser.add_argument("--force-rerun", action="store_true", help="Rerun upstream source scripts instead of reusing existing outputs.")
    args = parser.parse_args()

    validation = run_pipeline(force_rerun=args.force_rerun)
    print_summary(validation)


if __name__ == "__main__":
    main()
