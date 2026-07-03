from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover - handled at runtime
    minimize = None
    pearsonr = None
    spearmanr = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PRICE_FILE = PROJECT_ROOT / "data" / "raw" / "coin_all_prices_full.csv"
UNIVERSE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "coin_universe_150K_40_valid_price_filtered_daily_lastbar_nolookahead_expanded_hourly.csv"
)
OUT_DATA = PROJECT_ROOT / "data" / "processed" / "research_pca" / "residual_comovement_penalized_pca"
OUT_REPORT = PROJECT_ROOT / "reports" / "research_pca" / "residual_comovement_penalized_pca"
FIG_DIR = OUT_REPORT / "figures"

WINDOW = 360
K_FACTORS = 3
M_CANDIDATE_PCS = 8
MIN_ELIGIBLE = 20
DEFAULT_STRIDE_HOURS = 24
DEFAULT_LAMBDAS = [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
PLOT_LAMBDAS = [0.5, 2, 10, 50]
RANDOM_SEED = 20260628
DEFAULT_MAXITER = 75


def log(message: str) -> None:
    print(f"[residual_comovement_pca] {message}", flush=True)


def ensure_dirs() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def lambda_label(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "ordinary"
    text = f"{value:g}".replace(".", "p").replace("-", "m")
    return text


def method_name(value: float | None) -> str:
    if value is None:
        return "ordinary_pca_k3"
    return f"advanced_lambda_{lambda_label(value)}"


def load_price_panel() -> pd.DataFrame:
    df = pd.read_csv(RAW_PRICE_FILE)
    time_col = "startTime" if "startTime" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    drop_cols = [c for c in ["time", "date", "datetime", "timestamp"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df.apply(pd.to_numeric, errors="coerce")


def load_universe_panel() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(UNIVERSE_FILE)
    time_col = "startTime" if "startTime" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    rank_cols = [c for c in df.columns if str(c).isdigit()]
    if not rank_cols:
        rank_cols = list(df.columns)
    return df, rank_cols


def active_universe_from_row(row: pd.Series, price_cols: set[str]) -> list[str]:
    tickers: list[str] = []
    for value in row.dropna().tolist():
        ticker = str(value).strip()
        if ticker and ticker in price_cols and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def standardize_window(ret_window: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    complete = ret_window.notna().sum(axis=0).eq(len(ret_window))
    std = ret_window.std(axis=0, ddof=1).replace([np.inf, -np.inf], np.nan)
    eligible = [c for c in ret_window.columns if bool(complete.get(c, False) and std.get(c, np.nan) > 0)]
    x = ret_window[eligible].to_numpy(dtype=float)
    y = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    return y, eligible


@dataclass
class PCABasis:
    eigvals: np.ndarray
    eigvecs: np.ndarray
    evr: np.ndarray


def pca_basis(y: np.ndarray) -> PCABasis:
    cov = np.cov(y, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    total = float(eigvals.sum())
    evr = eigvals / total if total > 0 else np.full_like(eigvals, np.nan)
    return PCABasis(eigvals=eigvals, eigvecs=eigvecs, evr=evr)


def residual_from_loadings(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    return y - (y @ q @ q.T)


def reconstruction_loss(y: np.ndarray, residual: np.ndarray) -> float:
    denom = float(np.sum(y * y))
    return float(np.sum(residual * residual) / denom) if denom > 0 else np.nan


def residual_cov_pc1_evr(residual: np.ndarray, original_total_variance: float) -> float:
    if residual.shape[1] < 1 or original_total_variance <= 0:
        return np.nan
    cov = np.cov(residual, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    return float(max(eigvals.max(), 0.0) / original_total_variance)


def residual_corr_matrix(residual: np.ndarray) -> np.ndarray | None:
    std = residual.std(axis=0, ddof=1)
    valid = np.isfinite(std) & (std > 1e-12)
    if int(valid.sum()) < 2:
        return None
    z = residual[:, valid]
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=1)
    corr = np.corrcoef(z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def residual_corr_pc1_evr(residual: np.ndarray) -> float:
    corr = residual_corr_matrix(residual)
    if corr is None:
        return np.nan
    eigvals = np.linalg.eigvalsh(corr)
    return float(max(eigvals.max(), 0.0) / corr.shape[0])


def avg_abs_residual_corr(residual: np.ndarray) -> float:
    corr = residual_corr_matrix(residual)
    if corr is None or corr.shape[0] < 2:
        return np.nan
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.mean(np.abs(corr[mask])))


def residual_effective_rank(residual: np.ndarray) -> float:
    corr = residual_corr_matrix(residual)
    if corr is None:
        return np.nan
    eigvals = np.maximum(np.linalg.eigvalsh(corr), 0.0)
    total = float(eigvals.sum())
    if total <= 0:
        return np.nan
    p = eigvals[eigvals > 0] / total
    return float(np.exp(-np.sum(p * np.log(p))))


def loading_concentration_herfindahl(q: np.ndarray) -> float:
    if q.size == 0:
        return np.nan
    return float(np.mean(np.sum(q**4, axis=0)))


def factor_metrics(y: np.ndarray, q: np.ndarray, basis: PCABasis) -> dict[str, float]:
    residual = residual_from_loadings(y, q)
    total_variance = float(basis.eigvals.sum())
    return {
        "reconstruction_loss": reconstruction_loss(y, residual),
        "residual_pc1_evr_corr": residual_corr_pc1_evr(residual),
        "residual_pc1_evr_cov": residual_cov_pc1_evr(residual, total_variance),
        "avg_abs_residual_corr": avg_abs_residual_corr(residual),
        "residual_effective_rank": residual_effective_rank(residual),
        "loading_concentration_herfindahl": loading_concentration_herfindahl(q),
    }


def canonical_a(m: int, k: int) -> np.ndarray:
    a = np.zeros((m, k))
    a[:k, :k] = np.eye(k)
    return a


def orthonormalize(z_flat: np.ndarray, m: int, k: int) -> np.ndarray:
    z = z_flat.reshape(m, k)
    qr, r = np.linalg.qr(z)
    signs = np.sign(np.diag(r)[:k])
    signs[signs == 0] = 1.0
    return qr[:, :k] * signs


def optimize_advanced_pca(
    y: np.ndarray,
    basis: PCABasis,
    m: int,
    k: int,
    lambda_penalty: float,
    rng: np.random.Generator,
    random_starts: int,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, object]]:
    v_m = basis.eigvecs[:, :m]
    a0 = canonical_a(m, k)
    baseline_q = v_m @ a0
    baseline_metrics = factor_metrics(y, baseline_q, basis)
    baseline_objective = baseline_metrics["reconstruction_loss"] + lambda_penalty * baseline_metrics["residual_pc1_evr_corr"]

    if lambda_penalty == 0 or minimize is None:
        return a0, {
            "optimizer_success": bool(lambda_penalty == 0),
            "optimizer_objective_value": baseline_objective,
            "optimizer_n_iter": 0,
            "fallback_flag": bool(minimize is None and lambda_penalty != 0),
            "fallback_reason": "scipy_unavailable" if minimize is None and lambda_penalty != 0 else "",
        }

    best_a = a0
    best_fun = float(baseline_objective)
    best_success = False
    best_nit = 0

    def objective(z_flat: np.ndarray) -> float:
        a = orthonormalize(z_flat, m, k)
        q = v_m @ a
        residual = residual_from_loadings(y, q)
        rec = reconstruction_loss(y, residual)
        penalty = residual_corr_pc1_evr(residual)
        if not np.isfinite(rec) or not np.isfinite(penalty):
            return 1e6
        return float(rec + lambda_penalty * penalty)

    starts = [a0 + 1e-4 * rng.standard_normal((m, k))]
    for _ in range(random_starts):
        starts.append(rng.standard_normal((m, k)))

    for start in starts:
        try:
            res = minimize(
                objective,
                start.ravel(),
                method="BFGS",
                options={"maxiter": maxiter, "gtol": 1e-5},
            )
        except Exception:
            continue
        if np.isfinite(res.fun) and float(res.fun) < best_fun:
            best_fun = float(res.fun)
            best_a = orthonormalize(res.x, m, k)
            best_success = bool(res.success or float(res.fun) <= baseline_objective)
            best_nit = int(getattr(res, "nit", 0))

    fallback = not np.isfinite(best_fun)
    if fallback:
        best_a = a0
        best_fun = float(baseline_objective)
    return best_a, {
        "optimizer_success": best_success,
        "optimizer_objective_value": best_fun,
        "optimizer_n_iter": best_nit,
        "fallback_flag": fallback,
        "fallback_reason": "optimizer_failed_all_starts" if fallback else "",
    }


def selected_positions(n: int, window: int, stride_hours: int, full_run: bool, max_timestamps: int | None) -> list[int]:
    positions = list(range(window, n))
    if not full_run:
        positions = positions[::stride_hours]
    if max_timestamps is not None:
        positions = positions[:max_timestamps]
    return positions


def row_common(ts: pd.Timestamp, basis: PCABasis, n_assets: int, m: int) -> dict[str, object]:
    return {
        "timestamp": ts,
        "n_assets": n_assets,
        "window": WINDOW,
        "K": K_FACTORS,
        "M": m,
        "ordinary_pc1_evr": float(basis.evr[0]) if len(basis.evr) > 0 else np.nan,
        "ordinary_pc2_evr": float(basis.evr[1]) if len(basis.evr) > 1 else np.nan,
        "ordinary_pc3_evr": float(basis.evr[2]) if len(basis.evr) > 2 else np.nan,
        "ordinary_pc4_evr": float(basis.evr[3]) if len(basis.evr) > 3 else np.nan,
        "ordinary_pc1_3_cum_evr": float(np.nansum(basis.evr[:3])),
    }


def build_outputs(args: argparse.Namespace) -> dict[str, object]:
    ensure_dirs()
    rng = np.random.default_rng(args.seed)

    log("loading price and universe panels")
    prices = load_price_panel()
    universe, rank_cols = load_universe_panel()
    common_index = prices.index.intersection(universe.index)
    prices = prices.reindex(common_index)
    universe = universe.reindex(common_index)
    returns = np.log(prices.where(prices > 0)).diff()
    price_cols = set(prices.columns)

    positions = selected_positions(len(common_index), WINDOW, args.stride_hours, args.full_run, args.max_timestamps)
    log(f"evaluating {len(positions)} timestamps")

    rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    random_starts = args.random_starts if args.random_starts is not None else (5 if args.full_run else 0)

    for counter, pos in enumerate(positions, start=1):
        if counter == 1 or counter % 25 == 0:
            log(f"processed {counter}/{len(positions)} diagnostic timestamps")
        ts = common_index[pos]
        candidates = active_universe_from_row(universe.loc[ts, rank_cols], price_cols)
        if len(candidates) < MIN_ELIGIBLE:
            skipped_rows.append({"timestamp": ts, "reason": "insufficient_candidates", "candidate_tickers": len(candidates), "eligible_tickers": 0})
            continue
        ret_window = returns.iloc[pos - WINDOW : pos][candidates]
        y, eligible = standardize_window(ret_window)
        if len(eligible) < MIN_ELIGIBLE:
            skipped_rows.append({"timestamp": ts, "reason": "insufficient_complete_window", "candidate_tickers": len(candidates), "eligible_tickers": len(eligible)})
            continue

        basis = pca_basis(y)
        m = min(args.m_components, len(eligible) - 1, len(basis.eigvals))
        if m < K_FACTORS:
            skipped_rows.append({"timestamp": ts, "reason": "insufficient_m_components", "candidate_tickers": len(candidates), "eligible_tickers": len(eligible)})
            continue
        common = row_common(ts, basis, len(eligible), m)

        q_ordinary = basis.eigvecs[:, :K_FACTORS]
        ordinary_metrics = factor_metrics(y, q_ordinary, basis)
        rows.append(
            {
                **common,
                "method": "ordinary_pca_k3",
                "lambda_penalty": np.nan,
                **ordinary_metrics,
                "optimizer_success": True,
                "optimizer_objective_value": ordinary_metrics["reconstruction_loss"],
                "optimizer_n_iter": 0,
                "fallback_flag": False,
                "fallback_reason": "",
            }
        )
        sanity_rows.append(
            {
                "timestamp": ts,
                "ordinary_pc4_evr": common["ordinary_pc4_evr"],
                "residual_pc1_evr_cov_after_pc3": ordinary_metrics["residual_pc1_evr_cov"],
                "residual_pc1_evr_corr_after_pc3": ordinary_metrics["residual_pc1_evr_corr"],
                "abs_diff_cov_vs_pc4": abs(float(common["ordinary_pc4_evr"]) - ordinary_metrics["residual_pc1_evr_cov"]),
                "ratio_cov_vs_pc4": ordinary_metrics["residual_pc1_evr_cov"] / float(common["ordinary_pc4_evr"]) if float(common["ordinary_pc4_evr"]) > 0 else np.nan,
            }
        )

        for lambda_penalty in args.lambda_penalty:
            a, opt = optimize_advanced_pca(
                y=y,
                basis=basis,
                m=m,
                k=K_FACTORS,
                lambda_penalty=float(lambda_penalty),
                rng=rng,
                random_starts=random_starts,
                maxiter=args.maxiter,
            )
            q = basis.eigvecs[:, :m] @ a
            metrics = factor_metrics(y, q, basis)
            objective_value = metrics["reconstruction_loss"] + float(lambda_penalty) * metrics["residual_pc1_evr_corr"]
            if np.isfinite(objective_value):
                opt["optimizer_objective_value"] = objective_value
            rows.append(
                {
                    **common,
                    "method": method_name(float(lambda_penalty)),
                    "lambda_penalty": float(lambda_penalty),
                    **metrics,
                    **opt,
                }
            )

    ts_df = pd.DataFrame(rows)
    sanity = pd.DataFrame(sanity_rows)
    skipped = pd.DataFrame(skipped_rows)

    ts_path = OUT_DATA / "pca_residual_comovement_timeseries.csv"
    sanity_path = OUT_REPORT / "ordinary_pca_residual_pc4_sanity.csv"
    skipped_path = OUT_DATA / "skipped_timestamps.csv"
    ts_df.to_csv(ts_path, index=False)
    sanity.to_csv(sanity_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    summary = build_summary(ts_df)
    summary_path = OUT_REPORT / "pca_residual_comovement_summary.csv"
    summary.to_csv(summary_path, index=False)

    make_figures(ts_df, sanity, summary)
    write_report(ts_df, sanity, summary, args)
    return validate(ts_df, sanity, summary)


def build_summary(ts_df: pd.DataFrame) -> pd.DataFrame:
    if ts_df.empty:
        return pd.DataFrame()
    grouped = ts_df.groupby("method", dropna=False)
    summary = grouped.agg(
        mean_residual_pc1_evr_corr=("residual_pc1_evr_corr", "mean"),
        median_residual_pc1_evr_corr=("residual_pc1_evr_corr", "median"),
        q25_residual_pc1_evr_corr=("residual_pc1_evr_corr", lambda s: s.quantile(0.25)),
        q75_residual_pc1_evr_corr=("residual_pc1_evr_corr", lambda s: s.quantile(0.75)),
        mean_avg_abs_residual_corr=("avg_abs_residual_corr", "mean"),
        mean_effective_rank=("residual_effective_rank", "mean"),
        mean_reconstruction_loss=("reconstruction_loss", "mean"),
        mean_objective=("optimizer_objective_value", "mean"),
        optimizer_success_rate=("optimizer_success", "mean"),
        fallback_rate=("fallback_flag", "mean"),
        mean_n_assets=("n_assets", "mean"),
        timestamps=("timestamp", "count"),
    ).reset_index()
    lambda_map = ts_df.groupby("method")["lambda_penalty"].first().to_dict()
    summary.insert(1, "lambda_penalty", summary["method"].map(lambda_map))
    return summary.sort_values(["lambda_penalty", "method"], na_position="first")


def corr_pair(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 3:
        return np.nan, np.nan
    if pearsonr is None or spearmanr is None:
        return float(x[mask].corr(y[mask], method="pearson")), float(x[mask].corr(y[mask], method="spearman"))
    return float(pearsonr(x[mask], y[mask]).statistic), float(spearmanr(x[mask], y[mask]).statistic)


def make_figures(ts_df: pd.DataFrame, sanity: pd.DataFrame, summary: pd.DataFrame) -> None:
    if ts_df.empty:
        return
    plot_ts = ts_df.copy()
    plot_ts["timestamp"] = pd.to_datetime(plot_ts["timestamp"], utc=True)
    if not sanity.empty:
        sanity_plot = sanity.copy()
        sanity_plot["timestamp"] = pd.to_datetime(sanity_plot["timestamp"], utc=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sanity_plot["timestamp"], sanity_plot["ordinary_pc4_evr"], label="Ordinary PC4 EVR", linewidth=1.0)
        ax.plot(sanity_plot["timestamp"], sanity_plot["residual_pc1_evr_cov_after_pc3"], label="Residual PC1 EVR, cov", linewidth=1.0)
        ax.plot(sanity_plot["timestamp"], sanity_plot["residual_pc1_evr_corr_after_pc3"], label="Residual PC1 EVR, corr", linewidth=1.0)
        ax.set_title("Ordinary PC4 vs residual PC1 explained variance")
        ax.set_ylabel("Explained variance ratio")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "ordinary_pc4_vs_residual_pc1_evr_timeseries.png", dpi=160)
        plt.close(fig)

        pearson, spearman = corr_pair(sanity_plot["ordinary_pc4_evr"], sanity_plot["residual_pc1_evr_cov_after_pc3"])
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sanity_plot["ordinary_pc4_evr"], sanity_plot["residual_pc1_evr_cov_after_pc3"], s=12, alpha=0.65)
        ax.set_title(f"PC4 vs residual PC1 cov EVR\nPearson={pearson:.3f}, Spearman={spearman:.3f}")
        ax.set_xlabel("Ordinary PC4 EVR")
        ax.set_ylabel("Residual PC1 EVR, cov")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "ordinary_pc4_vs_residual_pc1_scatter.png", dpi=160)
        plt.close(fig)

    keep_methods = ["ordinary_pca_k3"] + [method_name(x) for x in PLOT_LAMBDAS]
    sub = plot_ts[plot_ts["method"].isin(keep_methods)].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    for method, group in sub.groupby("method"):
        ax.plot(group["timestamp"], group["residual_pc1_evr_corr"], label=method.replace("advanced_lambda_", "lambda="), linewidth=1.0)
    ax.set_title("Residual PC1 EVR by lambda")
    ax.set_ylabel("Residual PC1 EVR, corr")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residual_pc1_evr_by_lambda_timeseries.png", dpi=160)
    plt.close(fig)

    advanced = plot_ts[plot_ts["method"].str.startswith("advanced_lambda_")].copy()
    if not advanced.empty:
        advanced["lambda_label"] = advanced["lambda_penalty"].map(lambda_label)
        labels = [lambda_label(x) for x in sorted(advanced["lambda_penalty"].dropna().unique())]
        data = [advanced.loc[advanced["lambda_label"] == label, "residual_pc1_evr_corr"].dropna().to_numpy() for label in labels]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title("Residual PC1 EVR distribution by lambda")
        ax.set_xlabel("lambda")
        ax.set_ylabel("Residual PC1 EVR, corr")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "residual_pc1_evr_by_lambda_boxplot.png", dpi=160)
        plt.close(fig)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(summary["mean_reconstruction_loss"], summary["mean_residual_pc1_evr_corr"], s=45)
        for _, row in summary.iterrows():
            label = "ordinary" if row["method"] == "ordinary_pca_k3" else f"{row['lambda_penalty']:g}"
            ax.annotate(label, (row["mean_reconstruction_loss"], row["mean_residual_pc1_evr_corr"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_title("Reconstruction vs residual comovement tradeoff")
        ax.set_xlabel("Mean reconstruction loss")
        ax.set_ylabel("Mean residual PC1 EVR, corr")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "reconstruction_loss_vs_residual_comovement_tradeoff.png", dpi=160)
        plt.close(fig)

        metric_specs = [
            ("mean_avg_abs_residual_corr", "avg_abs_residual_corr_by_lambda.png", "Mean absolute residual correlation"),
            ("mean_effective_rank", "effective_rank_by_lambda.png", "Mean residual effective rank"),
            ("optimizer_success_rate", "optimizer_success_by_lambda.png", "Optimizer success rate"),
        ]
        for metric, filename, title in metric_specs:
            fig, ax = plt.subplots(figsize=(9, 4))
            labels = summary["method"].str.replace("advanced_lambda_", "lambda=", regex=False).str.replace("ordinary_pca_k3", "ordinary", regex=False)
            ax.bar(labels, summary[metric])
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIG_DIR / filename, dpi=160)
            plt.close(fig)


def write_report(ts_df: pd.DataFrame, sanity: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    pearson = spearman = np.nan
    if not sanity.empty:
        pearson, spearman = corr_pair(sanity["ordinary_pc4_evr"], sanity["residual_pc1_evr_cov_after_pc3"])
    lambda0_check = {}
    ordinary = ts_df[ts_df["method"] == "ordinary_pca_k3"].set_index("timestamp")
    lambda0 = ts_df[ts_df["method"] == "advanced_lambda_0"].set_index("timestamp")
    common_idx = ordinary.index.intersection(lambda0.index)
    if len(common_idx):
        lambda0_check = {
            "mean_abs_diff_residual_pc1_evr_corr": float((ordinary.loc[common_idx, "residual_pc1_evr_corr"] - lambda0.loc[common_idx, "residual_pc1_evr_corr"]).abs().mean()),
            "mean_abs_diff_reconstruction_loss": float((ordinary.loc[common_idx, "reconstruction_loss"] - lambda0.loc[common_idx, "reconstruction_loss"]).abs().mean()),
        }
    else:
        lambda0_check = {"mean_abs_diff_residual_pc1_evr_corr": np.nan, "mean_abs_diff_reconstruction_loss": np.nan}

    def md_table(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "_No rows generated._"
        return df[cols].to_markdown(index=False, floatfmt=".4f")

    lines = [
        "# Residual-Comovement-Penalized PCA Diagnostic",
        "",
        "## 1. Motivation",
        "",
        "Ordinary PCA ranks factors by explained variance, but a statistical-arbitrage residual can still contain shared movement after removing PC1-PC3. This diagnostic measures that leftover common movement by running PCA on the residual return matrix and by penalizing the first residual correlation eigenmode.",
        "",
        "## 2. Ordinary PCA Residual PC1 Diagnostic",
        "",
        "For each timestamp, the script uses only the rolling return window `[t-W, t-1]` and the no-lookahead universe row at `t`. Returns are standardized by ticker, ordinary PCA is fit, PC1-PC3 are projected out, and the residual matrix is diagnosed with covariance and correlation PCA.",
        "",
        f"- Window: `{WINDOW}` hourly bars",
        f"- Timestamps evaluated: `{ts_df['timestamp'].nunique() if not ts_df.empty else 0}`",
        f"- Average assets: `{ts_df.drop_duplicates(['timestamp'])['n_assets'].mean() if not ts_df.empty else np.nan:.2f}`",
        f"- PC4 vs residual cov PC1 Pearson/Spearman: `{pearson:.4f}` / `{spearman:.4f}`",
        "",
        "![PC4 timeseries](figures/ordinary_pc4_vs_residual_pc1_evr_timeseries.png)",
        "",
        "![PC4 scatter](figures/ordinary_pc4_vs_residual_pc1_scatter.png)",
        "",
        "## 3. Advanced PCA Objective",
        "",
        "The prototype searches inside the ordinary PCA span PC1-PCM for K orthonormal factors and minimizes:",
        "",
        "`reconstruction_loss + lambda * residual_pc1_explained_variance_corr`",
        "",
        "The reconstruction term keeps the selected subspace close to ordinary PCA explanatory power. The penalty term discourages a dominant residual correlation mode.",
        "",
        "## 4. Results Across Lambda",
        "",
        md_table(
            summary,
            [
                "method",
                "lambda_penalty",
                "mean_residual_pc1_evr_corr",
                "mean_reconstruction_loss",
                "mean_avg_abs_residual_corr",
                "mean_effective_rank",
                "optimizer_success_rate",
                "fallback_rate",
            ],
        ),
        "",
        "![Residual PC1 by lambda](figures/residual_pc1_evr_by_lambda_timeseries.png)",
        "",
        "![Residual PC1 boxplot](figures/residual_pc1_evr_by_lambda_boxplot.png)",
        "",
        "![Tradeoff](figures/reconstruction_loss_vs_residual_comovement_tradeoff.png)",
        "",
        "![Average residual corr](figures/avg_abs_residual_corr_by_lambda.png)",
        "",
        "![Effective rank](figures/effective_rank_by_lambda.png)",
        "",
        "![Optimizer success](figures/optimizer_success_by_lambda.png)",
        "",
        "## 5. Interpretation",
        "",
        "If higher lambda values lower residual PC1 EVR without a large reconstruction-loss increase, the ordinary PC1-PC3 span is leaving a removable residual common mode. If the tradeoff is steep, the residual comovement may be part of the same variance structure that ordinary PCA is already capturing efficiently.",
        "",
        "## 6. Caveats",
        "",
        "- This is a diagnostic and factor-extraction prototype, not a replacement for the final trading pipeline.",
        "- The optimizer searches within the ordinary PC1-PCM span, so it is not a fully general constrained PCA estimator.",
        "- QR parameterization has rotation and sign ambiguity; only subspace-level residual metrics should be interpreted.",
        "- Same rolling-window and no-lookahead discipline is used, but no walk-forward trading validation is performed here.",
        "",
        "## 7. Validation",
        "",
        f"- Lambda 0 mean absolute residual-PC1 difference vs ordinary PCA: `{lambda0_check['mean_abs_diff_residual_pc1_evr_corr']:.8f}`",
        f"- Lambda 0 mean absolute reconstruction-loss difference vs ordinary PCA: `{lambda0_check['mean_abs_diff_reconstruction_loss']:.8f}`",
        f"- No NaN in summary numeric metrics: `{summary.select_dtypes(include=[np.number]).notna().all().all() if not summary.empty else False}`",
        f"- Full run: `{args.full_run}`; stride hours: `{args.stride_hours}`; random starts: `{args.random_starts if args.random_starts is not None else (5 if args.full_run else 0)}`",
    ]
    (OUT_REPORT / "residual_comovement_penalized_pca_report.md").write_text("\n".join(lines), encoding="utf-8")


def validate(ts_df: pd.DataFrame, sanity: pd.DataFrame, summary: pd.DataFrame) -> dict[str, object]:
    ordinary = ts_df[ts_df["method"] == "ordinary_pca_k3"].set_index("timestamp")
    lambda0 = ts_df[ts_df["method"] == "advanced_lambda_0"].set_index("timestamp")
    common_idx = ordinary.index.intersection(lambda0.index)
    lambda0_residual_diff = np.nan
    lambda0_recon_diff = np.nan
    if len(common_idx):
        lambda0_residual_diff = float((ordinary.loc[common_idx, "residual_pc1_evr_corr"] - lambda0.loc[common_idx, "residual_pc1_evr_corr"]).abs().mean())
        lambda0_recon_diff = float((ordinary.loc[common_idx, "reconstruction_loss"] - lambda0.loc[common_idx, "reconstruction_loss"]).abs().mean())
    pearson = spearman = np.nan
    if not sanity.empty:
        pearson, spearman = corr_pair(sanity["ordinary_pc4_evr"], sanity["residual_pc1_evr_cov_after_pc3"])
    numeric_summary = summary.select_dtypes(include=[np.number]).drop(columns=["lambda_penalty"], errors="ignore")
    no_nan_summary = bool(numeric_summary.notna().all().all()) if not numeric_summary.empty else False
    return {
        "input_price_file": str(RAW_PRICE_FILE),
        "input_universe_file": str(UNIVERSE_FILE),
        "timestamps_evaluated": int(ts_df["timestamp"].nunique()) if not ts_df.empty else 0,
        "average_n_assets": float(ts_df.drop_duplicates(["timestamp"])["n_assets"].mean()) if not ts_df.empty else math.nan,
        "pc4_residual_cov_pearson": pearson,
        "pc4_residual_cov_spearman": spearman,
        "lambda0_mean_abs_residual_pc1_diff": lambda0_residual_diff,
        "lambda0_mean_abs_reconstruction_loss_diff": lambda0_recon_diff,
        "summary_no_nan_numeric": no_nan_summary,
        "figures_count": len(list(FIG_DIR.glob("*.png"))),
        "report_path": str(OUT_REPORT / "residual_comovement_penalized_pca_report.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run residual-comovement-penalized PCA diagnostics.")
    parser.add_argument("--full-run", action="store_true", help="Evaluate every eligible hourly timestamp instead of stride sampling.")
    parser.add_argument("--stride-hours", type=int, default=DEFAULT_STRIDE_HOURS, help="Stride used when --full-run is not set.")
    parser.add_argument("--max-timestamps", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--m-components", type=int, default=M_CANDIDATE_PCS, help="Candidate ordinary PCA span size M.")
    parser.add_argument("--random-starts", type=int, default=None, help="Random optimizer starts per lambda; defaults to 0 or 5 for full-run.")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER, help="Maximum BFGS iterations per optimizer start.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for optimizer starts.")
    parser.add_argument("--lambda-penalty", type=float, nargs="+", default=DEFAULT_LAMBDAS, help="Lambda penalty values to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride_hours < 1:
        raise ValueError("--stride-hours must be >= 1")
    if args.m_components < K_FACTORS:
        raise ValueError("--m-components must be >= K")
    validation = build_outputs(args)
    print("\n=== Residual-comovement PCA validation ===")
    for key, value in validation.items():
        print(f"{key}: {value}")
    print(f"timeseries: {OUT_DATA / 'pca_residual_comovement_timeseries.csv'}")
    print(f"summary: {OUT_REPORT / 'pca_residual_comovement_summary.csv'}")
    print(f"report: {OUT_REPORT / 'residual_comovement_penalized_pca_report.md'}")


if __name__ == "__main__":
    main()
