from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "scripts" / "research_pca"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

import run_advanced_pca_strategy_comparison as research  # noqa: E402
from run_residual_comovement_penalized_pca import (  # noqa: E402
    canonical_a,
    factor_metrics,
    orthonormalize,
    reconstruction_loss,
    residual_corr_pc1_evr,
    residual_from_loadings,
)


CONVERGED_PATH = PROJECT_ROOT / "scripts" / "final_pipeline" / "09_run_converged_mainline.py"
spec = importlib.util.spec_from_file_location("converged_mainline", CONVERGED_PATH)
converged = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converged
spec.loader.exec_module(converged)  # type: ignore[union-attr]


OUT_DIR = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "advanced_pca_lambda_full_grid"
REPORT_DIR = PROJECT_ROOT / "reports" / "final_report" / "advanced_pca_lambda_full_grid"
LAMBDA_GRID = [0.3, 0.5, 0.7, 1.0]
PORTFOLIO_LAMBDA = 3.0
PCA_LABEL = {0.3: "advanced_lambda_0p3", 0.5: "advanced_lambda_0p5", 0.7: "advanced_lambda_0p7", 1.0: "advanced_lambda_1"}


def log(message: str) -> None:
    print(f"[advanced_pca_lambda_full_grid] {message}", flush=True)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(x))
    except TypeError:
        return False


def lambda_label(value: float) -> str:
    return PCA_LABEL[value]


def optimize_with_start(
    y: np.ndarray,
    basis: Any,
    m: int,
    k: int,
    lambda_penalty: float,
    start_a: np.ndarray | None,
    rng: np.random.Generator,
    random_starts: int,
    maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    v_m = basis.eigvecs[:, :m]
    a0 = canonical_a(m, k)
    baseline_q = v_m @ a0
    baseline_metrics = factor_metrics(y, baseline_q, basis)
    baseline_objective = baseline_metrics["reconstruction_loss"] + lambda_penalty * baseline_metrics["residual_pc1_evr_corr"]
    if lambda_penalty == 0:
        return a0, {
            "optimizer_success": True,
            "optimizer_objective_value": float(baseline_objective),
            "optimizer_n_iter": 0,
            "warm_start_used": False,
        }

    starts: list[np.ndarray] = []
    if start_a is not None and start_a.shape == (m, k):
        starts.append(start_a)
    else:
        starts.append(a0 + 1e-4 * rng.standard_normal((m, k)))
    for _ in range(random_starts):
        starts.append(rng.standard_normal((m, k)))

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

    return best_a, {
        "optimizer_success": best_success,
        "optimizer_objective_value": best_fun,
        "optimizer_n_iter": best_nit,
        "warm_start_used": bool(start_a is not None and start_a.shape == (m, k)),
    }


def compute_timestamp(
    pos: int,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    rank_cols: list[str],
    price_cols: set[str],
    args: argparse.Namespace,
    warm_starts: dict[float, np.ndarray | None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ts = prices.index[pos]
    candidates = research.active_universe_from_row(universe.loc[ts, rank_cols], price_cols)
    if len(candidates) < args.min_assets:
        return [], {"timestamp": ts, "skip_reason": "insufficient_candidates", "candidate_tickers": len(candidates), "eligible_tickers": 0}

    ret_window = returns.iloc[pos - research.WINDOW : pos][candidates]
    y, eligible = research.standardize_window(ret_window)
    if len(eligible) < args.min_assets:
        return [], {"timestamp": ts, "skip_reason": "insufficient_complete_window", "candidate_tickers": len(candidates), "eligible_tickers": len(eligible)}

    current_ret = returns.iloc[pos][eligible]
    current_price = prices.iloc[pos][eligible]
    good_now = current_ret.notna() & current_price.gt(0)
    if int(good_now.sum()) < args.min_assets:
        return [], {"timestamp": ts, "skip_reason": "insufficient_current_valid", "candidate_tickers": len(candidates), "eligible_tickers": int(good_now.sum())}

    eligible_now = [c for c in eligible if bool(good_now.get(c, False))]
    ret_window_now = ret_window[eligible_now]
    y_fit, eligible_now_check = research.standardize_window(ret_window_now)
    if eligible_now_check != eligible_now:
        return [], {"timestamp": ts, "skip_reason": "eligible_recheck_mismatch", "candidate_tickers": len(candidates), "eligible_tickers": len(eligible_now)}

    basis = research.pca_basis(y_fit)
    m = min(args.m_components, len(eligible_now) - 1, len(basis.eigvals))
    if m < research.K_FACTORS:
        return [], {"timestamp": ts, "skip_reason": "insufficient_m_components", "candidate_tickers": len(candidates), "eligible_tickers": len(eligible_now)}

    mean = ret_window_now.mean(axis=0)
    std = ret_window_now.std(axis=0, ddof=1)
    y_current = ((current_ret[eligible_now] - mean) / std).to_numpy(dtype=float)
    vol = std.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "timestamp": ts,
        "skip_reason": "",
        "candidate_tickers": len(candidates),
        "eligible_tickers": len(eligible_now),
        "m_components": m,
    }

    for lambda_value in LAMBDA_GRID:
        rng = np.random.default_rng(args.seed + int(pos) * 1000 + int(lambda_value * 100))
        start_a = warm_starts.get(lambda_value) if warm_starts is not None else None
        a, opt = optimize_with_start(
            y=y_fit,
            basis=basis,
            m=m,
            k=research.K_FACTORS,
            lambda_penalty=lambda_value,
            start_a=start_a,
            rng=rng,
            random_starts=args.random_starts,
            maxiter=args.maxiter,
        )
        if warm_starts is not None:
            warm_starts[lambda_value] = a
        q = basis.eigvecs[:, :m] @ a
        y_resid = y_current - (y_current @ q @ q.T)
        beta = q[:, 0] / np.where(vol > 0, vol, np.nan)
        beta = (beta - np.nanmean(beta)) / np.nanstd(beta) if np.nanstd(beta) > 0 else beta * 0.0
        method = lambda_label(lambda_value)
        meta[f"{method}_optimizer_success"] = bool(opt.get("optimizer_success", False))
        meta[f"{method}_optimizer_objective_value"] = opt.get("optimizer_objective_value", np.nan)
        for ticker, raw_ret, price, resid, beta_i in zip(
            eligible_now,
            current_ret[eligible_now].to_numpy(dtype=float),
            current_price[eligible_now].to_numpy(dtype=float),
            y_resid,
            beta,
        ):
            rows.append(
                {
                    "timestamp": ts,
                    "method": method,
                    "ticker": ticker,
                    "raw_log_return": float(raw_ret),
                    "price": float(price),
                    "residual_return": float(resid),
                    "beta_factor1": float(beta_i),
                }
            )
    return rows, meta


def compute_block(
    block_positions: list[int],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    rank_cols: list[str],
    price_cols: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warm_starts: dict[float, np.ndarray | None] = {x: None for x in LAMBDA_GRID}
    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for pos in block_positions:
        r, m = compute_timestamp(pos, prices, returns, universe, rank_cols, price_cols, args, warm_starts=warm_starts)
        rows.extend(r)
        meta_rows.append(m)
    return rows, meta_rows


def build_residual_panel(args: argparse.Namespace) -> pd.DataFrame:
    residual_path = OUT_DIR / "advanced_lambda_grid_residual_returns_long.csv"
    meta_path = OUT_DIR / "advanced_lambda_grid_pca_metadata.csv"
    if args.reuse and residual_path.exists():
        return pd.read_csv(residual_path, parse_dates=["timestamp"], low_memory=False)

    prices = research.load_price_panel()
    universe, rank_cols = research.load_universe_panel()
    common_index = prices.index.intersection(universe.index)
    prices = prices.reindex(common_index)
    universe = universe.reindex(common_index)
    returns = np.log(prices.where(prices > 0)).diff()
    price_cols = set(prices.columns)

    stop = len(common_index) if args.max_hours is None else min(len(common_index), research.WINDOW + args.max_hours)
    positions = list(range(research.WINDOW, stop))
    blocks = [positions[i : i + args.block_size] for i in range(0, len(positions), args.block_size)]
    log(f"parallel warm-start PCA grid timestamps={len(positions)} blocks={len(blocks)} block_size={args.block_size} n_jobs={args.n_jobs} lambdas={LAMBDA_GRID}")
    results = Parallel(n_jobs=args.n_jobs, backend="loky", batch_size=args.batch_size, verbose=10)(
        delayed(compute_block)(block, prices, returns, universe, rank_cols, price_cols, args) for block in blocks
    )
    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for r, m in results:
        rows.extend(r)
        meta_rows.extend(m)
    residual = pd.DataFrame(rows)
    metadata = pd.DataFrame(meta_rows)
    residual.to_csv(residual_path, index=False)
    metadata.to_csv(meta_path, index=False)
    return residual


def build_signal_panel(residual: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    signal_path = OUT_DIR / "advanced_lambda_grid_signal_panel.csv"
    if args.reuse and signal_path.exists():
        return pd.read_csv(signal_path, parse_dates=["timestamp"], low_memory=False)
    old_out_data = research.OUT_DATA
    old_out_report = research.OUT_REPORT
    old_thresholds = research.signal_thresholds
    try:
        research.OUT_DATA = OUT_DIR
        research.OUT_REPORT = REPORT_DIR
        research.signal_thresholds = converged.fixed_signal_thresholds
        signal = research.build_signal_panel(residual)
    finally:
        research.OUT_DATA = old_out_data
        research.OUT_REPORT = old_out_report
        research.signal_thresholds = old_thresholds
    signal.to_csv(signal_path, index=False)
    return signal


def run_grid_backtest(signal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = converged.normalize_signal(signal)
    normalized.to_csv(OUT_DIR / "advanced_lambda_grid_signal_panel_normalized.csv", index=False)
    prices = converged.load_price_panel()
    times = pd.DatetimeIndex([ts for ts in prices.index if ts >= normalized["timestamp"].min() and ts <= normalized["timestamp"].max()])
    universe_sets = converged.build_universe_sets(times, set(normalized["ticker"].unique()))
    all_equity = []
    all_positions = []
    all_sleeves = []
    for method in [lambda_label(x) for x in LAMBDA_GRID]:
        log(f"backtesting {method} portfolio_lambda={PORTFOLIO_LAMBDA}")
        converged.PORTFOLIO_LAMBDA[method] = PORTFOLIO_LAMBDA
        eq, pos, slv = converged.run_engine(normalized, method, prices, universe_sets)
        all_equity.append(eq)
        all_positions.append(pos)
        all_sleeves.append(slv)
    equity = pd.concat(all_equity, ignore_index=True)
    positions = pd.concat([p for p in all_positions if not p.empty], ignore_index=True)
    sleeves = pd.concat([s for s in all_sleeves if not s.empty], ignore_index=True)
    summary = converged.summarize(equity, positions)
    summary["pca_lambda"] = summary["method"].map({lambda_label(x): x for x in LAMBDA_GRID})
    summary["portfolio_lambda"] = PORTFOLIO_LAMBDA
    equity.to_csv(OUT_DIR / "advanced_lambda_grid_hourly_equity.csv", index=False)
    positions.to_csv(OUT_DIR / "advanced_lambda_grid_positions.csv", index=False)
    sleeves.to_csv(OUT_DIR / "advanced_lambda_grid_sleeves.csv", index=False)
    summary.to_csv(REPORT_DIR / "advanced_lambda_grid_summary.csv", index=False)
    return equity, positions, summary


def write_report(summary: pd.DataFrame, args: argparse.Namespace) -> None:
    view = summary[summary["fee_bps"].eq(5)].sort_values("final_net_equity", ascending=False)
    lines = [
        "# Advanced PCA Lambda Full Grid",
        "",
        f"- PCA lambdas: `{LAMBDA_GRID}`",
        f"- Portfolio soft beta lambda: `{PORTFOLIO_LAMBDA}`",
        f"- PCA maxiter: `{args.maxiter}`; random starts: `{args.random_starts}`",
        f"- n_jobs: `{args.n_jobs}`",
        "",
        "## 5bps Ranking",
        "",
        view.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (REPORT_DIR / "advanced_lambda_grid_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    residual = build_residual_panel(args)
    signal = build_signal_panel(residual, args)
    _, _, summary = run_grid_backtest(signal)
    write_report(summary, args)
    print(summary[summary["fee_bps"].eq(5)].sort_values("final_net_equity", ascending=False).to_string(index=False))
    print(f"summary: {REPORT_DIR / 'advanced_lambda_grid_summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-sample multiprocess advanced PCA lambda grid.")
    parser.add_argument("--max-hours", type=int, default=None)
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--m-components", type=int, default=8)
    parser.add_argument("--min-assets", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--random-starts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=48, help="Contiguous timestamps per worker task; warm start is applied inside each block.")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    if args.full_run:
        args.max_hours = None
    return args


if __name__ == "__main__":
    run(parse_args())
