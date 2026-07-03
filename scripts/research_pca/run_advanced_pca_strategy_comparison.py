from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = Path(__file__).resolve().parent
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from run_residual_comovement_penalized_pca import (  # noqa: E402
    K_FACTORS,
    WINDOW,
    active_universe_from_row,
    canonical_a,
    factor_metrics,
    load_price_panel,
    load_universe_panel,
    optimize_advanced_pca,
    pca_basis,
    residual_from_loadings,
    standardize_window,
)


OUT_DATA = PROJECT_ROOT / "data" / "processed" / "research_pca" / "advanced_pca_strategy_comparison"
OUT_REPORT = PROJECT_ROOT / "reports" / "research_pca" / "advanced_pca_strategy_comparison"
FINAL_TABLES = PROJECT_ROOT / "reports" / "final_report" / "tables"

OU_WINDOW = 360
GROSS_CAP = 2.5
SOFT_PC1_LAMBDA = 100.0
FEE_BPS_LIST = [0, 5, 10]
METHODS = [("ordinary", None), ("advanced_lambda_0p5", 0.5)]


def log(message: str) -> None:
    print(f"[advanced_pca_strategy] {message}", flush=True)


def ensure_dirs() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)


def half_life_bucket(hours: float) -> str | None:
    if not np.isfinite(hours) or hours <= 0:
        return None
    if hours < 9:
        return "hl_0_9"
    if hours < 18:
        return "hl_9_18"
    if hours < 36:
        return "hl_18_36"
    if hours < 60:
        return "hl_36_60"
    if hours <= 90:
        return "hl_60_90"
    return None


def signal_thresholds() -> dict[tuple[str, str], tuple[float, float]]:
    path = FINAL_TABLES / "final_signal_rules.csv"
    df = pd.read_csv(path)
    return {(r.side, r.hl_bucket): (float(r.entry_abs), float(r.exit_abs)) for r in df.itertuples(index=False)}


@dataclass
class Position:
    position_id: int
    sleeve_id: int
    method: str
    strategy: str
    side: str
    ticker: str
    entry_time: pd.Timestamp
    entry_price: float
    notional: float
    beta: float
    entry_s_score: float
    entry_half_life: float
    hl_bucket: str


def side_sign(side: str) -> float:
    return 1.0 if side == "long" else -1.0


def max_drawdown(series: pd.Series) -> float:
    return float((series - series.cummax()).min()) if len(series) else 0.0


def sharpe_like(pnl: pd.Series) -> float:
    std = pnl.std(ddof=1)
    return float(pnl.mean() / std * np.sqrt(len(pnl))) if std and std > 0 else np.nan


def build_residual_panels(args: argparse.Namespace) -> pd.DataFrame:
    prices = load_price_panel()
    universe, rank_cols = load_universe_panel()
    common_index = prices.index.intersection(universe.index)
    prices = prices.reindex(common_index)
    universe = universe.reindex(common_index)
    returns = np.log(prices.where(prices > 0)).diff()
    price_cols = set(prices.columns)
    rng = np.random.default_rng(args.seed)

    start = WINDOW
    stop = len(common_index) if args.max_hours is None else min(len(common_index), WINDOW + args.max_hours)
    rows: list[dict[str, object]] = []
    skipped = 0
    opt_cache: dict[tuple[str, int], np.ndarray] = {}

    for count, pos in enumerate(range(start, stop), start=1):
        if count == 1 or count % 250 == 0:
            log(f"residual generation {count}/{stop - start}")
        ts = common_index[pos]
        candidates = active_universe_from_row(universe.loc[ts, rank_cols], price_cols)
        if len(candidates) < args.min_assets:
            skipped += 1
            continue
        ret_window = returns.iloc[pos - WINDOW : pos][candidates]
        y, eligible = standardize_window(ret_window)
        if len(eligible) < args.min_assets:
            skipped += 1
            continue
        current_ret = returns.iloc[pos][eligible]
        current_price = prices.iloc[pos][eligible]
        good_now = current_ret.notna() & current_price.gt(0)
        if int(good_now.sum()) < args.min_assets:
            skipped += 1
            continue
        eligible_now = [c for c in eligible if bool(good_now.get(c, False))]
        idx = [eligible.index(c) for c in eligible_now]
        ret_window_now = ret_window[eligible_now]
        y_fit, eligible_now_check = standardize_window(ret_window_now)
        if eligible_now_check != eligible_now:
            skipped += 1
            continue
        basis = pca_basis(y_fit)
        m = min(args.m_components, len(eligible_now) - 1, len(basis.eigvals))
        if m < K_FACTORS:
            skipped += 1
            continue
        mean = ret_window_now.mean(axis=0)
        std = ret_window_now.std(axis=0, ddof=1)
        y_current = ((current_ret[eligible_now] - mean) / std).to_numpy(dtype=float)
        vol = std.to_numpy(dtype=float)

        method_q: dict[str, np.ndarray] = {"ordinary": basis.eigvecs[:, :K_FACTORS]}
        for method, lambda_penalty in METHODS:
            if lambda_penalty is None:
                continue
            # Warm-start the exact prototype lightly. This is still no-lookahead:
            # only [t-W, t-1] is used to choose the subspace for timestamp t.
            a, _ = optimize_advanced_pca(
                y=y_fit,
                basis=basis,
                m=m,
                k=K_FACTORS,
                lambda_penalty=lambda_penalty,
                rng=rng,
                random_starts=args.random_starts,
                maxiter=args.maxiter,
            )
            method_q[method] = basis.eigvecs[:, :m] @ a

        for method, q in method_q.items():
            y_resid = y_current - (y_current @ q @ q.T)
            # Beta is a z-scored first-factor loading proxy used only by the soft
            # factor-constrained sleeve optimizer.
            beta = q[:, 0] / np.where(vol > 0, vol, np.nan)
            beta = (beta - np.nanmean(beta)) / np.nanstd(beta) if np.nanstd(beta) > 0 else beta * 0.0
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
                        "raw_log_return": raw_ret,
                        "price": price,
                        "residual_return": resid,
                        "beta_factor1": beta_i,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA / "advanced_residual_returns_long.csv", index=False)
    pd.DataFrame([{"skipped_pca_timestamps": skipped, "rows": len(out)}]).to_csv(OUT_DATA / "residual_generation_summary.csv", index=False)
    return out


def estimate_ou_window(residuals: np.ndarray) -> tuple[float, float, float, float] | None:
    if len(residuals) < OU_WINDOW or np.any(~np.isfinite(residuals)):
        return None
    level = np.cumsum(residuals)
    x = level[:-1]
    y = level[1:]
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    var_x = float(np.sum((x - x_mean) ** 2))
    if var_x <= 0:
        return None
    b = float(np.sum((x - x_mean) * (y - y_mean)) / var_x)
    a = y_mean - b * x_mean
    pred = a + b * x
    err = y - pred
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else np.nan
    if not np.isfinite(b) or not (0 < b < 1):
        return None
    sigma_eps = float(np.std(err, ddof=1))
    sigma_eq = sigma_eps / math.sqrt(max(1e-12, 1 - b * b))
    mu = a / (1 - b)
    half_life = -math.log(2) / math.log(b)
    s_score = (level[-1] - mu) / sigma_eq if sigma_eq > 0 else np.nan
    if not all(np.isfinite(x) for x in [sigma_eq, half_life, s_score, r2]):
        return None
    return b, half_life, sigma_eq, r2, s_score


def build_signal_panel(residual_panel: pd.DataFrame) -> pd.DataFrame:
    thresholds = signal_thresholds()
    rows: list[dict[str, object]] = []
    residual_panel = residual_panel.sort_values(["method", "ticker", "timestamp"])
    for (method, ticker), group in residual_panel.groupby(["method", "ticker"], sort=False):
        residuals = group["residual_return"].to_numpy(dtype=float)
        records = group.to_dict("records")
        for i in range(OU_WINDOW - 1, len(group)):
            est = estimate_ou_window(residuals[i - OU_WINDOW + 1 : i + 1])
            if est is None:
                continue
            b, half_life, sigma_eq, r2, s_score = est
            bucket = half_life_bucket(half_life)
            if bucket is None:
                continue
            rec = records[i]
            long_entry, long_exit = thresholds[("long", bucket)]
            short_entry, short_exit = thresholds[("short", bucket)]
            long_signal = s_score <= -long_entry
            short_signal = s_score >= short_entry
            rows.append(
                {
                    **rec,
                    "s_score": s_score,
                    "ou_b": b,
                    "ou_half_life_hours": half_life,
                    "ou_sigma_eq": sigma_eq,
                    "regression_r2": r2,
                    "hl_bucket": bucket,
                    "long_entry_abs": long_entry,
                    "long_exit_abs": long_exit,
                    "short_entry_abs": short_entry,
                    "short_exit_abs": short_exit,
                    "long_signal": long_signal,
                    "short_signal": short_signal,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA / "advanced_signal_panel.csv", index=False)
    return out


def pnl_for_position(pos: Position, current_price: float) -> float:
    ret = math.log(current_price / pos.entry_price)
    return pos.notional * side_sign(pos.side) * ret


def fee_amount(notional: float, fee_bps: float) -> float:
    return abs(notional) * fee_bps / 10000.0


def run_naive(signal_panel: pd.DataFrame, fee_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions: dict[tuple[str, str], Position] = {}
    closed: list[dict[str, object]] = []
    equity_rows = []
    pid = 0
    methods = sorted(signal_panel["method"].unique())
    realized = {m: 0.0 for m in methods}
    entry_fees = {m: 0.0 for m in methods}
    times = sorted(signal_panel["timestamp"].unique())
    for ts in times:
        panel = signal_panel[signal_panel["timestamp"] == ts].set_index(["method", "ticker"])
        # Exits and mark-to-market.
        for key, pos in list(positions.items()):
            row = panel.loc[key] if key in panel.index else None
            exit_now = row is None
            exit_reason = "missing_exit" if exit_now else ""
            if row is not None:
                s = float(row["s_score"])
                exit_abs = float(row["long_exit_abs"] if pos.side == "long" else row["short_exit_abs"])
                if (pos.side == "long" and s >= -exit_abs) or (pos.side == "short" and s <= exit_abs):
                    exit_now = True
                    exit_reason = "threshold_exit"
            if exit_now:
                exit_price = pos.entry_price if row is None else float(row["price"])
                gross_pnl = pnl_for_position(pos, exit_price)
                exit_fee = fee_amount(pos.notional, fee_bps)
                realized[pos.method] += gross_pnl - exit_fee
                closed.append({**pos.__dict__, "exit_time": ts, "exit_reason": exit_reason, "gross_pnl": gross_pnl, "net_pnl": gross_pnl - fee_amount(pos.notional, fee_bps) - exit_fee})
                del positions[key]
        # Entries.
        for row in signal_panel[signal_panel["timestamp"] == ts].itertuples(index=False):
            method = row.method
            ticker = row.ticker
            if (method, ticker) in positions:
                continue
            side = "long" if bool(row.long_signal) else "short" if bool(row.short_signal) else None
            if side is None:
                continue
            pid += 1
            notional = 1.0
            entry_fees[method] += fee_amount(notional, fee_bps)
            positions[(method, ticker)] = Position(pid, pid, method, "naive_1dollar", side, ticker, ts, float(row.price), notional, float(row.beta_factor1), float(row.s_score), float(row.ou_half_life_hours), row.hl_bucket)
        # Equity by method.
        for method in methods:
            unrealized = 0.0
            active = [p for k, p in positions.items() if p.method == method]
            for pos in active:
                key = (method, pos.ticker)
                if key in panel.index:
                    unrealized += pnl_for_position(pos, float(panel.loc[key]["price"]))
            equity_rows.append({"timestamp": ts, "method": method, "strategy": "naive_1dollar", "fee_bps": fee_bps, "net_equity": realized[method] - entry_fees[method] + unrealized, "n_active_positions": len(active)})
    return pd.DataFrame(equity_rows), pd.DataFrame(closed)


def optimize_soft_weights(candidates: pd.DataFrame, side_notional: float) -> np.ndarray:
    n = len(candidates)
    sides = np.where(candidates["side"].to_numpy() == "long", 1.0, -1.0)
    beta = candidates["beta_factor1"].to_numpy(dtype=float)
    long_mask = sides > 0
    short_mask = sides < 0
    equal = np.zeros(n)
    equal[long_mask] = side_notional / max(1, int(long_mask.sum()))
    equal[short_mask] = side_notional / max(1, int(short_mask.sum()))

    def obj(w: np.ndarray) -> float:
        return float(np.sum((w - equal) ** 2) + SOFT_PC1_LAMBDA * (np.sum(w * sides * beta) ** 2))

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w[long_mask]) - side_notional},
        {"type": "eq", "fun": lambda w: np.sum(w[short_mask]) - side_notional},
    ]
    bounds = [(0.0, side_notional) for _ in range(n)]
    res = minimize(obj, equal, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 100, "ftol": 1e-9})
    if not res.success or np.any(~np.isfinite(res.x)):
        return equal
    return res.x


def run_sleeve(signal_panel: pd.DataFrame, strategy: str, fee_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    active: dict[int, Position] = {}
    closed: list[dict[str, object]] = []
    equity_rows = []
    pid = 0
    sleeve_id = 0
    realized = {m: 0.0 for m in signal_panel["method"].unique()}
    entry_fees = {m: 0.0 for m in signal_panel["method"].unique()}
    times = sorted(signal_panel["timestamp"].unique())
    for ts in times:
        for method, panel_m in signal_panel[signal_panel["timestamp"] == ts].groupby("method"):
            panel = panel_m.set_index("ticker")
            for key, pos in list(active.items()):
                if pos.method != method:
                    continue
                row = panel.loc[pos.ticker] if pos.ticker in panel.index else None
                exit_now = row is None
                exit_reason = "missing_exit" if exit_now else ""
                if row is not None:
                    s = float(row["s_score"])
                    exit_abs = float(row["long_exit_abs"] if pos.side == "long" else row["short_exit_abs"])
                    if (pos.side == "long" and s >= -exit_abs) or (pos.side == "short" and s <= exit_abs):
                        exit_now = True
                        exit_reason = "threshold_exit"
                if exit_now:
                    exit_price = pos.entry_price if row is None else float(row["price"])
                    gross_pnl = pnl_for_position(pos, exit_price)
                    exit_fee = fee_amount(pos.notional, fee_bps)
                    realized[method] += gross_pnl - exit_fee
                    closed.append({**pos.__dict__, "exit_time": ts, "exit_reason": exit_reason, "gross_pnl": gross_pnl, "net_pnl": gross_pnl - fee_amount(pos.notional, fee_bps) - exit_fee})
                    del active[key]

            active_tickers = {p.ticker for p in active.values() if p.method == method}
            entries = []
            for row in panel_m.itertuples(index=False):
                if row.ticker in active_tickers:
                    continue
                side = "long" if bool(row.long_signal) else "short" if bool(row.short_signal) else None
                if side is not None:
                    entries.append({**row._asdict(), "side": side})
            entries_df = pd.DataFrame(entries)
            if not entries_df.empty and {"long", "short"}.issubset(set(entries_df["side"])):
                current_gross = sum(p.notional for p in active.values() if p.method == method)
                capacity = max(0.0, GROSS_CAP - current_gross)
                side_notional = min(1.0, capacity / 2.0)
                if side_notional > 1e-9:
                    sleeve_id += 1
                    if strategy == "dollar_neutral":
                        weights = np.zeros(len(entries_df))
                        for side in ["long", "short"]:
                            mask = entries_df["side"].to_numpy() == side
                            weights[mask] = side_notional / int(mask.sum())
                    else:
                        weights = optimize_soft_weights(entries_df, side_notional)
                    for row, weight in zip(entries_df.itertuples(index=False), weights):
                        if weight <= 0:
                            continue
                        pid += 1
                        entry_fees[method] += fee_amount(weight, fee_bps)
                        active[pid] = Position(pid, sleeve_id, method, strategy, row.side, row.ticker, ts, float(row.price), float(weight), float(row.beta_factor1), float(row.s_score), float(row.ou_half_life_hours), row.hl_bucket)

            unrealized = 0.0
            active_m = [p for p in active.values() if p.method == method]
            for pos in active_m:
                if pos.ticker in panel.index:
                    unrealized += pnl_for_position(pos, float(panel.loc[pos.ticker]["price"]))
            gross = sum(p.notional for p in active_m)
            net = sum(p.notional * side_sign(p.side) for p in active_m)
            pc1 = sum(p.notional * side_sign(p.side) * p.beta for p in active_m)
            equity_rows.append({"timestamp": ts, "method": method, "strategy": strategy, "fee_bps": fee_bps, "net_equity": realized[method] - entry_fees[method] + unrealized, "active_gross_exposure": gross, "active_net_exposure": net, "active_pc1_exposure": pc1, "n_active_positions": len(active_m)})
    return pd.DataFrame(equity_rows), pd.DataFrame(closed)


def summarize(equity: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, strategy, fee), group in equity.groupby(["method", "strategy", "fee_bps"]):
        group = group.sort_values("timestamp")
        pnl = group["net_equity"].diff().fillna(group["net_equity"])
        pos_count = 0
        if not positions.empty:
            pos_count = int(((positions["method"] == method) & (positions["strategy"] == strategy)).sum())
        rows.append(
            {
                "method": method,
                "strategy": strategy,
                "fee_bps": fee,
                "final_net_equity": float(group["net_equity"].iloc[-1]),
                "max_drawdown_net": max_drawdown(group["net_equity"]),
                "sharpe_like_net": sharpe_like(pnl),
                "avg_active_gross_exposure": float(group.get("active_gross_exposure", pd.Series(np.nan, index=group.index)).mean()),
                "avg_abs_active_net_exposure": float(group.get("active_net_exposure", pd.Series(np.nan, index=group.index)).abs().mean()),
                "avg_abs_active_pc1_exposure": float(group.get("active_pc1_exposure", pd.Series(np.nan, index=group.index)).abs().mean()),
                "total_positions": pos_count,
            }
        )
    return pd.DataFrame(rows)


def summarize_equity_series(group: pd.DataFrame, equity_col: str) -> dict[str, float]:
    group = group.sort_values("timestamp").copy()
    eq = group[equity_col] - float(group[equity_col].iloc[0])
    pnl = eq.diff().fillna(eq)
    return {
        "final_net_equity": float(eq.iloc[-1]),
        "max_drawdown_net": max_drawdown(eq),
        "sharpe_like_net": sharpe_like(pnl),
    }


def add_mainline_baseline(summary: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    rows = []
    naive_eq = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "naive_backtest" / "blend_equity_curves.csv", parse_dates=["timestamp"])
    naive_eq = naive_eq[(naive_eq["timestamp"] >= start_ts) & (naive_eq["timestamp"] <= end_ts) & (naive_eq["config"] == "long_short_50_50")]
    for fee, group in naive_eq.groupby("fee_bps"):
        metrics = summarize_equity_series(group, "equity")
        rows.append({"method": "mainline_baseline", "strategy": "naive_1dollar", "fee_bps": fee, **metrics})

    final_eq = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "final_strategy" / "soft_pc1_refined_hourly_equity.csv", parse_dates=["timestamp"])
    final_eq = final_eq[(final_eq["timestamp"] >= start_ts) & (final_eq["timestamp"] <= end_ts)]
    mapping = {
        "baseline_equal_weight_gross_cap_2p5": "dollar_neutral",
        "soft_pc1_z_lambda_100_gross_cap_2p5": "soft_pc1_lambda100",
    }
    for config, strategy in mapping.items():
        group = final_eq[final_eq["config"] == config].copy()
        if group.empty:
            continue
        for fee in FEE_BPS_LIST:
            metrics = summarize_equity_series(group, f"net_equity_{fee}bps")
            rows.append(
                {
                    "method": "mainline_baseline",
                    "strategy": strategy,
                    "fee_bps": fee,
                    **metrics,
                    "avg_active_gross_exposure": float(group["active_gross_exposure"].mean()),
                    "avg_abs_active_net_exposure": float(group["active_net_exposure"].abs().mean()),
                    "avg_abs_active_pc1_exposure": float(group["active_pc1_exposure_used_beta"].abs().mean()),
                    "total_positions": np.nan,
                }
            )
    return pd.concat([summary, pd.DataFrame(rows)], ignore_index=True)


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    residual_path = OUT_DATA / "advanced_residual_returns_long.csv"
    signal_path = OUT_DATA / "advanced_signal_panel.csv"
    if args.reuse and residual_path.exists():
        residual_panel = pd.read_csv(residual_path, parse_dates=["timestamp"])
    else:
        residual_panel = build_residual_panels(args)
    if args.reuse and signal_path.exists():
        signal_panel = pd.read_csv(signal_path, parse_dates=["timestamp"])
    else:
        signal_panel = build_signal_panel(residual_panel)
    if signal_panel.empty:
        raise RuntimeError("No signal rows generated. Increase --max-hours above the PCA+OU warmup window.")

    all_equity = []
    all_pos = []
    for fee in FEE_BPS_LIST:
        log(f"backtesting fee={fee}bps naive")
        eq, pos = run_naive(signal_panel, fee)
        all_equity.append(eq)
        all_pos.append(pos)
        for strategy in ["dollar_neutral", "soft_pc1_lambda100"]:
            log(f"backtesting fee={fee}bps {strategy}")
            eq, pos = run_sleeve(signal_panel, strategy, fee)
            all_equity.append(eq)
            all_pos.append(pos)
    equity = pd.concat(all_equity, ignore_index=True)
    positions = pd.concat([p for p in all_pos if not p.empty], ignore_index=True) if any(not p.empty for p in all_pos) else pd.DataFrame()
    summary = summarize(equity, positions)
    start_ts = pd.to_datetime(signal_panel["timestamp"].min())
    end_ts = pd.to_datetime(signal_panel["timestamp"].max())
    comparison = add_mainline_baseline(summary, start_ts, end_ts)
    equity.to_csv(OUT_DATA / "strategy_equity.csv", index=False)
    positions.to_csv(OUT_DATA / "strategy_positions.csv", index=False)
    summary.to_csv(OUT_REPORT / "advanced_pca_strategy_summary.csv", index=False)
    comparison.to_csv(OUT_REPORT / "advanced_pca_vs_mainline_baseline_summary.csv", index=False)
    write_report(comparison, signal_panel, args)


def write_report(comparison: pd.DataFrame, signal_panel: pd.DataFrame, args: argparse.Namespace) -> None:
    view = comparison[comparison["fee_bps"] == 5].copy()
    view = view.sort_values(["strategy", "method"])
    lines = [
        "# Advanced PCA Strategy Comparison",
        "",
        "This research run compares ordinary W360/PC3 PCA with residual-comovement-penalized PCA at lambda 0.5. Sigma-stop logic is intentionally excluded.",
        "",
        f"- Signal rows: `{len(signal_panel)}`",
        f"- Timestamp range: `{signal_panel['timestamp'].min()}` to `{signal_panel['timestamp'].max()}`",
        f"- Advanced PCA max hours: `{args.max_hours if args.max_hours is not None else 'full'}`",
        f"- PCA optimizer maxiter: `{args.maxiter}`; random starts: `{args.random_starts}`",
        "",
        "## 5bps Summary",
        "",
        view[["method", "strategy", "final_net_equity", "max_drawdown_net", "sharpe_like_net", "avg_active_gross_exposure", "avg_abs_active_pc1_exposure", "total_positions"]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Mainline baseline rows are loaded from the existing final report tables. Research rows are generated by this script on the same input price/universe data with the same entry/exit threshold table.",
    ]
    (OUT_REPORT / "advanced_pca_strategy_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare advanced PCA residual signals against mainline baseline strategies.")
    parser.add_argument("--max-hours", type=int, default=1500, help="Cap PCA residual generation hours for research iteration; omit with --full-run.")
    parser.add_argument("--full-run", action="store_true", help="Run all available hours.")
    parser.add_argument("--m-components", type=int, default=8)
    parser.add_argument("--min-assets", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--random-starts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--reuse", action="store_true", help="Reuse existing residual/signal panels.")
    args = parser.parse_args()
    if args.full_run:
        args.max_hours = None
    return args


if __name__ == "__main__":
    run(parse_args())
