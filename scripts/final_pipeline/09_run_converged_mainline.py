from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "scripts" / "research_pca"
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

import run_advanced_pca_strategy_comparison as research  # noqa: E402
from run_residual_comovement_penalized_pca import load_price_panel, load_universe_panel, active_universe_from_row  # noqa: E402


OUT_DIR = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "converged_mainline"
REPORT_DIR = PROJECT_ROOT / "reports" / "final_report" / "converged_mainline"
FIG_DIR = REPORT_DIR / "figures"
OLD_ADVANCED_SIGNAL = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "advanced_pca_v1" / "advanced_signal_panel.csv"
FROZEN_ORDINARY_SIGNAL = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "final_strategy" / "w360_pc3_signal_panel_with_beta.csv"

PCA_LAMBDA_ADVANCED = 0.5
PORTFOLIO_LAMBDA = {"ordinary": 0.0, "advanced_lambda_0p5": 3.0}
GROSS_CAP = 1.5
FEE_BPS_LIST = [0, 5, 10]
SHORT_ENTRY = 1.0
SHORT_EXIT = 0.25
LONG_ENTRY = 1.0
LONG_EXIT = 0.5


@dataclass
class Position:
    position_id: int
    sleeve_id: int
    method: str
    side: str
    ticker: str
    entry_time: pd.Timestamp
    entry_price: float
    notional: float
    beta: float
    entry_s_score: float
    entry_half_life: float
    left_universe_flag: bool = False
    first_left_universe_time: pd.Timestamp | None = None
    pnl_at_first_left_universe: float = np.nan


def log(message: str) -> None:
    print(f"[converged_mainline] {message}", flush=True)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(x))
    except TypeError:
        return False


def fixed_signal_thresholds() -> dict[tuple[str, str], tuple[float, float]]:
    buckets = ["hl_0_9", "hl_9_18", "hl_18_36", "hl_36_60", "hl_60_90"]
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for bucket in buckets:
        out[("long", bucket)] = (LONG_ENTRY, LONG_EXIT)
        out[("short", bucket)] = (SHORT_ENTRY, SHORT_EXIT)
    return out


def configure_research() -> None:
    research.OUT_DATA = OUT_DIR
    research.OUT_REPORT = REPORT_DIR
    research.METHODS = [("ordinary", None), ("advanced_lambda_0p5", PCA_LAMBDA_ADVANCED)]
    research.signal_thresholds = fixed_signal_thresholds


def build_or_load_signal(args: argparse.Namespace) -> pd.DataFrame:
    configure_research()
    if args.use_frozen_panels:
        if not FROZEN_ORDINARY_SIGNAL.exists() or not OLD_ADVANCED_SIGNAL.exists():
            raise FileNotFoundError("Frozen ordinary/advanced signal panels are missing.")
        ordinary = pd.read_csv(FROZEN_ORDINARY_SIGNAL, parse_dates=["timestamp"], low_memory=False)
        ordinary = ordinary.copy()
        ordinary["method"] = "ordinary"
        ordinary["beta_factor1"] = pd.to_numeric(ordinary["beta_PC1"], errors="coerce")
        g = ordinary.groupby("timestamp")["beta_factor1"]
        ordinary["beta_factor1"] = ((ordinary["beta_factor1"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
        advanced = pd.read_csv(OLD_ADVANCED_SIGNAL, parse_dates=["timestamp"], low_memory=False)
        advanced = advanced[advanced["method"].eq("advanced_lambda_0p5")].copy()
        keep = [
            "timestamp",
            "method",
            "ticker",
            "raw_log_return",
            "price",
            "residual_return",
            "beta_factor1",
            "s_score",
            "ou_b",
            "ou_half_life_hours",
            "ou_sigma_eq",
            "regression_r2",
        ]
        cols = [c for c in keep if c in ordinary.columns and c in advanced.columns]
        # `residual_return` is not needed by the final engine, so tolerate its absence in the ordinary panel.
        base_cols = ["timestamp", "method", "ticker", "raw_log_return", "price", "beta_factor1", "s_score", "ou_b", "ou_half_life_hours", "ou_sigma_eq", "regression_r2"]
        out = pd.concat([ordinary[base_cols], advanced[base_cols]], ignore_index=True)
        out.to_csv(OUT_DIR / "advanced_signal_panel.csv", index=False)
        return out
    residual_path = OUT_DIR / "advanced_residual_returns_long.csv"
    signal_path = OUT_DIR / "advanced_signal_panel.csv"
    if args.reuse and signal_path.exists():
        return pd.read_csv(signal_path, parse_dates=["timestamp"], low_memory=False)
    if args.reuse and residual_path.exists():
        residual = pd.read_csv(residual_path, parse_dates=["timestamp"], low_memory=False)
    else:
        residual = research.build_residual_panels(args)
    signal = research.build_signal_panel(residual)
    if signal.empty:
        raise RuntimeError("empty signal panel")
    signal.to_csv(signal_path, index=False)
    return signal


def normalize_signal(signal: pd.DataFrame) -> pd.DataFrame:
    df = signal.copy()
    df["beta_used"] = pd.to_numeric(df["beta_factor1"], errors="coerce")
    df["in_no_lookahead_universe"] = True
    for col in ["timestamp", "ticker", "method", "price", "raw_log_return", "s_score", "ou_half_life_hours", "beta_used"]:
        if col not in df.columns:
            raise ValueError(f"signal missing {col}")
    for col in ["price", "raw_log_return", "s_score", "ou_half_life_hours", "beta_used"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["timestamp", "ticker", "method"]).reset_index(drop=True)


def build_universe_sets(times: pd.DatetimeIndex, tickers: set[str]) -> dict[pd.Timestamp, set[str]]:
    universe, rank_cols = load_universe_panel()
    universe = universe.reindex(times)
    price_cols = set(tickers)
    out: dict[pd.Timestamp, set[str]] = {}
    for ts, row in universe[rank_cols].iterrows():
        out[ts] = set(active_universe_from_row(row, price_cols))
    return out


def current_raw_price(price_row: pd.Series | None, ticker: str, fallback: float) -> float:
    if price_row is not None and ticker in price_row.index:
        px = price_row[ticker]
        if finite(px) and float(px) > 0:
            return float(px)
    return fallback


def abs_exposure(pos: Position, price: float) -> float:
    return pos.notional * price / pos.entry_price


def signed_exposure(pos: Position, price: float) -> float:
    value = abs_exposure(pos, price)
    return value if pos.side == "long" else -value


def position_pnl(pos: Position, price: float) -> float:
    ratio = price / pos.entry_price
    return pos.notional * (ratio - 1.0) if pos.side == "long" else pos.notional * (1.0 - ratio)


def entry_quality_ok(row: Any) -> bool:
    return (
        finite(row.ou_half_life_hours)
        and 0 < float(row.ou_half_life_hours) <= 90
        and finite(row.s_score)
        and finite(row.raw_log_return)
        and finite(row.price)
        and float(row.price) > 0
        and bool(row.in_no_lookahead_universe)
    )


def exit_reason(pos: Position, row: Any | None) -> str | None:
    if row is None:
        return "missing_signal_exit_last_price"
    if not finite(row.price) or float(row.price) <= 0 or not finite(row.s_score):
        return "missing_exit_last_price"
    if not entry_quality_ok(row):
        return "eligibility_lost"
    if pos.side == "long" and float(row.s_score) > -LONG_EXIT:
        return "threshold_exit"
    if pos.side == "short" and float(row.s_score) < SHORT_EXIT:
        return "threshold_exit"
    return None


def optimize_soft_notional(long_rows: list[Any], short_rows: list[Any], side_target: float, soft_lambda: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_l = len(long_rows)
    n_s = len(short_rows)
    x0 = np.full(n_l, side_target / n_l)
    y0 = np.full(n_s, side_target / n_s)
    if soft_lambda <= 0:
        return x0, y0, {"optimizer_success_flag": True, "z_exposure_before": np.nan, "z_exposure_after": np.nan, "optimizer_skipped_equal_weight_flag": True}
    beta_l = np.array([float(r.beta_used) if finite(r.beta_used) else 0.0 for r in long_rows])
    beta_s = np.array([float(r.beta_used) if finite(r.beta_used) else 0.0 for r in short_rows])
    z0 = np.r_[x0, y0]
    pre = float(np.dot(x0, beta_l) - np.dot(y0, beta_s))

    def obj(z: np.ndarray) -> float:
        x = z[:n_l]
        y = z[n_l:]
        exposure = float(np.dot(x, beta_l) - np.dot(y, beta_s))
        return float(np.sum((x - x0) ** 2) + np.sum((y - y0) ** 2) + soft_lambda * exposure * exposure)

    constraints = [
        {"type": "eq", "fun": lambda z: np.sum(z[:n_l]) - side_target},
        {"type": "eq", "fun": lambda z: np.sum(z[n_l:]) - side_target},
    ]
    bounds = [(0.0, 2.0 * v) for v in z0]
    res = minimize(obj, z0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 200, "ftol": 1e-10, "disp": False})
    success = bool(res.success and np.all(np.isfinite(res.x)))
    z = res.x if success else z0
    post = float(np.dot(z[:n_l], beta_l) - np.dot(z[n_l:], beta_s))
    return z[:n_l], z[n_l:], {"optimizer_success_flag": success, "z_exposure_before": pre, "z_exposure_after": post}


def close_position(pos: Position, ts: pd.Timestamp, row: Any | None, px: float, reason: str, fee_charged: bool) -> dict[str, Any]:
    gpnl = position_pnl(pos, px)
    return {
        "position_id": pos.position_id,
        "sleeve_id": pos.sleeve_id,
        "method": pos.method,
        "side": pos.side,
        "ticker": pos.ticker,
        "entry_time": pos.entry_time,
        "exit_time": ts,
        "exit_reason": reason,
        "entry_price": pos.entry_price,
        "exit_price": px,
        "initial_notional": pos.notional,
        "exit_abs_exposure": abs_exposure(pos, px),
        "gross_pnl": gpnl,
        "gross_return": gpnl / pos.notional if pos.notional else np.nan,
        "holding_hours": (ts - pos.entry_time).total_seconds() / 3600.0,
        "entry_s_score": pos.entry_s_score,
        "exit_s_score": float(row.s_score) if row is not None and finite(row.s_score) else np.nan,
        "entry_half_life": pos.entry_half_life,
        "exit_half_life": float(row.ou_half_life_hours) if row is not None and finite(row.ou_half_life_hours) else np.nan,
        "beta_used": pos.beta,
        "left_universe_flag": pos.left_universe_flag,
        "first_left_universe_time": pos.first_left_universe_time,
        "pnl_at_first_left_universe": pos.pnl_at_first_left_universe,
        "pnl_after_first_left_universe": gpnl - pos.pnl_at_first_left_universe if finite(pos.pnl_at_first_left_universe) else np.nan,
        "exit_fee_charged_flag": fee_charged,
    }


def run_engine(panel: pd.DataFrame, method: str, prices: pd.DataFrame, universe_sets: dict[pd.Timestamp, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub = panel[panel["method"].eq(method)].copy()
    times = [ts for ts in prices.index if ts >= sub["timestamp"].min() and ts <= sub["timestamp"].max() and ts in universe_sets]
    rows_by_time = {ts: {r.ticker: r for r in g.itertuples(index=False)} for ts, g in sub.groupby("timestamp", sort=False)}
    soft_lambda = PORTFOLIO_LAMBDA[method]
    active: list[Position] = []
    last_signal_price: dict[str, float] = {}
    last_raw_price: dict[str, float] = {}
    realized_gross = 0.0
    cumulative_fees = {fee: 0.0 for fee in FEE_BPS_LIST}
    position_rows: list[dict[str, Any]] = []
    sleeve_rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    position_id = 1
    sleeve_id = 1

    for ts in times:
        row_map = rows_by_time.get(ts, {})
        price_row = prices.loc[ts] if ts in prices.index else None
        active_universe = universe_sets.get(ts, set())
        for ticker, row in row_map.items():
            if finite(row.price) and float(row.price) > 0:
                last_signal_price[ticker] = float(row.price)
        for pos in active:
            px_raw = current_raw_price(price_row, pos.ticker, last_raw_price.get(pos.ticker, pos.entry_price))
            if finite(px_raw) and px_raw > 0:
                last_raw_price[pos.ticker] = px_raw

        exited_tickers: set[str] = set()
        next_active: list[Position] = []
        n_exits = 0
        for pos in active:
            row = row_map.get(pos.ticker)
            signal_px = float(row.price) if row is not None and finite(row.price) and float(row.price) > 0 else last_signal_price.get(pos.ticker, np.nan)
            raw_px = current_raw_price(price_row, pos.ticker, last_raw_price.get(pos.ticker, signal_px))
            if pos.ticker not in active_universe:
                if not pos.left_universe_flag:
                    pos.left_universe_flag = True
                    pos.first_left_universe_time = ts
                    pos.pnl_at_first_left_universe = position_pnl(pos, raw_px)
                if finite(raw_px) and raw_px > 0:
                    out = close_position(pos, ts, row, raw_px, "universe_lost_exit", True)
                    realized_gross += out["gross_pnl"]
                    for fee in FEE_BPS_LIST:
                        cumulative_fees[fee] += fee / 10000.0 * out["exit_abs_exposure"]
                    position_rows.append(out)
                    exited_tickers.add(pos.ticker)
                    n_exits += 1
                    continue
            reason = exit_reason(pos, row)
            if reason is None:
                next_active.append(pos)
                continue
            if finite(signal_px) and signal_px > 0:
                out = close_position(pos, ts, row, signal_px, reason, True)
                realized_gross += out["gross_pnl"]
                for fee in FEE_BPS_LIST:
                    cumulative_fees[fee] += fee / 10000.0 * out["exit_abs_exposure"]
                position_rows.append(out)
                exited_tickers.add(pos.ticker)
                n_exits += 1
            else:
                next_active.append(pos)
        active = next_active

        active_tickers = {p.ticker for p in active}
        long_rows = []
        short_rows = []
        for row in row_map.values():
            if row.ticker in active_tickers or row.ticker in exited_tickers or row.ticker not in active_universe:
                continue
            if not entry_quality_ok(row):
                continue
            if float(row.s_score) <= -LONG_ENTRY:
                long_rows.append(row)
            if float(row.s_score) >= SHORT_ENTRY:
                short_rows.append(row)

        n_new_sleeves = 0
        if long_rows and short_rows:
            current_gross = sum(abs_exposure(p, last_signal_price.get(p.ticker, p.entry_price)) for p in active)
            capacity = max(0.0, GROSS_CAP - current_gross)
            side_target = min(capacity / 2.0, float(min(len(long_rows), len(short_rows))))
            if side_target > 1e-8:
                long_notional, short_notional, opt_meta = optimize_soft_notional(long_rows, short_rows, side_target, soft_lambda)
                sleeve_rows.append(
                    {
                        "sleeve_id": sleeve_id,
                        "method": method,
                        "entry_time": ts,
                        "soft_lambda": soft_lambda,
                        "total_long_notional": float(np.sum(long_notional)),
                        "total_short_notional": float(np.sum(short_notional)),
                        "gross_exposure_at_entry": float(np.sum(long_notional) + np.sum(short_notional)),
                        **opt_meta,
                    }
                )
                for side, rows, notionals in [("long", long_rows, long_notional), ("short", short_rows, short_notional)]:
                    for row, notional in zip(rows, notionals):
                        if notional <= 1e-10:
                            continue
                        for fee in FEE_BPS_LIST:
                            cumulative_fees[fee] += fee / 10000.0 * float(notional)
                        active.append(
                            Position(
                                position_id=position_id,
                                sleeve_id=sleeve_id,
                                method=method,
                                side=side,
                                ticker=row.ticker,
                                entry_time=ts,
                                entry_price=float(row.price),
                                notional=float(notional),
                                beta=float(row.beta_used) if finite(row.beta_used) else 0.0,
                                entry_s_score=float(row.s_score),
                                entry_half_life=float(row.ou_half_life_hours),
                            )
                        )
                        position_id += 1
                sleeve_id += 1
                n_new_sleeves = 1

        unreal = gross_exposure = long_exposure = short_exposure = net_exposure = beta_exposure = 0.0
        for pos in active:
            px = last_signal_price.get(pos.ticker, pos.entry_price)
            expo_abs = abs_exposure(pos, px)
            expo_signed = signed_exposure(pos, px)
            unreal += position_pnl(pos, px)
            gross_exposure += expo_abs
            long_exposure += expo_abs if pos.side == "long" else 0.0
            short_exposure -= expo_abs if pos.side == "short" else 0.0
            net_exposure += expo_signed
            beta_exposure += expo_signed * pos.beta
        gross_equity = realized_gross + unreal
        equity_rows.append(
            {
                "timestamp": ts,
                "method": method,
                "gross_equity": gross_equity,
                "net_equity_0bps": gross_equity - cumulative_fees[0],
                "net_equity_5bps": gross_equity - cumulative_fees[5],
                "net_equity_10bps": gross_equity - cumulative_fees[10],
                "drawdown_5bps": np.nan,
                "active_gross_exposure": gross_exposure,
                "active_long_exposure": long_exposure,
                "active_short_exposure": short_exposure,
                "active_net_exposure": net_exposure,
                "active_beta_exposure": beta_exposure,
                "n_active_positions": len(active),
                "n_new_sleeves": n_new_sleeves,
                "n_exits": n_exits,
            }
        )

    final_ts = times[-1]
    final_row_map = rows_by_time.get(final_ts, {})
    for pos in active:
        px = last_signal_price.get(pos.ticker, np.nan)
        if finite(px) and px > 0:
            position_rows.append(close_position(pos, final_ts, final_row_map.get(pos.ticker), px, "open_mark_to_market", False))

    equity = pd.DataFrame(equity_rows)
    if not equity.empty:
        equity["drawdown_5bps"] = equity["net_equity_5bps"] - equity["net_equity_5bps"].cummax()
    positions = pd.DataFrame(position_rows)
    if not positions.empty:
        for fee in FEE_BPS_LIST:
            positions[f"entry_fee_{fee}bps"] = fee / 10000.0 * positions["initial_notional"]
            exit_base = positions["exit_abs_exposure"].where(positions["exit_fee_charged_flag"].fillna(True), 0.0)
            positions[f"exit_fee_{fee}bps"] = fee / 10000.0 * exit_base
            positions[f"net_pnl_{fee}bps"] = positions["gross_pnl"] - positions[f"entry_fee_{fee}bps"] - positions[f"exit_fee_{fee}bps"]
    return equity, positions, pd.DataFrame(sleeve_rows)


def summarize(equity: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, eq in equity.groupby("method", sort=False):
        eq = eq.sort_values("timestamp")
        pos = positions[positions["method"].eq(method)]
        for fee in FEE_BPS_LIST:
            col = f"net_equity_{fee}bps"
            pnl = eq[col].diff().fillna(eq[col])
            dd = eq[col] - eq[col].cummax()
            vol = pnl.std()
            rows.append(
                {
                    "method": method,
                    "portfolio_lambda": PORTFOLIO_LAMBDA[method],
                    "fee_bps": fee,
                    "final_gross_equity": float(eq["gross_equity"].iloc[-1]),
                    "final_net_equity": float(eq[col].iloc[-1]),
                    "total_fees_paid": float(eq["gross_equity"].iloc[-1] - eq[col].iloc[-1]),
                    "max_drawdown_net": float(dd.min()),
                    "sharpe_like_net": float(pnl.mean() / vol * np.sqrt(24 * 365)) if vol and np.isfinite(vol) and vol > 0 else np.nan,
                    "avg_active_gross_exposure": float(eq["active_gross_exposure"].mean()),
                    "avg_abs_active_net_exposure": float(eq["active_net_exposure"].abs().mean()),
                    "positions": int(len(pos)),
                    "sleeves": int(pos["sleeve_id"].nunique()) if not pos.empty else 0,
                    "median_holding_hours": float(pos["holding_hours"].median()) if not pos.empty else np.nan,
                    "universe_lost_exits": int(pos["exit_reason"].eq("universe_lost_exit").sum()) if not pos.empty else 0,
                }
            )
    return pd.DataFrame(rows)


def validation(equity: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, eq in equity.groupby("method", sort=False):
        pos = positions[positions["method"].eq(method)]
        for fee in FEE_BPS_LIST:
            curve = float(eq.sort_values("timestamp")[f"net_equity_{fee}bps"].iloc[-1])
            summed = float(pos[f"net_pnl_{fee}bps"].sum()) if not pos.empty else 0.0
            rows.append({"check": "final_equity_reconciliation", "method": method, "fee_bps": fee, "error": curve - summed, "pass_flag": abs(curve - summed) <= 1e-8})
        shorts = pos[pos["side"].eq("short")]
        expected = np.where(shorts["exit_price"] < shorts["entry_price"], shorts["gross_pnl"] > 0, shorts["gross_pnl"] <= 0) if not shorts.empty else []
        rows.append({"check": "short_sign_validation", "method": method, "fee_bps": np.nan, "error": int((~pd.Series(expected)).sum()) if len(expected) else 0, "pass_flag": int((~pd.Series(expected)).sum()) == 0 if len(expected) else True})
    return pd.DataFrame(rows)


def plot_equity(equity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for method, g in equity.groupby("method", sort=False):
        ax.plot(g["timestamp"], g["net_equity_5bps"], label=method, linewidth=1.0)
    ax.set_title("Converged mainline net equity, 5bps")
    ax.set_ylabel("Net equity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "converged_mainline_net_equity_5bps.png", dpi=160)
    plt.close(fig)


def compare_small_sample(signal: pd.DataFrame) -> pd.DataFrame:
    if not OLD_ADVANCED_SIGNAL.exists():
        return pd.DataFrame([{"check": "old_advanced_signal_exists", "pass_flag": False}])
    new_adv = signal[signal["method"].eq("advanced_lambda_0p5")].copy()
    old = pd.read_csv(OLD_ADVANCED_SIGNAL, parse_dates=["timestamp"], low_memory=False)
    old = old[old["method"].eq("advanced_lambda_0p5")].copy()
    keys = ["timestamp", "ticker", "method"]
    merged = new_adv.merge(old, on=keys, suffixes=("_new", "_old"), how="inner")
    rows = [{"check": "intersection_rows", "metric": len(merged), "pass_flag": len(merged) > 0}]
    for col in ["residual_return", "s_score", "ou_half_life_hours", "beta_factor1"]:
        if f"{col}_new" in merged and f"{col}_old" in merged and len(merged):
            diff = (pd.to_numeric(merged[f"{col}_new"], errors="coerce") - pd.to_numeric(merged[f"{col}_old"], errors="coerce")).abs()
            rows.append({"check": f"{col}_max_abs_diff", "metric": float(diff.max()), "pass_flag": float(diff.max()) < 1e-8})
    return pd.DataFrame(rows)


def write_report(summary_df: pd.DataFrame, validation_df: pd.DataFrame, compare_df: pd.DataFrame) -> None:
    lines = [
        "# Converged Mainline",
        "",
        "Dynamic eligible universe is used for both ordinary and advanced PCA: for timestamp `t`, PCA uses only returns in `[t-360h, t-1h]`; tickers with any missing value in that window or invalid current price/return are excluded and receive no s-score for that timestamp.",
        "",
        "Filters are simplified to finite price/return/s-score and `0 < half_life <= 90h`. No sigma percentile or R2 filter is applied. OU estimation itself only returns valid mean-reverting `0 < b < 1` fits.",
        "",
        "Existing positions are force-closed when the ticker leaves the raw no-lookahead universe. Ordinary PCA baseline uses equal-weight dollar-neutral sleeves with no soft-beta optimization; advanced PCA uses the soft beta optimizer.",
        "",
        f"The displayed mainline uses `gross_cap = {GROSS_CAP:g}`.",
        "",
        "## Performance",
        summary_df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Validation",
        validation_df.to_markdown(index=False, floatfmt=".8f"),
        "",
        "## Small Sample Reproduction Check",
        compare_df.to_markdown(index=False, floatfmt=".12f") if not compare_df.empty else "Not run.",
    ]
    (REPORT_DIR / "converged_mainline_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    configure_research()
    signal = normalize_signal(build_or_load_signal(args))
    signal.to_csv(OUT_DIR / "converged_signal_panel.csv", index=False)
    prices = load_price_panel()
    times = pd.DatetimeIndex([ts for ts in prices.index if ts >= signal["timestamp"].min() and ts <= signal["timestamp"].max()])
    universe_sets = build_universe_sets(times, set(signal["ticker"].unique()))
    all_equity = []
    all_positions = []
    all_sleeves = []
    for method in ["ordinary", "advanced_lambda_0p5"]:
        log(f"running final engine {method}")
        eq, pos, slv = run_engine(signal, method, prices, universe_sets)
        all_equity.append(eq)
        all_positions.append(pos)
        all_sleeves.append(slv)
    equity = pd.concat(all_equity, ignore_index=True)
    positions = pd.concat([p for p in all_positions if not p.empty], ignore_index=True)
    sleeves = pd.concat([s for s in all_sleeves if not s.empty], ignore_index=True)
    summary_df = summarize(equity, positions)
    validation_df = validation(equity, positions)
    compare_df = compare_small_sample(signal) if args.compare_existing else pd.DataFrame()
    equity.to_csv(OUT_DIR / "converged_hourly_equity.csv", index=False)
    positions.to_csv(OUT_DIR / "converged_positions.csv", index=False)
    sleeves.to_csv(OUT_DIR / "converged_sleeves.csv", index=False)
    summary_df.to_csv(REPORT_DIR / "converged_summary.csv", index=False)
    validation_df.to_csv(REPORT_DIR / "converged_validation.csv", index=False)
    compare_df.to_csv(REPORT_DIR / "small_sample_existing_comparison.csv", index=False)
    plot_equity(equity)
    write_report(summary_df, validation_df, compare_df)
    print(summary_df.to_string(index=False))
    if not compare_df.empty:
        print(compare_df.to_string(index=False))
    print(f"report: {REPORT_DIR / 'converged_mainline_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run converged ordinary/advanced PCA mainline.")
    parser.add_argument("--max-hours", type=int, default=None)
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--m-components", type=int, default=8)
    parser.add_argument("--min-assets", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--random-starts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--use-frozen-panels", action="store_true", help="Use audited final_pipeline ordinary and advanced signal panels instead of rebuilding PCA.")
    parser.add_argument("--compare-existing", action="store_true")
    args = parser.parse_args()
    if args.full_run:
        args.max_hours = None
    return args


if __name__ == "__main__":
    run(parse_args())
