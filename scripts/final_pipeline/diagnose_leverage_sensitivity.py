from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONVERGED_SCRIPT = PROJECT_ROOT / "scripts" / "final_pipeline" / "09_run_converged_mainline.py"
OUT_DATA = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "leverage_sensitivity"
OUT_REPORT = PROJECT_ROOT / "reports" / "final_report" / "leverage_sensitivity"
FIG_DIR = OUT_REPORT / "figures"
SIGNAL_PATH = PROJECT_ROOT / "data" / "processed" / "final_pipeline" / "converged_mainline" / "converged_signal_panel.csv"
LEVERAGE_LEVELS = [1.0, 1.5, 2.0, 2.5]
METHOD_ORDER = ["ordinary", "advanced_lambda_0p5"]
METHOD_LABELS = {
    "ordinary": "Ordinary PCA equal-weight",
    "advanced_lambda_0p5": "Advanced PCA + optimizer",
}


def load_converged_module():
    spec = importlib.util.spec_from_file_location("converged_mainline", CONVERGED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CONVERGED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def max_drawdown(series: pd.Series) -> float:
    return float((series - series.cummax()).min()) if len(series) else np.nan


def run_sweep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    converged = load_converged_module()
    signal = pd.read_csv(SIGNAL_PATH, parse_dates=["timestamp"], low_memory=False)
    signal = converged.normalize_signal(signal)
    prices = converged.load_price_panel()
    times = pd.DatetimeIndex([ts for ts in prices.index if signal["timestamp"].min() <= ts <= signal["timestamp"].max()])
    universe_sets = converged.build_universe_sets(times, set(signal["ticker"].unique()))

    all_equity: list[pd.DataFrame] = []
    all_positions: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    original_gross_cap = converged.GROSS_CAP

    try:
        for gross_cap in LEVERAGE_LEVELS:
            converged.GROSS_CAP = gross_cap
            for method in METHOD_ORDER:
                print(f"[leverage_sensitivity] gross_cap={gross_cap:g} method={method}", flush=True)
                equity, positions, _sleeves = converged.run_engine(signal, method, prices, universe_sets)
                equity["gross_cap"] = gross_cap
                positions["gross_cap"] = gross_cap
                summary = converged.summarize(equity, positions)
                summary["gross_cap"] = gross_cap
                all_equity.append(equity)
                all_positions.append(positions)
                all_summary.append(summary)
    finally:
        converged.GROSS_CAP = original_gross_cap

    equity_out = pd.concat(all_equity, ignore_index=True)
    positions_out = pd.concat([p for p in all_positions if not p.empty], ignore_index=True)
    summary_out = pd.concat(all_summary, ignore_index=True)
    return equity_out, positions_out, summary_out


def build_leverage_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary[summary["fee_bps"].eq(5)].copy()
    table["model"] = table["method"].map(METHOD_LABELS).fillna(table["method"])
    table["final_net_equity_per_gross_cap"] = table["final_net_equity"] / table["gross_cap"]
    table["max_drawdown_per_gross_cap"] = table["max_drawdown_net"] / table["gross_cap"]
    table["avg_gross_exposure_to_cap"] = table["avg_active_gross_exposure"] / table["gross_cap"]
    cols = [
        "model",
        "method",
        "gross_cap",
        "fee_bps",
        "final_net_equity",
        "final_net_equity_per_gross_cap",
        "max_drawdown_net",
        "max_drawdown_per_gross_cap",
        "sharpe_like_net",
        "avg_active_gross_exposure",
        "avg_gross_exposure_to_cap",
        "positions",
        "sleeves",
    ]
    return table[cols].sort_values(["method", "gross_cap"]).reset_index(drop=True)


def plot_equity(equity: pd.DataFrame) -> None:
    fee_col = "net_equity_5bps"
    for method in METHOD_ORDER:
        fig, ax = plt.subplots(figsize=(12, 5))
        sub = equity[equity["method"].eq(method)].copy()
        for gross_cap, g in sub.groupby("gross_cap", sort=True):
            ax.plot(g["timestamp"], g[fee_col], label=f"{gross_cap:g}x", linewidth=1.0)
        ax.set_title(f"{METHOD_LABELS[method]} net equity by gross cap, 5bps")
        ax.set_ylabel("Net equity")
        ax.grid(alpha=0.25)
        ax.legend(title="Gross cap")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{method}_net_equity_by_leverage_5bps.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, method in zip(axes, METHOD_ORDER):
        sub = equity[equity["method"].eq(method)].copy()
        for gross_cap, g in sub.groupby("gross_cap", sort=True):
            ax.plot(g["timestamp"], g[fee_col], label=f"{gross_cap:g}x", linewidth=1.0)
        ax.set_title(METHOD_LABELS[method])
        ax.set_ylabel("Net equity")
        ax.grid(alpha=0.25)
        ax.legend(title="Gross cap", ncol=4)
    axes[-1].set_xlabel("Timestamp")
    fig.suptitle("Leverage sensitivity: net equity by gross cap, 5bps")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_advanced_net_equity_by_leverage_5bps.png", dpi=160)
    plt.close(fig)


def plot_normalized_bar(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(LEVERAGE_LEVELS))
    width = 0.36
    for i, method in enumerate(METHOD_ORDER):
        sub = table[table["method"].eq(method)].set_index("gross_cap").reindex(LEVERAGE_LEVELS)
        offset = -width / 2 if i == 0 else width / 2
        ax.bar(x + offset, sub["final_net_equity_per_gross_cap"], width, label=METHOD_LABELS[method])
    ax.set_xticks(x, [f"{v:g}x" for v in LEVERAGE_LEVELS])
    ax.set_title("Final net equity divided by gross cap, 5bps")
    ax.set_ylabel("Final net equity / gross cap")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_equity_per_leverage_bar_5bps.png", dpi=160)
    plt.close(fig)


def write_report(table: pd.DataFrame) -> None:
    lines = [
        "# Leverage Sensitivity Diagnostic",
        "",
        "This diagnostic reruns the converged final engine with the same signal panels and parameters, changing only `gross_cap`.",
        "",
        "Gross cap levels: `1.0`, `1.5`, `2.0`, `2.5`.",
        "",
        "## 5bps Summary",
        "",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Figures",
        "",
        "![Combined equity](figures/baseline_advanced_net_equity_by_leverage_5bps.png)",
        "",
        "![Final equity per leverage](figures/final_equity_per_leverage_bar_5bps.png)",
    ]
    (OUT_REPORT / "leverage_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    equity, positions, summary = run_sweep()
    table = build_leverage_table(summary)
    equity.to_csv(OUT_DATA / "leverage_sensitivity_hourly_equity.csv", index=False)
    positions.to_csv(OUT_DATA / "leverage_sensitivity_positions.csv", index=False)
    summary.to_csv(OUT_REPORT / "leverage_sensitivity_summary.csv", index=False)
    table.to_csv(OUT_REPORT / "leverage_sensitivity_equity_per_leverage.csv", index=False)
    (OUT_REPORT / "leverage_sensitivity_equity_per_leverage.md").write_text(table.to_markdown(index=False, floatfmt=".4f") + "\n", encoding="utf-8")
    plot_equity(equity)
    plot_normalized_bar(table)
    write_report(table)
    print(table.to_string(index=False))
    print(f"report: {OUT_REPORT / 'leverage_sensitivity_report.md'}")


if __name__ == "__main__":
    main()
