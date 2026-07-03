# Crypto PCA Residual Statistical Arbitrage

## 1. Overview

This project builds an hourly crypto PCA residual statistical arbitrage research pipeline. The converged mainline compares ordinary PCA residuals against a residual-comovement-penalized advanced PCA model under the same audited backtest engine.

Main converged 5bps result: ordinary PCA equal-weight net equity `1.4295`, max drawdown `-1.1939`, Sharpe-like `0.7970`; advanced PCA + optimizer net equity `4.7158`, max drawdown `-0.6293`, Sharpe-like `2.9666`.

## 2. Data and No-Lookahead Universe

The research uses hourly close data and the no-lookahead universe work from the earlier data-quality stage. Universe membership is treated as known only when it would have been observable. Structural zero prices outside the active universe are not treated as active data-quality failures.

## 3. PCA Factor Diagnostics

Final PCA setting is W360 / PC3. PC1 behaves like a broad crypto market factor. PC2 and PC3 capture secondary relative structure that is useful for residual construction but less stable than PC1. Ordinary PCA focuses on explained variance; advanced PCA adds a residual-comovement penalty so that residuals are cleaner for statistical arbitrage, not only better explained by factors.

![Explained variance](figures/pca/explained_variance_ratio_over_time.png)

![PC loadings](figures/pca/pc_loading_bar_charts_selected_dates.png)

![Eigenportfolio returns](figures/pca/eigenportfolio_cumulative_returns.png)

## 4. OU Residual Modeling

Not every residual is suitable for OU modeling. In AR(1) terms, `0 < b < 1` is the stable mean-reverting region; `b <= 0` is unstable for this signal interpretation, and `b >= 1` is near-unit-root or explosive. The converged mainline keeps the entry filter simple: finite price / return / s-score and `0 < half_life <= 90h`.

![Good OU residual](figures/ou/good_ou_residual_example.png)

![Bad OU residual](figures/ou/b_greater_than_one_bad_residual_example.png)

## 5. Signal Construction

Signals use OU residual s-scores. Earlier diagnostics explored half-life bucket thresholds, but the converged mainline uses fixed thresholds: long entry/exit `1 / 0.5` and short entry/exit `1 / 0.25`, with `0 < half_life <= 90h`.

Final signal rules are saved in `tables/final_signal_rules.csv`.

## 6. Naive 1-Dollar Backtest

The naive signal backtest shows alpha exists in the signal stream, but 1-dollar-per-position sizing creates uncontrolled exposure. This motivates the matched-sleeve portfolio construction.

![Naive long-short curves](figures/naive/naive_long_short_equity_curves.png)

![Naive exposure curves](figures/naive/naive_exposure_curves.png)

## 7. Matched-Sleeve Dollar-Neutral Portfolio

The portfolio opens sleeves that match long and short notional at entry. Dollar neutrality is enforced only when a new sleeve is opened. Existing positions are not resized, there is no hourly target-weight rebalance, and fees are charged only on entry and exit. Gross cap is applied only when opening a new sleeve.

![Matched sleeve gross exposure](figures/portfolio/matched_sleeve_gross_exposure.png)

![Final actual exposure](figures/portfolio/final_strategy_actual_exposure_timeseries.png)

## 8. Soft z-PC1 Beta Penalty Optimizer Diagnostic

The retained optimizer diagnostic minimizes distance to equal-weight sizing plus a soft z-PC1 exposure penalty:

`minimize distance_to_equal_weight + lambda_portfolio_zbeta * z_PC1_exposure^2`

Hard constraints enforce equal long and short notional at sleeve entry, nonnegative long/short notionals, concentration cap, and gross-cap capacity. The z-PC1 term is a soft penalty, not a hard equality constraint. Hard PC1 neutrality was often infeasible; soft penalty preserves sleeve coverage while reducing relative beta mismatch.

![Soft PC1 vs equal weight](figures/portfolio/soft_pc1_vs_equal_weight_equity_5bps.png)

![Soft PC1 drawdown](figures/portfolio/soft_pc1_vs_equal_weight_drawdown_5bps.png)

## 9. Retained Matched-Sleeve Benchmark

This section is retained as a technical benchmark from the earlier matched-sleeve / soft z-PC1 optimizer line. The converged ordinary-vs-advanced mainline is reported in Section 11.

|   fee_bps |   final_net_equity |   max_drawdown_net |   sharpe_like_net |   total_fees_paid |
|----------:|-------------------:|-------------------:|------------------:|------------------:|
|    0.0000 |             3.6426 |            -0.5859 |            2.5140 |            0.0000 |
|    5.0000 |             2.7607 |            -0.6298 |            1.9077 |            0.8818 |
|   10.0000 |             1.8789 |            -0.7107 |            1.2997 |            1.7637 |

Benchmark comparison:

| config                              |   final_net_equity |   max_drawdown_net |   sharpe_like_net |   avg_active_gross_exposure |
|:------------------------------------|-------------------:|-------------------:|------------------:|----------------------------:|
| baseline_equal_weight_gross_cap_2p5 |             1.7314 |            -0.8305 |            1.2608 |                      2.1804 |
| soft_pc1_z_lambda_100_gross_cap_2p0 |             2.1092 |            -0.5056 |            1.8103 |                      1.7637 |
| soft_pc1_z_lambda_100_gross_cap_2p5 |             2.7607 |            -0.6298 |            1.9077 |                      2.2014 |

![Final equity](figures/attribution/final_0_5_10bps_equity_curve.png)

![Final drawdown](figures/attribution/final_drawdown_5bps.png)

## 10. Short Bad-Trade Mechanism

The retained bad-trade diagnostic focuses on short-side loss mechanisms from the naive 1-dollar-per-position stage. The representative cases show adverse s-score continuation, sigma expansion, and holding periods extending beyond the estimated half-life. The final mainline keeps the filter set deliberately simple.

|   case_id | mechanism                                     | side   | ticker   |   net_pnl_5bps |   holding_hours |   entry_half_life |   max_sigma_pct_during_trade |
|----------:|:----------------------------------------------|:-------|:---------|---------------:|----------------:|------------------:|-----------------------------:|
|         1 | short_squeeze_adverse_continuation            | short  | SAND     |        -0.3647 |        155.0000 |           82.4176 |                       0.9500 |
|         2 | slow_mean_reversion_holding_exceeds_half_life | short  | RSR      |        -0.5644 |        191.0000 |            7.0520 |                       0.8684 |
|         3 | model_instability_sigma_expansion             | short  | OMG      |        -0.3844 |        111.0000 |           11.3887 |                       0.9250 |

![Short bad trade case 1](figures/bad_trades/case_1_trade_lifecycle_4panel.png)

![Short bad trade case 2](figures/bad_trades/case_2_trade_lifecycle_4panel.png)

![Short bad trade case 3](figures/bad_trades/case_3_trade_lifecycle_4panel.png)

Detailed mechanism report: `bad_trade_mechanism_report.md`.

## 11. Converged Ordinary vs Advanced PCA

The converged mainline uses the audited dynamic eligible universe. For timestamp `t`, PCA uses only `[t-360h, t-1h]`; tickers with missing values in that window are excluded and receive no s-score for that timestamp. The final filter set is intentionally simple: finite price/return/s-score and `0 < half_life <= 90h`. OU estimation itself only admits valid `0 < b < 1` fits. There is no sigma percentile or R2 entry filter.

Advanced PCA fixes `lambda_pca_comovement = 0.5`; ordinary PCA uses equal-weight dollar-neutral sleeves, and advanced PCA uses portfolio soft beta penalty `lambda_portfolio_zbeta = 3.0`. Positions are force-closed when the ticker leaves the no-lookahead universe.

| method              |   portfolio_lambda |   fee_bps |   final_net_equity |   max_drawdown_net |   sharpe_like_net |   avg_active_gross_exposure |   universe_lost_exits |
|:--------------------|-------------------:|----------:|-------------------:|-------------------:|------------------:|----------------------------:|----------------------:|
| ordinary            |             0.0000 |         5 |             1.4295 |            -1.1939 |            0.7970 |                      2.2858 |                   648 |
| advanced_lambda_0p5 |             3.0000 |         5 |             4.7158 |            -0.6293 |            2.9666 |                      2.2201 |                   431 |

![Converged mainline equity](converged_mainline/figures/converged_mainline_net_equity_5bps.png)

Detailed converged report: `converged_mainline/converged_mainline_report.md`.

## 12. Attribution

Long and short both contribute meaningfully at 5bps:

| side   |   position_count |   net_pnl |   total_fees |   win_rate |   median_holding_hours |
|:-------|-----------------:|----------:|-------------:|-----------:|-----------------------:|
| long   |             2521 |    1.3718 |       0.4418 |     0.5534 |                17.0000 |
| short  |             2692 |    1.3890 |       0.4400 |     0.5691 |                18.0000 |

Half-life bucket attribution highlights that the strategy is not uniformly strong across buckets:

| side   | hl_bucket   |   position_count |   net_pnl |   win_rate |   median_holding_hours |
|:-------|:------------|-----------------:|----------:|-----------:|-----------------------:|
| long   | hl_36_60    |              237 |    0.9671 |     0.5949 |                34.0000 |
| short  | hl_18_36    |              860 |    0.8526 |     0.5581 |                29.0000 |
| short  | hl_9_18     |              907 |    0.4668 |     0.5921 |                16.0000 |
| long   | hl_0_9      |              311 |    0.3213 |     0.6013 |                 8.0000 |
| long   | hl_60_90    |              240 |    0.3141 |     0.5042 |                 5.0000 |
| long   | hl_9_18     |              846 |    0.2822 |     0.5544 |                15.0000 |
| short  | hl_60_90    |              275 |    0.2761 |     0.5127 |                 7.0000 |
| short  | hl_0_9      |              317 |    0.0213 |     0.6025 |                10.0000 |
| short  | hl_36_60    |              333 |   -0.2279 |     0.5495 |                28.0000 |
| long   | hl_18_36    |              887 |   -0.5129 |     0.5378 |                27.0000 |

Largest drawdown:

| peak_time                 | trough_time               | recovery_time             |   drawdown_depth |   duration_hours |   long_pnl_during_dd |   short_pnl_during_dd |
|:--------------------------|:--------------------------|:--------------------------|-----------------:|-----------------:|---------------------:|----------------------:|
| 2021-10-28 10:00:00+00:00 | 2021-11-26 11:00:00+00:00 | 2022-02-05 04:00:00+00:00 |          -0.6298 |         697.0000 |              -0.1014 |               -0.4147 |

The largest drawdown is mostly short-side driven. Future work can consider short-side risk control or regime filters, but this final report deliberately avoids further in-sample optimization.

## 13. Caveats

- Same-close execution is optimistic.
- Hourly data cannot verify intrabar fills.
- No true bid-ask spread, slippage, or market impact is modeled.
- Short borrow, funding, and availability are not modeled.
- `gross_cap=2.5` is moderate leverage.
- The strategy still needs out-of-sample / walk-forward validation.

## 14. Future Work

- Out-of-sample / walk-forward validation.
- More realistic execution and cost model.
- Funding and borrow cost modeling.
- Liquidity or rank penalty.
- Short-side regime filter.
- Possible bucket refinement, but not in this report to avoid overfitting.
