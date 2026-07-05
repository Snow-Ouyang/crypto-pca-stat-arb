# Converged Mainline

Dynamic eligible universe is used for both ordinary and advanced PCA: for timestamp `t`, PCA uses only returns in `[t-360h, t-1h]`; tickers with any missing value in that window or invalid current price/return are excluded and receive no s-score for that timestamp.

Filters are simplified to finite price/return/s-score and `0 < half_life <= 90h`. No sigma percentile or R2 filter is applied. OU estimation itself only returns valid mean-reverting `0 < b < 1` fits.

Existing positions are force-closed when the ticker leaves the raw no-lookahead universe. Ordinary PCA baseline uses equal-weight dollar-neutral sleeves with no soft-beta optimization; advanced PCA uses the soft beta optimizer.

The displayed mainline uses `gross_cap = 1.5`.

## Performance
| method              |   portfolio_lambda |   fee_bps |   final_gross_equity |   final_net_equity |   total_fees_paid |   max_drawdown_net |   sharpe_like_net |   avg_active_gross_exposure |   avg_abs_active_net_exposure |   positions |   sleeves |   median_holding_hours |   universe_lost_exits |
|:--------------------|-------------------:|----------:|---------------------:|-------------------:|------------------:|-------------------:|------------------:|----------------------------:|------------------------------:|------------:|----------:|-----------------------:|----------------------:|
| ordinary            |           0.000000 |         0 |             1.464851 |           1.464851 |          0.000000 |          -0.642354 |          1.358177 |                    1.371645 |                      0.390054 |        8165 |      2545 |              15.000000 |                   648 |
| ordinary            |           0.000000 |         5 |             1.464851 |           0.862807 |          0.602044 |          -0.720854 |          0.800751 |                    1.371645 |                      0.390054 |        8165 |      2545 |              15.000000 |                   648 |
| ordinary            |           0.000000 |        10 |             1.464851 |           0.260763 |          1.204088 |          -0.801334 |          0.242213 |                    1.371645 |                      0.390054 |        8165 |      2545 |              15.000000 |                   648 |
| advanced_lambda_0p5 |           3.000000 |         0 |             3.300959 |           3.300959 |          0.000000 |          -0.368850 |          3.410142 |                    1.332844 |                      0.292123 |        4598 |      1711 |              24.000000 |                   430 |
| advanced_lambda_0p5 |           3.000000 |         5 |             3.300959 |           2.816845 |          0.484114 |          -0.377507 |          2.912201 |                    1.332844 |                      0.292123 |        4598 |      1711 |              24.000000 |                   430 |
| advanced_lambda_0p5 |           3.000000 |        10 |             3.300959 |           2.332731 |          0.968228 |          -0.386164 |          2.413133 |                    1.332844 |                      0.292123 |        4598 |      1711 |              24.000000 |                   430 |

## Validation
| check                       | method              |      fee_bps |       error | pass_flag   |
|:----------------------------|:--------------------|-------------:|------------:|:------------|
| final_equity_reconciliation | ordinary            |   0.00000000 | -0.00000000 | True        |
| final_equity_reconciliation | ordinary            |   5.00000000 | -0.00000000 | True        |
| final_equity_reconciliation | ordinary            |  10.00000000 | -0.00000000 | True        |
| short_sign_validation       | ordinary            | nan          |  0.00000000 | True        |
| final_equity_reconciliation | advanced_lambda_0p5 |   0.00000000 | -0.00000000 | True        |
| final_equity_reconciliation | advanced_lambda_0p5 |   5.00000000 | -0.00000000 | True        |
| final_equity_reconciliation | advanced_lambda_0p5 |  10.00000000 | -0.00000000 | True        |
| short_sign_validation       | advanced_lambda_0p5 | nan          |  0.00000000 | True        |

## Small Sample Reproduction Check
Not run.