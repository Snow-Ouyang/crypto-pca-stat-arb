from run_final_pipeline import build_naive_backtest, ensure_dirs, write_final_signal_rules


if __name__ == "__main__":
    ensure_dirs()
    write_final_signal_rules()
    build_naive_backtest()
