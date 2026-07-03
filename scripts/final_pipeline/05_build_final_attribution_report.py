from run_final_pipeline import build_final_attribution, ensure_dirs, validate_outputs, write_report


if __name__ == "__main__":
    ensure_dirs()
    build_final_attribution()
    write_report()
    print(validate_outputs())
