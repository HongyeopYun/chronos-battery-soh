"""Run the paired C2 output-head-only experiment, reusing the full-tuning protocol."""
from run_c2_fewshot_full_tuning import main


if __name__ == "__main__":
    main(default_mode="head")
