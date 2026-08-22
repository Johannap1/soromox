import sys
from pathlib import Path

import pytest

MODULE_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secIVb_parallel_rollouts_gpu"
    / "code"
)
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import generate_benchmark_gpu  # noqa: E402


def test_output_paths_default_to_case_data_directory():
    args = generate_benchmark_gpu.parse_args([])

    assert args.csv == generate_benchmark_gpu.DATA_DIR / "benchmark_results.csv"


def test_csv_output_path_can_be_overridden(tmp_path):
    csv_path = tmp_path / "custom.csv"

    args = generate_benchmark_gpu.parse_args(["--csv", str(csv_path)])

    assert args.csv == csv_path


def test_npz_output_option_is_not_supported():
    with pytest.raises(SystemExit):
        generate_benchmark_gpu.parse_args(["--npz", "results.npz"])
