from pathlib import Path

import pandas as pd
import pytest

from tools.csv_reader import load_csv, CSVLoadError


def test_load_valid_csv():
    result = load_csv(Path("data/sample_sales.csv"))

    assert result.rows == 6
    assert len(result.columns) == 5
    assert result.missing_cells == 0
    assert "Revenue" in result.columns


def test_missing_file():
    with pytest.raises(CSVLoadError):
        load_csv("data/does_not_exist.csv")


def test_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_bytes(b"")

    with pytest.raises(CSVLoadError):
        load_csv(empty_file)