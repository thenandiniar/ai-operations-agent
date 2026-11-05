import pandas as pd
import pytest

from tools.profiler import profile_dataframe


def test_profile_numeric_and_categorical_columns():
    dataframe = pd.DataFrame({
        "Revenue": [100, 200, 300],
        "Category": ["A", "B", "A"],
    })

    result = profile_dataframe(dataframe)

    assert result["overview"]["row_count"] == 3
    assert result["overview"]["column_count"] == 2
    assert result["columns"]["Revenue"]["statistics"]["mean"] == 200
    assert result["columns"]["Category"]["unique_values"] == 2


def test_detect_missing_values():
    dataframe = pd.DataFrame({
        "Revenue": [100, None, 300],
    })

    result = profile_dataframe(dataframe)

    assert result["overview"]["total_missing_cells"] == 1
    assert result["columns"]["Revenue"]["missing_count"] == 1


def test_detect_duplicate_rows():
    dataframe = pd.DataFrame({
        "Product": ["Laptop", "Mouse", "Laptop"],
        "Revenue": [50000, 800, 50000],
    })

    result = profile_dataframe(dataframe)

    assert result["overview"]["duplicate_rows"] == 1


def test_reject_invalid_input():
    with pytest.raises(TypeError):
        profile_dataframe([1, 2, 3])