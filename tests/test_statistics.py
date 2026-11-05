
import pandas as pd
import pytest

from tools.statistics import (
    analyze_sales,
    calculate_correlations,
    detect_iqr_anomalies,
)


def sample_data():
    return pd.DataFrame({
        "OrderID": [101, 102, 103, 104, 105, 106],
        "Product": [
            "Laptop", "Mouse", "Keyboard",
            "Monitor", "Desk", "Chair"
        ],
        "Category": [
            "Electronics", "Accessories", "Accessories",
            "Electronics", "Furniture", "Furniture"
        ],
        "Revenue": [55000, 800, 1500, 12000, 7500, 4500],
        "Quantity": [2, 5, 3, 2, 1, 2],
    })


def test_sales_metrics():
    result = analyze_sales(sample_data())

    assert result["total_revenue"] == 81300
    assert result["average_revenue"] == 13550
    assert result["total_quantity"] == 15


def test_category_revenue():
    result = analyze_sales(sample_data())

    assert result["category_revenue"]["Electronics"] == 67000
    assert result["category_revenue"]["Accessories"] == 2300
    assert result["category_revenue"]["Furniture"] == 12000


def test_iqr_anomaly_detection():
    result = detect_iqr_anomalies(sample_data(), "Revenue")

    assert result["anomaly_count"] >= 1
    assert result["anomalies"][0]["Revenue"] == 55000


def test_correlations():
    result = calculate_correlations(sample_data())

    assert "Revenue" in result
    assert "Quantity" in result
    assert "Revenue" in result["Quantity"]


def test_missing_required_column():
    dataframe = pd.DataFrame({
        "Revenue": [100, 200],
        "Quantity": [1, 2],
    })

    with pytest.raises(ValueError):
        analyze_sales(dataframe)


def test_correlations_exclude_identifier_columns():
    result = calculate_correlations(sample_data())

    assert "OrderID" not in result
    assert "Revenue" in result
    assert "Quantity" in result
    assert "OrderID" not in result["Revenue"]


def test_correlations_return_empty_when_only_one_measurement_exists():
    dataframe = pd.DataFrame({
        "OrderID": [101, 102, 103],
        "Revenue": [100, 200, 300],
    })

    result = calculate_correlations(dataframe)

    assert result == {}
