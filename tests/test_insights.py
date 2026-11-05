
import pandas as pd
import pytest

from tools.insights import (
    generate_business_insights,
    InsightsError,
)


@pytest.fixture
def sales_data():
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


def test_generate_business_insights(sales_data):
    result = generate_business_insights(sales_data)

    assert result["key_metrics"]["total_revenue"] == 81300
    assert result["key_metrics"]["total_quantity"] == 15
    assert result["anomaly_count"] == 1
    assert len(result["findings"]) >= 3


def test_identify_top_category(sales_data):
    result = generate_business_insights(sales_data)

    finding = next(
        item for item in result["findings"]
        if item["type"] == "category_performance"
    )

    assert finding["evidence"]["category"] == "Electronics"
    assert finding["evidence"]["revenue"] == 67000


def test_detect_revenue_concentration(sales_data):
    result = generate_business_insights(sales_data)

    concentration = [
        item for item in result["findings"]
        if item["type"] == "revenue_concentration"
    ]

    assert len(concentration) == 1
    assert concentration[0]["evidence"]["share_percent"] > 70


def test_reject_empty_dataframe():
    with pytest.raises(InsightsError):
        generate_business_insights(pd.DataFrame())


def test_reject_invalid_input():
    with pytest.raises(InsightsError):
        generate_business_insights([1, 2, 3])


def test_reject_missing_required_columns():
    dataframe = pd.DataFrame({
        "Revenue": [100, 200],
        "Quantity": [1, 2],
    })

    with pytest.raises(ValueError):
        generate_business_insights(dataframe)
