import pandas as pd
import pytest

from tools.visualizer import (
    create_category_revenue_chart,
    create_revenue_distribution_chart,
    create_correlation_heatmap,
    save_chart,
)


def sample_data():
    return pd.DataFrame({
        "Category": [
            "Electronics", "Accessories", "Accessories",
            "Electronics", "Furniture", "Furniture"
        ],
        "Revenue": [55000, 800, 1500, 12000, 7500, 4500],
        "Quantity": [2, 5, 3, 2, 1, 2],
    })


def test_category_revenue_chart():
    figure = create_category_revenue_chart(sample_data())

    assert figure.data
    assert figure.layout.title.text == "Revenue by Category"


def test_revenue_distribution_chart():
    figure = create_revenue_distribution_chart(sample_data())

    assert figure.data
    assert figure.layout.title.text == "Revenue Distribution"


def test_correlation_heatmap():
    figure = create_correlation_heatmap(sample_data())

    assert figure.data
    assert figure.layout.title.text == "Numerical Correlation Heatmap"


def test_save_chart(tmp_path):
    figure = create_category_revenue_chart(sample_data())
    output = tmp_path / "chart.html"

    saved_path = save_chart(figure, str(output))

    assert output.exists()
    assert output.stat().st_size > 0
    assert saved_path.endswith("chart.html")


def test_reject_missing_column():
    with pytest.raises(ValueError):
        create_category_revenue_chart(
            sample_data(),
            category_column="Unknown"
        )