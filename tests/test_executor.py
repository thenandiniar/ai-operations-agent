
import pandas as pd
import pytest

from core.planner import plan_analysis
from core.executor import execute_plan


SAMPLE_COLUMNS = [
    "OrderID",
    "Product",
    "Category",
    "Revenue",
    "Quantity",
]


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


def test_execute_sales_analysis():
    dataframe = sample_data()

    plan = plan_analysis(
        "Analyze my sales data",
        SAMPLE_COLUMNS,
    )

    report = execute_plan(plan, dataframe)

    assert report.status == "success"
    assert report.successful_actions >= 1
    assert report.failed_actions == 0

    sales_result = next(
        item.result
        for item in report.results
        if item.tool == "analyze_sales"
    )

    assert sales_result["total_revenue"] == 81300


def test_execute_visualization_and_save_charts(tmp_path):
    dataframe = sample_data()

    plan = plan_analysis(
        "Create charts for my sales data",
        SAMPLE_COLUMNS,
    )

    report = execute_plan(
        plan,
        dataframe,
        output_dir=str(tmp_path),
    )

    assert report.status == "success"

    assert (tmp_path / "category_revenue.html").exists()
    assert (tmp_path / "revenue_distribution.html").exists()


def test_reject_invalid_dataframe():
    plan = plan_analysis(
        "Analyze my sales data",
        SAMPLE_COLUMNS,
    )

    with pytest.raises(TypeError):
        execute_plan(plan, [1, 2, 3])


def test_reject_empty_dataframe():
    plan = plan_analysis(
        "Analyze my sales data",
        SAMPLE_COLUMNS,
    )

    with pytest.raises(ValueError):
        execute_plan(plan, pd.DataFrame())


def test_stop_after_first_failure():
    dataframe = pd.DataFrame({
        "Revenue": [100, 200, 300],
        "Quantity": [1, 2, 3],
    })

    plan = plan_analysis(
        "Find unusual revenue values",
        SAMPLE_COLUMNS,
    )

    report = execute_plan(
        plan,
        dataframe,
        continue_on_error=False,
    )

    assert report.status == "failed"
    assert report.failed_actions == 1
    assert len(report.results) == 1
