
import pytest

from core.planner import plan_analysis, PlannerError


SAMPLE_COLUMNS = [
    "OrderID",
    "Product",
    "Category",
    "Revenue",
    "Quantity",
]


def test_plan_sales_analysis():
    plan = plan_analysis(
        "Analyze my sales data",
        SAMPLE_COLUMNS,
    )

    tool_names = [action.tool for action in plan.actions]

    assert "analyze_sales" in tool_names
    assert plan.planner_type == "rule_based"


def test_plan_anomaly_detection():
    plan = plan_analysis(
        "Find unusual revenue values",
        SAMPLE_COLUMNS,
    )

    anomaly_actions = [
        action for action in plan.actions
        if action.tool == "detect_iqr_anomalies"
    ]

    assert len(anomaly_actions) == 1
    assert anomaly_actions[0].parameters["column"] == "Revenue"


def test_plan_visualization():
    plan = plan_analysis(
        "Create charts for my sales data",
        SAMPLE_COLUMNS,
    )

    tool_names = [action.tool for action in plan.actions]

    assert "create_category_revenue_chart" in tool_names
    assert "create_revenue_distribution_chart" in tool_names


def test_reject_unsupported_request():
    with pytest.raises(PlannerError):
        plan_analysis("Tell me a joke", SAMPLE_COLUMNS)


def test_reject_empty_goal():
    with pytest.raises(PlannerError):
        plan_analysis("", SAMPLE_COLUMNS)
