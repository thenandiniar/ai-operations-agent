
import pytest

from llm import planner
from llm.planner import (
    GeminiPlanResponse,
    GeminiPlanAction,
    GeminiPlannerError,
    plan_with_gemini,
)


COLUMNS = ["OrderID", "Product", "Category", "Revenue", "Quantity"]


def mock_response(actions):
    return GeminiPlanResponse(
        goal="Analyze sales",
        actions=actions,
    )


def test_gemini_creates_valid_sales_plan(monkeypatch):
    response = mock_response([
        GeminiPlanAction(
            tool="analyze_sales",
            reason="Calculate sales metrics.",
        )
    ])

    monkeypatch.setattr(
        planner,
        "generate_structured_response",
        lambda prompt, schema: response,
    )

    plan = plan_with_gemini("Analyze my sales", COLUMNS)

    assert plan.planner_type == "gemini"
    assert plan.actions[0].tool == "analyze_sales"


def test_gemini_plan_detects_anomalies(monkeypatch):
    response = mock_response([
        GeminiPlanAction(
            tool="detect_iqr_anomalies",
            column="Revenue",
            reason="Find unusual revenue values.",
        )
    ])

    monkeypatch.setattr(
        planner,
        "generate_structured_response",
        lambda prompt, schema: response,
    )

    plan = plan_with_gemini("Find unusual revenue", COLUMNS)

    assert plan.actions[0].tool == "detect_iqr_anomalies"
    assert plan.actions[0].parameters == {"column": "Revenue"}


def test_reject_unknown_tool(monkeypatch):
    response = mock_response([
        GeminiPlanAction(
            tool="execute_python",
            reason="Invalid tool.",
        )
    ])

    monkeypatch.setattr(
        planner,
        "generate_structured_response",
        lambda prompt, schema: response,
    )

    with pytest.raises(GeminiPlannerError):
        plan_with_gemini("Analyze sales", COLUMNS)


def test_reject_invalid_column(monkeypatch):
    response = mock_response([
        GeminiPlanAction(
            tool="detect_iqr_anomalies",
            column="FakeColumn",
            reason="Find anomalies.",
        )
    ])

    monkeypatch.setattr(
        planner,
        "generate_structured_response",
        lambda prompt, schema: response,
    )

    with pytest.raises(GeminiPlannerError):
        plan_with_gemini("Find anomalies", COLUMNS)


def test_reject_empty_actions(monkeypatch):
    response = mock_response([])

    monkeypatch.setattr(
        planner,
        "generate_structured_response",
        lambda prompt, schema: response,
    )

    with pytest.raises(GeminiPlannerError):
        plan_with_gemini("Analyze sales", COLUMNS)


def test_reject_duplicate_tools(monkeypatch):
    response = mock_response([
        GeminiPlanAction(
            tool="analyze_sales",
            reason="First analysis.",
        ),
        GeminiPlanAction(
            tool="analyze_sales",
            reason="Duplicate analysis.",
        ),
    ])

    monkeypatch.setattr(
        planner,
        "generate_structured_response",
        lambda prompt, schema: response,
    )

    with pytest.raises(GeminiPlannerError):
        plan_with_gemini("Analyze sales", COLUMNS)
