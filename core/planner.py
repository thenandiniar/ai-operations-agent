from typing import Any, Literal

from pydantic import BaseModel, Field


class PlannerError(Exception):
    """Raised when a user request cannot be converted into a plan."""


class PlanAction(BaseModel):
    tool: Literal[
        "profile_dataframe",
        "analyze_sales",
        "calculate_correlations",
        "detect_iqr_anomalies",
        "generate_business_insights",
        "create_category_revenue_chart",
        "create_revenue_distribution_chart",
        "create_correlation_heatmap",
    ]

    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str


class AnalysisPlan(BaseModel):
    goal: str
    actions: list[PlanAction]
    planner_type: str = "rule_based"


def plan_analysis(
    user_goal: str,
    available_columns: list[str],
) -> AnalysisPlan:
    """
    Convert a natural-language business request into a tool plan.

    This is the rule-based planner and does not call an LLM.
    """

    if not user_goal or not user_goal.strip():
        raise PlannerError("Please provide an analysis goal.")

    if not available_columns:
        raise PlannerError("No dataset columns are available.")

    goal = user_goal.strip()
    normalized_goal = goal.lower()

    actions = []
    selected_tools = set()

    def add_action(tool, parameters=None, reason=""):
        if tool in selected_tools:
            return

        actions.append(
            PlanAction(
                tool=tool,
                parameters=parameters or {},
                reason=reason,
            )
        )
        selected_tools.add(tool)

    # Dataset profiling
    if any(word in normalized_goal for word in [
        "profile", "overview", "dataset", "data quality"
    ]):
        add_action(
            "profile_dataframe",
            reason="Inspect dataset structure and quality.",
        )

    # Business and sales analysis
    if any(word in normalized_goal for word in [
        "sales", "revenue", "business", "analyze", "analysis"
    ]):
        if all(column in available_columns for column in [
            "Revenue", "Quantity", "Category"
        ]):
            add_action(
                "analyze_sales",
                reason="Calculate business metrics and category performance.",
            )

    # Business insights and recommendations
    if any(word in normalized_goal for word in [
        "insight", "insights", "finding", "findings",
        "recommendation", "recommendations",
        "summary", "summarize", "comprehensive",
        "executive summary",
    ]):
        add_action(
            "generate_business_insights",
            reason=(
                "Generate evidence-based business findings, "
                "key metrics and recommendations."
            ),
        )

    # Correlation analysis
    if any(word in normalized_goal for word in [
        "correlation", "relationship", "related"
    ]):
        add_action(
            "calculate_correlations",
            reason="Measure relationships between numeric columns.",
        )

        add_action(
            "create_correlation_heatmap",
            reason="Visualize correlations between numeric columns.",
        )

    # Anomaly detection
    if any(word in normalized_goal for word in [
        "anomaly", "anomalies", "outlier", "outliers",
        "unusual", "abnormal"
    ]):
        numeric_candidates = [
            column for column in available_columns
            if column.lower() in [
                "revenue", "quantity", "amount", "price"
            ]
        ]

        if not numeric_candidates:
            raise PlannerError(
                "No supported numeric column was found for anomaly detection."
            )

        target_column = (
            "Revenue" if "Revenue" in numeric_candidates
            else numeric_candidates[0]
        )

        add_action(
            "detect_iqr_anomalies",
            parameters={"column": target_column},
            reason=f"Find unusual values in {target_column}.",
        )

    # Visualization
    if any(word in normalized_goal for word in [
        "chart", "plot", "visualize", "visualization", "graph"
    ]):
        if (
            "Revenue" in available_columns
            and "Category" in available_columns
        ):
            add_action(
                "create_category_revenue_chart",
                reason="Compare revenue across categories.",
            )

        if "Revenue" in available_columns:
            add_action(
                "create_revenue_distribution_chart",
                reason="Show the distribution of revenue values.",
            )

    if not actions:
        raise PlannerError(
            "I could not identify a supported analysis task. "
            "Try asking for a data overview, sales analysis, "
            "business insights, correlations, anomalies or charts."
        )

    return AnalysisPlan(
        goal=goal,
        actions=actions,
    )