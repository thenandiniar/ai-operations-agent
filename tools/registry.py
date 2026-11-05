from pathlib import Path

import pandas as pd

from tools.profiler import profile_dataframe
from tools.statistics import (
    analyze_sales,
    calculate_correlations,
    detect_iqr_anomalies,
)
from tools.insights import generate_business_insights
from tools.visualizer import (
    create_category_revenue_chart,
    create_revenue_distribution_chart,
    create_correlation_heatmap,
    save_chart,
)


def create_tool_registry(output_dir: str = "reports") -> dict:
    """
    Create the registry of tools available to the AI agent.

    Every registered tool accepts a DataFrame as its first argument.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def category_chart(dataframe: pd.DataFrame, **kwargs):
        figure = create_category_revenue_chart(dataframe, **kwargs)
        return save_chart(
            figure,
            str(output_path / "category_revenue.html"),
        )

    def distribution_chart(dataframe: pd.DataFrame, **kwargs):
        figure = create_revenue_distribution_chart(dataframe, **kwargs)
        return save_chart(
            figure,
            str(output_path / "revenue_distribution.html"),
        )

    def correlation_heatmap(dataframe: pd.DataFrame, **kwargs):
        figure = create_correlation_heatmap(dataframe, **kwargs)
        return save_chart(
            figure,
            str(output_path / "correlation_heatmap.html"),
        )

    return {
        "profile_dataframe": profile_dataframe,
        "analyze_sales": analyze_sales,
        "calculate_correlations": calculate_correlations,
        "detect_iqr_anomalies": detect_iqr_anomalies,
        "generate_business_insights": generate_business_insights,
        "create_category_revenue_chart": category_chart,
        "create_revenue_distribution_chart": distribution_chart,
        "create_correlation_heatmap": correlation_heatmap,
    }


TOOL_DESCRIPTIONS = {
    "profile_dataframe": (
        "Inspect dataset structure, column types, missing values, "
        "duplicates and numerical statistics."
    ),
    "analyze_sales": (
        "Calculate sales metrics, total revenue, average revenue, "
        "quantity sold and category revenue."
    ),
    "calculate_correlations": (
        "Calculate correlations between numerical columns."
    ),
    "detect_iqr_anomalies": (
        "Find unusually high or low numerical values using the IQR method."
    ),
    "generate_business_insights": (
        "Generate evidence-based business findings, key metrics, "
        "category performance, revenue concentration, anomalies, "
        "correlation observations and actionable recommendations."
    ),
    "create_category_revenue_chart": (
        "Create an interactive bar chart comparing revenue by category."
    ),
    "create_revenue_distribution_chart": (
        "Create a histogram showing the distribution of revenue."
    ),
    "create_correlation_heatmap": (
        "Create a heatmap showing correlations between numeric columns."
    ),
}


def get_tool_descriptions() -> dict:
    """Return descriptions of all available tools."""
    return TOOL_DESCRIPTIONS.copy()