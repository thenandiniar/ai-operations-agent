from pathlib import Path

import pandas as pd
import plotly.express as px


def create_category_revenue_chart(
    dataframe: pd.DataFrame,
    category_column: str = "Category",
    revenue_column: str = "Revenue",
):
    """Create a bar chart of revenue by category."""

    if category_column not in dataframe.columns:
        raise ValueError(f"Column not found: {category_column}")

    if revenue_column not in dataframe.columns:
        raise ValueError(f"Column not found: {revenue_column}")

    data = dataframe[[category_column, revenue_column]].copy()
    data[revenue_column] = pd.to_numeric(
        data[revenue_column], errors="coerce"
    )
    data = data.dropna()

    if data.empty:
        raise ValueError("No valid data available for the chart.")

    summary = (
        data.groupby(category_column, as_index=False)[revenue_column]
        .sum()
        .sort_values(revenue_column, ascending=False)
    )

    return px.bar(
        summary,
        x=category_column,
        y=revenue_column,
        title="Revenue by Category",
        labels={
            category_column: "Category",
            revenue_column: "Revenue",
        },
        text_auto=True,
    )


def create_revenue_distribution_chart(
    dataframe: pd.DataFrame,
    revenue_column: str = "Revenue",
):
    """Create a histogram showing revenue distribution."""

    if revenue_column not in dataframe.columns:
        raise ValueError(f"Column not found: {revenue_column}")

    values = pd.to_numeric(
        dataframe[revenue_column], errors="coerce"
    ).dropna()

    if values.empty:
        raise ValueError("No valid revenue values available.")

    return px.histogram(
        x=values,
        nbins=10,
        title="Revenue Distribution",
        labels={"x": "Revenue", "y": "Frequency"},
    )


def create_correlation_heatmap(dataframe: pd.DataFrame):
    """Create a heatmap of correlations between numeric columns."""

    numeric_data = dataframe.select_dtypes(include="number")

    if numeric_data.shape[1] < 2:
        raise ValueError("At least two numeric columns are required.")

    correlation = numeric_data.corr()

    return px.imshow(
        correlation,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Numerical Correlation Heatmap",
    )


def save_chart(figure, output_path: str) -> str:
    """Save a Plotly chart as a standalone HTML file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure.write_html(str(path), include_plotlyjs=True)

    return str(path.resolve())