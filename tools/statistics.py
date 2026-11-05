
import re

import pandas as pd


def _is_identifier_column(column_name: str) -> bool:
    """
    Identify numeric columns that represent IDs rather than measurements.

    Examples:
        ID, OrderID, CustomerID, order_id, transaction_id
    """

    name = str(column_name).strip().lower()

    # Normalize names such as Order_ID, Order-ID and Order ID.
    normalized = re.sub(r"[^a-z0-9]", "", name)

    if normalized in {"id", "uuid", "guid"}:
        return True

    # Recognize common identifier naming conventions.
    identifier_prefixes = {
        "order",
        "customer",
        "user",
        "transaction",
        "product",
        "record",
        "invoice",
        "employee",
        "account",
        "payment",
        "session",
        "ticket",
        "item",
        "row",
    }

    if normalized.endswith("id"):
        prefix = normalized[:-2]
        if prefix in identifier_prefixes:
            return True

    # Also recognize names ending in a separate ID token.
    if re.search(r"(?:^|[_\s-])id$", name):
        return True

    return False


def calculate_correlations(dataframe: pd.DataFrame) -> dict:
    """
    Calculate correlations between meaningful numeric columns.

    Identifier columns such as OrderID and CustomerID are excluded
    because their numeric values do not represent measurements.
    """

    numeric_data = dataframe.select_dtypes(include="number")

    measurement_columns = [
        column
        for column in numeric_data.columns
        if not _is_identifier_column(column)
    ]

    numeric_data = numeric_data[measurement_columns]

    if numeric_data.shape[1] < 2:
        return {}

    return numeric_data.corr().round(3).to_dict()


def detect_iqr_anomalies(
    dataframe: pd.DataFrame,
    column: str
) -> dict:
    """
    Detect outliers using the Interquartile Range (IQR) method.
    """

    if column not in dataframe.columns:
        raise ValueError(f"Column not found: {column}")

    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()

    if values.empty:
        raise ValueError(f"Column '{column}' has no valid numeric values.")

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    numeric_column = pd.to_numeric(dataframe[column], errors="coerce")
    mask = (numeric_column < lower_bound) | (numeric_column > upper_bound)

    anomalies = dataframe.loc[mask].copy()

    return {
        "column": column,
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "anomaly_count": int(len(anomalies)),
        "anomalies": anomalies.to_dict(orient="records"),
    }


def analyze_sales(
    dataframe: pd.DataFrame,
    revenue_column: str = "Revenue",
    quantity_column: str = "Quantity",
    category_column: str = "Category",
) -> dict:
    """Generate business metrics and analysis from sales data."""

    required_columns = [
        revenue_column,
        quantity_column,
        category_column,
    ]

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    revenue = pd.to_numeric(
        dataframe[revenue_column], errors="coerce"
    ).dropna()

    quantity = pd.to_numeric(
        dataframe[quantity_column], errors="coerce"
    ).dropna()

    if revenue.empty:
        raise ValueError("No valid revenue values found.")

    if quantity.empty:
        raise ValueError("No valid quantity values found.")

    category_revenue = (
        dataframe.assign(
            _revenue=pd.to_numeric(
                dataframe[revenue_column], errors="coerce"
            )
        )
        .groupby(category_column)["_revenue"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "total_revenue": round(float(revenue.sum()), 2),
        "average_revenue": round(float(revenue.mean()), 2),
        "median_revenue": round(float(revenue.median()), 2),
        "total_quantity": round(float(quantity.sum()), 2),
        "category_revenue": {
            str(key): round(float(value), 2)
            for key, value in category_revenue.items()
        },
        "correlations": calculate_correlations(dataframe),
        "revenue_anomalies": detect_iqr_anomalies(
            dataframe, revenue_column
        ),
    }
