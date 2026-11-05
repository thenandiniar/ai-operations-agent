import pandas as pd


def profile_dataframe(dataframe: pd.DataFrame) -> dict:
    """
    Analyze a DataFrame and return a structured data profile.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty and len(dataframe.columns) == 0:
        raise ValueError("Cannot profile a dataset without columns.")

    numeric_columns = dataframe.select_dtypes(include="number").columns
    categorical_columns = dataframe.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns

    column_profiles = {}

    for column in dataframe.columns:
        series = dataframe[column]

        column_info = {
            "data_type": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round(
                float(series.isna().mean() * 100), 2
            ),
            "unique_values": int(series.nunique()),
        }

        if column in numeric_columns:
            column_info["statistics"] = {
                "mean": float(series.mean()) if series.notna().any() else None,
                "median": float(series.median()) if series.notna().any() else None,
                "minimum": float(series.min()) if series.notna().any() else None,
                "maximum": float(series.max()) if series.notna().any() else None,
                "standard_deviation": (
                    float(series.std()) if series.notna().any() else None
                ),
            }

        elif column in categorical_columns:
            column_info["top_values"] = {
                str(key): int(value)
                for key, value in series.value_counts().head(5).items()
            }

        column_profiles[str(column)] = column_info

    return {
        "overview": {
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "total_missing_cells": int(dataframe.isna().sum().sum()),
            "memory_usage_bytes": int(
                dataframe.memory_usage(deep=True).sum()
            ),
        },
        "columns": column_profiles,
    }