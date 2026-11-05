
import math

import pandas as pd

from tools.statistics import analyze_sales


class InsightsError(Exception):
    """Raised when business insights cannot be generated."""


def generate_business_insights(
    dataframe: pd.DataFrame,
    revenue_column: str = "Revenue",
    quantity_column: str = "Quantity",
    category_column: str = "Category",
) -> dict:
    """
    Generate evidence-based business findings from a sales dataset.

    This module interprets calculated metrics. It does not use an LLM
    and does not make unsupported causal claims.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise InsightsError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise InsightsError("Cannot analyze an empty dataset.")

    analysis = analyze_sales(
        dataframe,
        revenue_column=revenue_column,
        quantity_column=quantity_column,
        category_column=category_column,
    )

    total_revenue = analysis["total_revenue"]
    category_revenue = analysis["category_revenue"]
    anomalies = analysis["revenue_anomalies"]
    correlations = analysis["correlations"]

    findings = []

    # -----------------------------------------------------
    # 1. CATEGORY PERFORMANCE
    # -----------------------------------------------------

    if category_revenue:

        top_category = max(
            category_revenue,
            key=category_revenue.get,
        )

        top_revenue = category_revenue[top_category]

        top_share = (
            (top_revenue / total_revenue) * 100
            if total_revenue != 0
            else None
        )

        findings.append({
            "type": "category_performance",
            "title": "Highest revenue category",
            "observation": (
                f"{top_category} generated the highest category revenue."
            ),
            "evidence": {
                "category": top_category,
                "revenue": top_revenue,
                "revenue_share_percent": (
                    round(top_share, 2)
                    if top_share is not None
                    else None
                ),
            },
            "recommendation": (
                "Review this category's products and sales performance "
                "to understand what contributes to its revenue."
            ),
        })

        # Revenue concentration
        if top_share is not None and top_share >= 70:

            findings.append({
                "type": "revenue_concentration",
                "title": "Revenue concentration",
                "observation": (
                    f"{top_category} contributes "
                    f"{top_share:.2f}% of total revenue."
                ),
                "evidence": {
                    "category": top_category,
                    "share_percent": round(top_share, 2),
                },
                "recommendation": (
                    "Assess the business risk of relying heavily on one "
                    "category and investigate diversification opportunities."
                ),
            })

    # -----------------------------------------------------
    # 2. ANOMALY ANALYSIS
    # -----------------------------------------------------

    anomaly_count = anomalies["anomaly_count"]

    if anomaly_count > 0:

        anomaly_values = [
            record.get(revenue_column)
            for record in anomalies["anomalies"]
        ]

        findings.append({
            "type": "anomaly",
            "title": "Unusual revenue values detected",
            "observation": (
                f"The IQR method identified {anomaly_count} unusual "
                f"revenue record(s)."
            ),
            "evidence": {
                "anomaly_count": anomaly_count,
                "upper_bound": anomalies["upper_bound"],
                "lower_bound": anomalies["lower_bound"],
                "values": anomaly_values,
            },
            "recommendation": (
                "Investigate these transactions to determine whether "
                "they represent valid exceptional sales or data errors."
            ),
        })

    else:

        findings.append({
            "type": "anomaly",
            "title": "No revenue anomalies detected",
            "observation": (
                "The IQR method did not identify unusual revenue records."
            ),
            "evidence": {
                "anomaly_count": 0,
                "upper_bound": anomalies["upper_bound"],
                "lower_bound": anomalies["lower_bound"],
            },
            "recommendation": (
                "Continue monitoring revenue values as new transactions "
                "are recorded."
            ),
        })

    # -----------------------------------------------------
    # 3. CORRELATION ANALYSIS
    # -----------------------------------------------------

    correlation_pairs = []

    for first_column, related_columns in correlations.items():

        for second_column, coefficient in related_columns.items():

            if first_column >= second_column:
                continue

            if coefficient is None or not math.isfinite(coefficient):
                continue

            correlation_pairs.append({
                "column_1": first_column,
                "column_2": second_column,
                "correlation": round(float(coefficient), 3),
            })

    correlation_pairs.sort(
        key=lambda item: abs(item["correlation"]),
        reverse=True,
    )

    if correlation_pairs:

        strongest = correlation_pairs[0]
        coefficient = strongest["correlation"]

        if abs(coefficient) >= 0.7:
            strength = "strong"
        elif abs(coefficient) >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        direction = (
            "positive" if coefficient > 0
            else "negative" if coefficient < 0
            else "no linear"
        )

        findings.append({
            "type": "correlation",
            "title": "Strongest numerical relationship",
            "observation": (
                f"{strongest['column_1']} and "
                f"{strongest['column_2']} have a {strength} "
                f"{direction} correlation."
            ),
            "evidence": strongest,
            "recommendation": (
                "Investigate this relationship further. Correlation "
                "does not establish causation."
            ),
        })

    else:

        findings.append({
            "type": "correlation",
            "title": "Insufficient correlation data",
            "observation": (
                "There are not enough valid numerical relationships "
                "to identify a strongest correlation."
            ),
            "evidence": {
                "available_pairs": 0,
            },
            "recommendation": (
                "Collect additional numerical variables or records "
                "to support relationship analysis."
            ),
        })

    # -----------------------------------------------------
    # 4. EXECUTIVE SUMMARY
    # -----------------------------------------------------

    summary = (
        f"The dataset contains {len(dataframe)} records. "
        f"Total revenue is {total_revenue:,.2f}, "
        f"average revenue is {analysis['average_revenue']:,.2f}, "
        f"and total quantity sold is "
        f"{analysis['total_quantity']:,.2f}."
    )

    return {
        "summary": summary,
        "key_metrics": {
            "total_revenue": total_revenue,
            "average_revenue": analysis["average_revenue"],
            "median_revenue": analysis["median_revenue"],
            "total_quantity": analysis["total_quantity"],
            "category_revenue": category_revenue,
        },
        "findings": findings,
        "correlation_pairs": correlation_pairs,
        "anomaly_count": anomaly_count,
    }
