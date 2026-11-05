from tools.registry import (
    create_tool_registry,
    get_tool_descriptions,
)


def test_registry_contains_all_tools(tmp_path):
    registry = create_tool_registry(str(tmp_path))

    expected_tools = {
        "profile_dataframe",
        "analyze_sales",
        "calculate_correlations",
        "detect_iqr_anomalies",
        "generate_business_insights",
        "create_category_revenue_chart",
        "create_revenue_distribution_chart",
        "create_correlation_heatmap",
    }

    assert set(registry.keys()) == expected_tools


def test_all_registered_tools_are_callable(tmp_path):
    registry = create_tool_registry(str(tmp_path))

    for tool in registry.values():
        assert callable(tool)


def test_tool_descriptions_exist():
    descriptions = get_tool_descriptions()

    assert "generate_business_insights" in descriptions

    for description in descriptions.values():
        assert isinstance(description, str)
        assert description.strip()


def test_registered_sales_tool_executes(tmp_path):
    import pandas as pd

    registry = create_tool_registry(str(tmp_path))

    dataframe = pd.DataFrame({
        "OrderID": [1, 2],
        "Product": ["Laptop", "Mouse"],
        "Category": ["Electronics", "Accessories"],
        "Revenue": [50000, 1000],
        "Quantity": [2, 5],
    })

    result = registry["analyze_sales"](dataframe)

    assert result["total_revenue"] == 51000