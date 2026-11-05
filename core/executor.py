
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from core.planner import AnalysisPlan
from memory.store import save_execution
from tools.registry import create_tool_registry


class ToolExecutionResult(BaseModel):
    tool: str
    status: str
    reason: str
    result: Any = None
    error: str | None = None


class ExecutionReport(BaseModel):
    goal: str
    status: str
    total_actions: int
    successful_actions: int
    failed_actions: int
    results: list[ToolExecutionResult] = Field(default_factory=list)
    memory_saved: bool = False
    memory_error: str | None = None


def execute_plan(
    plan: AnalysisPlan,
    dataframe: pd.DataFrame,
    output_dir: str = "reports",
    continue_on_error: bool = True,
    save_history: bool = True,
) -> ExecutionReport:

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Executor input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Cannot execute a plan on an empty DataFrame.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tool_registry = create_tool_registry(output_dir)
    results = []

    for action in plan.actions:
        tool = tool_registry.get(action.tool)

        if tool is None:
            results.append(
                ToolExecutionResult(
                    tool=action.tool,
                    status="failed",
                    reason=action.reason,
                    error=f"Tool is not registered: {action.tool}",
                )
            )

            if not continue_on_error:
                break

            continue

        try:
            tool_result = tool(
                dataframe,
                **action.parameters,
            )

            results.append(
                ToolExecutionResult(
                    tool=action.tool,
                    status="success",
                    reason=action.reason,
                    result=tool_result,
                )
            )

        except Exception as error:
            results.append(
                ToolExecutionResult(
                    tool=action.tool,
                    status="failed",
                    reason=action.reason,
                    error=f"{type(error).__name__}: {error}",
                )
            )

            if not continue_on_error:
                break

    successful = sum(
        result.status == "success" for result in results
    )

    failed = sum(
        result.status == "failed" for result in results
    )

    if failed == 0:
        status = "success"
    elif successful > 0:
        status = "partial"
    else:
        status = "failed"

    report = ExecutionReport(
        goal=plan.goal,
        status=status,
        total_actions=len(plan.actions),
        successful_actions=successful,
        failed_actions=failed,
        results=results,
    )

    if save_history:
        try:
            save_execution(plan.goal, report)
            report.memory_saved = True
        except Exception as error:
            report.memory_error = str(error)

    return report
