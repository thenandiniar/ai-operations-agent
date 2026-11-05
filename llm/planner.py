
from pydantic import BaseModel

from core.planner import (
    AnalysisPlan,
    PlanAction,
    PlannerError,
)
from llm.gemini_client import (
    generate_structured_response,
    GeminiRequestError,
)
from tools.registry import get_tool_descriptions


class GeminiPlanAction(BaseModel):
    tool: str
    reason: str
    column: str | None = None


class GeminiPlanResponse(BaseModel):
    goal: str
    actions: list[GeminiPlanAction]


class GeminiPlannerError(Exception):
    """Raised when Gemini cannot create a valid execution plan."""


def plan_with_gemini(
    user_goal: str,
    available_columns: list[str],
) -> AnalysisPlan:
    """Generate and validate an execution plan using Gemini."""

    if not user_goal or not user_goal.strip():
        raise PlannerError("Please provide an analysis goal.")

    if not available_columns:
        raise PlannerError("No dataset columns are available.")

    descriptions = get_tool_descriptions()

    tool_information = "\n".join(
        f"- {name}: {description}"
        for name, description in descriptions.items()
    )

    prompt = f"""
You are the planning engine of an AI Operations Agent.

Convert the user's business objective into a structured execution plan.

USER OBJECTIVE:
{user_goal.strip()}

AVAILABLE DATASET COLUMNS:
{available_columns}

AVAILABLE TOOLS:
{tool_information}

RULES:
1. Only select tools from the supplied tool list.
2. Never invent tool names.
3. Use only columns that exist in the supplied dataset.
4. For detect_iqr_anomalies, provide the column field.
5. For every other tool, leave column as null.
6. Choose only tools relevant to the objective.
7. Avoid duplicate actions.
8. Keep each reason short and specific.
9. Do not generate Python code or SQL.
10. Do not claim that any tool has already executed.

Return the requested structured response.
"""

    try:
        response = generate_structured_response(
            prompt,
            GeminiPlanResponse,
        )

        if not response.goal.strip():
            raise GeminiPlannerError(
                "Gemini returned an empty goal."
            )

        if not response.actions:
            raise GeminiPlannerError(
                "Gemini did not select any tools."
            )

        validated_actions = []
        registered_tools = set(descriptions.keys())
        seen_tools = set()

        for action in response.actions:

            if action.tool not in registered_tools:
                raise GeminiPlannerError(
                    f"Gemini selected an unknown tool: {action.tool}"
                )

            if action.tool in seen_tools:
                raise GeminiPlannerError(
                    f"Gemini selected a duplicate tool: {action.tool}"
                )

            seen_tools.add(action.tool)

            parameters = {}

            if action.tool == "detect_iqr_anomalies":

                column = action.column

                if not column or column not in available_columns:
                    raise GeminiPlannerError(
                        f"Invalid anomaly column: {column}"
                    )

                parameters = {"column": column}

            elif action.column is not None:
                raise GeminiPlannerError(
                    f"Unexpected column for tool: {action.tool}"
                )

            try:
                validated_actions.append(
                    PlanAction(
                        tool=action.tool,
                        parameters=parameters,
                        reason=action.reason,
                    )
                )

            except Exception as error:
                raise GeminiPlannerError(
                    f"Invalid action structure: {error}"
                ) from error

        return AnalysisPlan(
            goal=user_goal.strip(),
            actions=validated_actions,
            planner_type="gemini",
        )

    except GeminiPlannerError:
        raise

    except GeminiRequestError as error:
        raise GeminiPlannerError(str(error)) from error
