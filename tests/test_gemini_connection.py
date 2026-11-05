import os

import pytest

from llm.gemini_client import (
    GeminiConfigurationError,
    GeminiRequestError,
    generate_structured_response,
)
from llm.planner import GeminiPlanResponse


@pytest.mark.skipif(
    os.getenv("RUN_GEMINI_LIVE_TESTS") != "1",
    reason="Live Gemini tests are opt-in.",
)
def test_live_gemini_connection():
    try:
        response = generate_structured_response(
            "Return a plan with goal 'connection test' and no actions.",
            GeminiPlanResponse,
        )

        assert isinstance(response, GeminiPlanResponse)
        assert response.goal

    except GeminiConfigurationError:
        pytest.fail("Gemini API key is not configured.")

    except GeminiRequestError as error:
        pytest.fail(f"Gemini API request failed: {error}")