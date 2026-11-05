import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError


class GeminiConfigurationError(Exception):
    """Raised when Gemini configuration is missing or invalid."""


class GeminiRequestError(Exception):
    """Raised when a Gemini request fails."""


load_dotenv()


def get_setting(name: str, default=None):
    """Read configuration from environment or Streamlit secrets."""

    value = os.getenv(name)

    if value:
        return value

    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def create_gemini_client():
    """Create an authenticated Gemini Developer API client."""

    api_key = get_setting("GEMINI_API_KEY")

    if not api_key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is missing. Configure it in .env "
            "or Streamlit Cloud Secrets."
        )

    return genai.Client(api_key=api_key)


def generate_structured_response(
    prompt: str,
    response_schema: type[BaseModel],
) -> BaseModel:
    """Generate and validate a structured Pydantic response."""

    model = get_setting("GEMINI_MODEL", "gemini-2.5-flash")
    client = create_gemini_client()

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.1,
            ),
        )

        if not response.text:
            raise GeminiRequestError(
                "Gemini returned an empty response."
            )

        try:
            return response_schema.model_validate_json(response.text)
        except ValidationError as error:
            raise GeminiRequestError(
                f"Gemini returned invalid structured data: {error}"
            ) from error

    except GeminiRequestError:
        raise

    except Exception as error:
        raise GeminiRequestError(
            f"Gemini request failed: {type(error).__name__}: {error}"
        ) from error

    finally:
        client.close()