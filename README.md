# AI Operations Agent

An AI-powered business data analysis application built with Python, Streamlit and Google Gemini.

## Features
- CSV dataset upload
- Automatic data profiling
- Gemini-powered natural-language planning
- Rule-based planning fallback
- Statistical and correlation analysis
- IQR anomaly detection
- Business insights and recommendations
- Interactive visualizations
- Execution history

## Tech Stack
Python, Streamlit, Google Gemini, Pandas, NumPy, Plotly, Pydantic and Pytest.

## Installation

Install dependencies:

    pip install -r requirements.txt

Create a .env file:

    GEMINI_API_KEY=your_api_key
    GEMINI_MODEL=gemini-2.5-flash

Run the application:

    streamlit run app/main.py

## Run Tests

    python -m pytest -q

## Security
Never commit API keys or .env files.
