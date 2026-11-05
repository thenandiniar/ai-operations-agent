
def initialize_state(session_state):
    """Initialize Streamlit session state."""

    defaults = {
        "dataframe": None,
        "file_name": None,
        "plan": None,
        "report": None,
        "error": None,
    }

    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


def set_dataset(session_state, dataframe, file_name):
    """Store the currently selected dataset."""

    session_state["dataframe"] = dataframe
    session_state["file_name"] = file_name
    session_state["plan"] = None
    session_state["report"] = None
    session_state["error"] = None


def clear_analysis(session_state):
    """Clear previous analysis results."""

    session_state["plan"] = None
    session_state["report"] = None
    session_state["error"] = None
