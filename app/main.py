
import json
import sys
from pathlib import Path

# =========================================================
# PROJECT CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components

from tools.csv_reader import load_csv
from core.planner import plan_analysis
from core.executor import execute_plan
from core.state import (
    initialize_state,
    set_dataset,
    clear_analysis,
)
from llm.planner import (
    plan_with_gemini,
    GeminiPlannerError,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Nexus AI | Operations Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================

initialize_state(st.session_state)

defaults = {
    "planner_source": None,
    "pending_goal": None,
    "approved_plan": False,
    "memory_history": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CUSTOM HYBRID + TEAL DESIGN
# =========================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --teal: #0D9488;
        --teal-dark: #0F766E;
        --teal-light: #CCFBF1;
        --navy: #101827;
        --navy-soft: #182438;
        --muted: #64748B;
        --surface: #FFFFFF;
        --background: #F4F7FA;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: var(--background);
    }

    /* Dark sidebar */
    section[data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #26354B;
    }

    section[data-testid="stSidebar"] * {
        color: #E2E8F0;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #CBD5E1;
    }

    section[data-testid="stSidebar"] .stButton button {
        background: var(--teal);
        color: white;
        border: none;
        border-radius: 9px;
    }

    section[data-testid="stSidebar"] .stButton button:hover {
        background: var(--teal-dark);
        color: white;
    }

    /* Main workspace */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        color: #152238;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    p, label {
        color: #475569;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(
            120deg,
            #102033 0%,
            #183C4A 55%,
            #0F766E 100%
        );
        padding: 30px 32px;
        border-radius: 18px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 12px 30px rgba(15, 40, 55, 0.12);
    }

    .hero h1 {
        color: white !important;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #D1E5E8 !important;
        margin-bottom: 0;
        font-size: 15px;
    }

    .hero-tag {
        color: #99F6E4;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    /* Cards */
    div[data-testid="stMetric"] {
        background: white;
        padding: 17px 18px;
        border: 1px solid #E2E8F0;
        border-radius: 13px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #102033;
        font-weight: 800;
    }

    /* Buttons */
    .stButton button,
    .stDownloadButton button {
        border-radius: 9px;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton button[kind="primary"] {
        background: var(--teal);
        border: 1px solid var(--teal);
    }

    .stButton button[kind="primary"]:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
    }

    /* Inputs */
    .stTextArea textarea,
    .stTextInput input {
        border-radius: 10px;
        border-color: #CBD5E1;
        background: white;
    }

    /* Expanders */
    details[data-testid="stExpander"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 11px;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Section label */
    .section-label {
        color: #0F766E;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-bottom: 5px;
    }

    .subtle {
        color: #64748B;
        font-size: 13px;
    }

    .finding-title {
        color: #0F766E;
        font-weight: 800;
        font-size: 16px;
    }

    .sidebar-brand {
        font-size: 22px;
        font-weight: 800;
        color: white;
        letter-spacing: -0.5px;
    }

    .sidebar-subtitle {
        font-size: 10px;
        color: #94A3B8;
        letter-spacing: 1.6px;
        font-weight: 700;
    }

    .status-dot {
        color: #5EEAD4;
        font-weight: 800;
    }

    hr {
        border-color: #E2E8F0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def json_safe(value):
    """Convert Python values into JSON-compatible values."""
    return json.loads(json.dumps(value, default=str))


def format_label(value):
    """Convert snake_case labels into readable titles."""
    return str(value).replace("_", " ").title()


def format_currency(value):
    """Format a numerical value as INR."""
    return f"₹{value:,.2f}"


def display_evidence(evidence):
    """Display supporting evidence from business findings."""

    if not isinstance(evidence, dict) or not evidence:
        st.caption("No structured evidence was provided.")
        return

    columns = st.columns(min(len(evidence), 3))

    for index, (key, value) in enumerate(evidence.items()):
        label = format_label(key)

        with columns[index % len(columns)]:
            if isinstance(value, (int, float)):
                if "percent" in key.lower():
                    formatted = f"{value:,.2f}%"
                elif "revenue" in key.lower():
                    formatted = format_currency(value)
                else:
                    formatted = f"{value:,.2f}"

                st.metric(label, formatted)

            elif isinstance(value, list):
                st.markdown(f"**{label}**")
                st.write(", ".join(map(str, value)))

            else:
                st.markdown(f"**{label}**")
                st.write(value)


def display_business_insights(result):
    """Render business intelligence findings."""

    st.markdown(
        '<div class="section-label">BUSINESS INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Business insights")

    metrics = result.get("key_metrics", {})

    if isinstance(metrics, dict):
        metric_items = [
            (key, value)
            for key, value in metrics.items()
            if not isinstance(value, (dict, list))
        ]

        if metric_items:
            columns = st.columns(min(len(metric_items), 4))

            for index, (name, value) in enumerate(metric_items):
                with columns[index % len(columns)]:
                    label = format_label(name)

                    if isinstance(value, (int, float)):
                        if "revenue" in name.lower():
                            formatted = format_currency(value)
                        else:
                            formatted = f"{value:,.2f}"
                    else:
                        formatted = str(value)

                    st.metric(label, formatted)

    # Category revenue table
    category_revenue = metrics.get("category_revenue")

    if isinstance(category_revenue, dict) and category_revenue:
        st.markdown("#### Revenue by category")

        category_rows = [
            {
                "Category": category,
                "Revenue (₹)": revenue,
                "Share (%)": (
                    revenue / sum(category_revenue.values()) * 100
                    if sum(category_revenue.values())
                    else 0
                ),
            }
            for category, revenue in category_revenue.items()
        ]

        st.dataframe(
            category_rows,
            width="stretch",
            hide_index=True,
        )

    # Executive summary
    summary = result.get("summary")

    if summary:
        st.markdown("#### Executive summary")
        st.info(summary)

    # Individual findings
    findings = result.get("findings", [])

    if findings:
        st.markdown("#### Key business findings")

        for index, finding in enumerate(findings, start=1):

            if not isinstance(finding, dict):
                st.write(finding)
                continue

            with st.container(border=True):

                st.markdown(
                    f'<div class="finding-title">'
                    f'{index:02d} · {finding.get("title", "Finding")}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                observation = finding.get("observation")

                if observation:
                    st.write(observation)

                evidence = finding.get("evidence")

                with st.expander("📊 Supporting evidence", expanded=True):
                    display_evidence(evidence)

                recommendation = finding.get("recommendation")

                if recommendation:
                    st.markdown("**Recommended action**")
                    st.success(recommendation)

    # Correlation results
    correlation_pairs = result.get("correlation_pairs", [])

    if correlation_pairs:
        st.markdown("#### Correlation analysis")
        st.dataframe(
            correlation_pairs,
            width="stretch",
            hide_index=True,
        )

    # Anomaly count
    anomaly_count = result.get("anomaly_count")

    if anomaly_count is not None:
        st.metric("Detected revenue anomalies", anomaly_count)


def display_chart(chart_value, result):
    """Embed a generated HTML chart inside the dashboard."""

    chart_path = Path(chart_value)

    if not chart_path.is_absolute():
        chart_path = PROJECT_ROOT / chart_path

    chart_path = chart_path.resolve()
    reports_dir = (PROJECT_ROOT / "reports").resolve()

    if (
        not chart_path.is_file()
        or reports_dir not in chart_path.parents
    ):
        st.warning("The generated chart file could not be found.")
        return

    try:
        html_content = chart_path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        components.html(
            html_content,
            height=560,
            scrolling=True,
        )

        st.download_button(
            "⬇ Download interactive chart",
            data=html_content,
            file_name=chart_path.name,
            mime="text/html",
            key=f"download_{result.tool}_{chart_path.name}",
        )

    except Exception as error:
        st.error(f"Unable to display chart: {error}")


def display_execution_result(result):
    """Render the output of an individual tool."""

    if result.error:
        st.error(result.error)
        return

    value = result.result

    if value is None:
        st.info("This tool did not return a result.")
        return

    if (
        result.tool == "generate_business_insights"
        and isinstance(value, dict)
    ):
        display_business_insights(value)

    elif (
        isinstance(value, str)
        and value.lower().endswith(".html")
    ):
        display_chart(value, result)

    else:
        st.json(json_safe(value))


def load_memory_history():
    """Load persistent execution history."""

    from memory.store import load_history

    return load_history(
        PROJECT_ROOT / "memory" / "execution_history.jsonl"
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">◈ NEXUS AI</div>
        <div class="sidebar-subtitle">OPERATIONS INTELLIGENCE</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### Workspace")
    st.markdown(
        '<span class="status-dot">●</span> Application ready',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### Dataset")

    uploaded_file = st.file_uploader(
        "Upload business CSV",
        type=["csv"],
        help="Upload a CSV file for analysis.",
    )

    if uploaded_file is not None:

        try:
            loaded = load_csv(uploaded_file)

            current_name = st.session_state["file_name"]

            if current_name != uploaded_file.name:

                set_dataset(
                    st.session_state,
                    loaded.dataframe,
                    uploaded_file.name,
                )

                clear_analysis(st.session_state)

                st.session_state["planner_source"] = None
                st.session_state["pending_goal"] = None
                st.session_state["approved_plan"] = False

            st.success(f"Loaded {uploaded_file.name}")

        except Exception as error:
            st.error(f"CSV loading failed: {error}")

    if st.button(
        "Load sample sales dataset",
        width="stretch",
    ):

        try:
            loaded = load_csv(
                str(PROJECT_ROOT / "data" / "sample_sales.csv")
            )

            set_dataset(
                st.session_state,
                loaded.dataframe,
                "sample_sales.csv",
            )

            clear_analysis(st.session_state)

            st.session_state["planner_source"] = None
            st.session_state["pending_goal"] = None
            st.session_state["approved_plan"] = False

            st.success("Sample dataset loaded!")

        except Exception as error:
            st.error(f"Sample loading failed: {error}")

    st.markdown("---")

    st.markdown("### Agent capabilities")

    st.markdown(
        """
        - 🧠 Gemini AI planning
        - 📊 Statistical analysis
        - 🔎 Anomaly detection
        - 📈 Correlation analysis
        - 💡 Business insights
        - 🗂 Persistent memory
        """
    )

    st.markdown("---")

    st.caption("AI Operations Studio")
    st.caption("Python · Gemini · Pandas · Plotly")


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">AI-POWERED BUSINESS INTELLIGENCE</div>
        <h1>Business intelligence,<br>powered by AI.</h1>
        <p>
            Transform raw business data into meaningful insights,
            actionable strategies and better decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

dataframe = st.session_state["dataframe"]

st.markdown(
    '<div class="section-label">01 / DATA WORKSPACE</div>',
    unsafe_allow_html=True,
)

st.header("Dataset overview")

if dataframe is not None:

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Total records", f"{len(dataframe):,}")
    metric2.metric("Columns", len(dataframe.columns))
    metric3.metric(
        "Missing values",
        int(dataframe.isna().sum().sum()),
    )
    metric4.metric(
        "Duplicate rows",
        int(dataframe.duplicated().sum()),
    )

    st.caption(
        f"Current dataset: {st.session_state['file_name']}"
    )

    with st.expander("Preview dataset", expanded=True):
        st.dataframe(
            dataframe,
            width="stretch",
            hide_index=True,
        )

    with st.expander("Column information"):
        st.dataframe(
            {
                "Column": dataframe.columns,
                "Data type": dataframe.dtypes.astype(str).values,
                "Missing values": dataframe.isna().sum().values,
                "Unique values": dataframe.nunique().values,
            },
            width="stretch",
            hide_index=True,
        )

else:

    st.info(
        "Upload a CSV file from the sidebar or load the sample "
        "dataset to begin."
    )


# =========================================================
# BUSINESS OBJECTIVE
# =========================================================

if dataframe is not None:

    st.divider()

    st.markdown(
        '<div class="section-label">02 / DEFINE YOUR GOAL</div>',
        unsafe_allow_html=True,
    )

    st.header("What should your agent investigate?")

    st.caption(
        "Describe your business problem in natural language. "
        "The AI planner will translate it into analytical tasks."
    )

    goal = st.text_area(
        "Business objective",
        placeholder=(
            "Example: Analyze sales performance, identify unusual "
            "transactions, compare categories and recommend ways "
            "to improve revenue."
        ),
        height=120,
        key="business_goal",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        generate_plan = st.button(
            "🧠 Generate plan",
            type="primary",
            width="stretch",
        )

    with col2:
        st.caption(
            "Your plan will be reviewed before any analysis is executed."
        )

    if generate_plan:

        if not goal.strip():

            st.warning("Please enter a business objective.")

        else:

            clear_analysis(st.session_state)

            st.session_state["planner_source"] = None
            st.session_state["pending_goal"] = None
            st.session_state["approved_plan"] = False

            try:

                with st.spinner(
                    "Your AI agent is creating an execution plan..."
                ):

                    try:

                        plan = plan_with_gemini(
                            goal,
                            dataframe.columns.tolist(),
                        )

                        source = "Gemini AI"

                    except GeminiPlannerError as gemini_error:

                        st.warning(
                            "Gemini is temporarily unavailable. "
                            "Switching to the rule-based planner. "
                            f"Details: {gemini_error}"
                        )

                        plan = plan_analysis(
                            goal,
                            dataframe.columns.tolist(),
                        )

                        source = "Rule-based fallback"

                st.session_state["plan"] = plan
                st.session_state["planner_source"] = source
                st.session_state["pending_goal"] = goal

                st.success(
                    f"Execution plan generated using {source}."
                )

            except Exception as error:
                st.session_state["error"] = str(error)


# =========================================================
# EXECUTION PLAN
# =========================================================

if dataframe is not None:

    plan = st.session_state["plan"]

    if plan is not None:

        st.divider()

        st.markdown(
            '<div class="section-label">03 / PLAN REVIEW</div>',
            unsafe_allow_html=True,
        )

        st.header("Your execution plan")

        st.write(f"**Business objective:** {plan.goal}")

        source = st.session_state["planner_source"]

        if source == "Gemini AI":
            st.success("🧠 Planner: Gemini AI")
        else:
            st.info("⚙️ Planner: Rule-based fallback")

        st.metric("Planned analytical actions", len(plan.actions))

        for index, action in enumerate(plan.actions, start=1):

            with st.container(border=True):

                col1, col2 = st.columns([0.12, 0.88])

                with col1:
                    st.markdown(f"### {index:02d}")

                with col2:
                    st.markdown(f"**{action.tool}**")
                    st.caption(action.reason)

                    if action.parameters:
                        st.json(action.parameters)

        st.warning(
            "Review the proposed actions. Execution begins only "
            "after you approve this plan."
        )

        if st.button(
            "🚀 Approve and execute",
            type="primary",
            width="stretch",
        ):

            try:

                with st.spinner(
                    "Executing your approved analytical workflow..."
                ):

                    report = execute_plan(
                        plan,
                        dataframe,
                        output_dir=str(PROJECT_ROOT / "reports"),
                    )

                    st.session_state["report"] = report
                    st.session_state["approved_plan"] = True

            except Exception as error:
                st.session_state["error"] = str(error)


# =========================================================
# EXECUTION RESULTS
# =========================================================

report = st.session_state["report"]

if report is not None:

    st.divider()

    st.markdown(
        '<div class="section-label">04 / ANALYSIS RESULTS</div>',
        unsafe_allow_html=True,
    )

    st.header("Analysis dashboard")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Successful actions",
        report.successful_actions,
    )

    metric2.metric(
        "Failed actions",
        report.failed_actions,
    )

    metric3.metric(
        "Total actions",
        report.total_actions,
    )

    if report.status == "success":
        st.success("All planned actions completed successfully.")

    elif report.status == "partial":
        st.warning("Analysis completed with some failures.")

    else:
        st.error("Analysis execution failed.")

    memory_status = (
        "Saved" if report.memory_saved else "Not saved"
    )

    st.caption(f"Persistent execution memory: {memory_status}")

    if report.memory_error:
        st.warning(report.memory_error)

    # Separate insight and chart sections
    insight_results = [
        result
        for result in report.results
        if result.tool == "generate_business_insights"
        and result.status == "success"
    ]

    chart_results = [
        result
        for result in report.results
        if (
            isinstance(result.result, str)
            and result.result.lower().endswith(".html")
        )
        and result.status == "success"
    ]

    other_results = [
        result
        for result in report.results
        if result not in insight_results
        and result not in chart_results
    ]

    tab_insights, tab_charts, tab_tools = st.tabs(
        [
            "💡 Business insights",
            "📊 Interactive charts",
            "⚙️ Tool execution details",
        ]
    )

    with tab_insights:

        if insight_results:

            for result in insight_results:

                if result.error:
                    st.error(result.error)
                else:
                    display_business_insights(result.result)

        else:
            st.info("No business insights were generated.")

    with tab_charts:

        if chart_results:

            for result in chart_results:

                st.markdown(f"#### {format_label(result.tool)}")
                display_chart(result.result, result)

        else:
            st.info("No interactive charts were generated.")

    with tab_tools:

        for result in other_results:

            with st.expander(
                f"{result.tool} · {result.status.upper()}",
                expanded=False,
            ):

                st.write(result.reason)
                display_execution_result(result)

    st.download_button(
        "⬇ Download complete execution report",
        data=report.model_dump_json(indent=2),
        file_name="execution_report.json",
        mime="application/json",
    )


# =========================================================
# ERROR DISPLAY
# =========================================================

if st.session_state["error"]:
    st.error(st.session_state["error"])


# =========================================================
# AGENT MEMORY
# =========================================================

st.divider()

st.markdown(
    '<div class="section-label">05 / PERSISTENT MEMORY</div>',
    unsafe_allow_html=True,
)

st.header("Agent execution history")

if st.button("🔄 Refresh execution history"):

    try:
        st.session_state["memory_history"] = load_memory_history()

    except Exception as error:
        st.error(f"Unable to load memory: {error}")


history = st.session_state["memory_history"]

if history is None:

    st.caption(
        "Refresh history to view previously saved agent executions."
    )

else:

    st.metric("Total saved executions", len(history))

    for index, item in enumerate(reversed(history[-10:])):

        with st.expander(
            item.get("goal", "Previous execution"),
        ):

            st.caption(
                f"Timestamp: {item.get('timestamp', 'Unavailable')}"
            )

            st.json(
                json_safe(item.get("execution", {}))
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:18px 0;
        color:#64748B;
        font-size:12px;
    ">
        <strong style="color:#0F766E;">NEXUS AI</strong>
        · AI Operations Studio
        <br>
        Built with Python · Gemini · Pandas · Plotly · Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
