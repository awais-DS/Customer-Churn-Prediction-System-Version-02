import streamlit as st
from api_client import check_health

# 1. Configure page metadata (must be the first Streamlit call)
st.set_page_config(
    page_title="Customer Churn Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Initialize shared session storage (replaces database for session-wide analytics)
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []

# 3. Sidebar: Live API Health Check & Session Counter
with st.sidebar:
    st.title(" Churn Intelligence")
    st.caption("End-to-End ML Retention System")
    st.divider()

    health = check_health()
    if health.get("status") == "healthy":
        st.success(f" API Online ({health.get('model_version', 'v1.0.0')})")
    else:
        st.error(" API Offline — Start FastAPI first")

    st.metric("Session Scored Customers", len(st.session_state["predictions"]))
    st.divider()

# 4. Multi-page Navigation Setup
pages = {
    "Overview": [
        st.Page("pages/1_executive_dashboard.py", title="Executive Dashboard", default=True),
    ],
    "Predictions": [
        st.Page("pages/2_single_prediction.py", title="Single Customer Prediction"),
        st.Page("pages/3_batch_prediction.py", title="Batch Customer Prediction"),
    ],
    "Analytics & Governance": [
        st.Page("pages/4_high_risk_explorer.py", title="High-Risk Explorer"),
        st.Page("pages/5_model_insights.py", title="Model Performance & Explainability"),
        st.Page("pages/6_prediction_history.py", title="Prediction History"),
    ],
}

pg = st.navigation(pages)
pg.run()
