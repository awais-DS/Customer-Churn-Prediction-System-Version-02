import pandas as pd
import streamlit as st

st.title("Prediction History")
st.markdown("Session audit log of all single and batch customer churn predictions.")

predictions = st.session_state.get("predictions", [])

if not predictions:
    st.info("No prediction history available yet. Score customers on the Single or Batch pages to view logs here.")
    st.stop()

df = pd.DataFrame(predictions)

if "primary_churn_driver" not in df.columns:
    df["primary_churn_driver"] = df["top_reasons"].apply(
        lambda reasons: reasons[0]["feature"] if isinstance(reasons, list) and len(reasons) > 0 else None
    )

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    search_id = st.text_input("Search by Customer ID", placeholder="Enter customer ID...")

with col2:
    selected_tiers = st.multiselect(
        "Filter by Risk Tier",
        options=["Low", "Medium", "High"],
        default=["Low", "Medium", "High"],
    )

with col3:
    st.write("")
    st.write("")
    if st.button("Clear Session History", use_container_width=True):
        st.session_state["predictions"] = []
        st.rerun()

filtered_df = df[df["risk_tier"].isin(selected_tiers)].copy()

if search_id.strip():
    filtered_df = filtered_df[
        filtered_df["customer_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)
    ]

if "timestamp" in filtered_df.columns:
    filtered_df = filtered_df.sort_values(by="timestamp", ascending=False)

st.divider()
st.caption(f"Showing {len(filtered_df)} of {len(df)} total session records.")

display_cols = [
    c
    for c in [
        "timestamp",
        "customer_id",
        "churn_probability",
        "churn_prediction",
        "risk_tier",
        "primary_churn_driver",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "model_version",
    ]
    if c in filtered_df.columns
]

st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

st.download_button(
    label="Download Full Session History (CSV)",
    data=filtered_df[display_cols].to_csv(index=False).encode("utf-8"),
    file_name="prediction_session_history.csv",
    mime="text/csv",
    use_container_width=True,
)
