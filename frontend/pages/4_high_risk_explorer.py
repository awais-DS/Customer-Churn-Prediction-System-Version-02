import pandas as pd
import streamlit as st

st.title("High-Risk Customer Explorer")
st.markdown("Filter scored customers above a chosen risk threshold to prioritize retention outreach.")

predictions = st.session_state.get("predictions", [])

if not predictions:
    st.info("No customers have been scored yet. Run Single or Batch predictions first to populate this explorer.")
    st.stop()

df = pd.DataFrame(predictions)

if "primary_churn_driver" not in df.columns:
    df["primary_churn_driver"] = df["top_reasons"].apply(
        lambda reasons: reasons[0]["feature"] if isinstance(reasons, list) and len(reasons) > 0 else None
    )

col1, col2 = st.columns([2, 1])

with col1:
    threshold = st.slider(
        "Minimum Churn Probability Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05,
    )

with col2:
    contract_options = ["All"]
    if "Contract" in df.columns:
        contract_options += sorted(df["Contract"].dropna().unique().tolist())
    selected_contract = st.selectbox("Filter by Contract Type", options=contract_options)

filtered_df = df[df["churn_probability"] >= threshold].copy()

if selected_contract != "All" and "Contract" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]

filtered_df = filtered_df.sort_values(by="churn_probability", ascending=False)

m1, m2, m3 = st.columns(3)
m1.metric("Flagged High-Risk Customers", len(filtered_df))
m2.metric("Average Risk Score", f"{(filtered_df['churn_probability'].mean() * 100):.1f}%" if not filtered_df.empty else "0.0%")

revenue_exposure = 0.0
if not filtered_df.empty and "MonthlyCharges" in filtered_df.columns:
    revenue_exposure = float(filtered_df["MonthlyCharges"].fillna(0).sum())
m3.metric("Monthly Revenue Exposure", f"${revenue_exposure:,.2f}")

st.divider()

if filtered_df.empty:
    st.warning("No scored customers match the current threshold and filter criteria.")
else:
    display_cols = [
        c
        for c in [
            "customer_id",
            "churn_probability",
            "risk_tier",
            "primary_churn_driver",
            "Contract",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ]
        if c in filtered_df.columns
    ]

    st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

    st.download_button(
        label="Export Retention Action List (CSV)",
        data=filtered_df[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="high_risk_retention_list.csv",
        mime="text/csv",
        use_container_width=True,
    )
