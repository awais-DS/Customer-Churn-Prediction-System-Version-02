import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Executive Dashboard")
st.markdown("High-level overview of the scored customer population and churn risk exposure.")

predictions = st.session_state.get("predictions", [])

if not predictions:
    st.info("No customers have been scored yet. Use Single Customer Prediction or Batch Customer Prediction to populate this dashboard.")
    st.stop()

df = pd.DataFrame(predictions)

total_scored = len(df)
predicted_churners = int((df["churn_prediction"] == 1).sum())
churn_rate = (predicted_churners / total_scored) * 100
high_risk_count = int((df["risk_tier"] == "High").sum())

mrr_at_risk = 0.0
if "MonthlyCharges" in df.columns:
    mrr_at_risk = float(df.loc[df["risk_tier"] == "High", "MonthlyCharges"].fillna(0).sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Scored Customers", f"{total_scored:,}")
k2.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
k3.metric("High-Risk Customers", f"{high_risk_count:,}")
k4.metric("High-Risk Monthly Revenue", f"${mrr_at_risk:,.2f}")

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Risk Tier Distribution")
    tier_counts = (
        df["risk_tier"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
        .reset_index()
    )
    tier_counts.columns = ["Risk Tier", "Customers"]

    fig_tier = px.bar(
        tier_counts,
        x="Risk Tier",
        y="Customers",
        color="Risk Tier",
        color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
        text="Customers",
    )
    fig_tier.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customer Count")
    st.plotly_chart(fig_tier, use_container_width=True)

with col_right:
    st.subheader("Churn Probability Distribution")
    fig_hist = px.histogram(
        df,
        x="churn_probability",
        nbins=15,
        color="risk_tier",
        color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
        labels={"churn_probability": "Churn Probability", "risk_tier": "Risk Tier"},
    )
    fig_hist.update_layout(yaxis_title="Customer Count")
    st.plotly_chart(fig_hist, use_container_width=True)

if "Contract" in df.columns and df["Contract"].notna().any():
    st.subheader("Average Churn Probability by Contract Type")
    contract_summary = (
        df.dropna(subset=["Contract"])
        .groupby("Contract", as_index=False)["churn_probability"]
        .mean()
    )
    contract_summary["churn_probability"] = contract_summary["churn_probability"] * 100
    fig_contract = px.bar(
        contract_summary,
        x="Contract",
        y="churn_probability",
        text_auto=".1f",
        labels={"churn_probability": "Avg Churn Probability (%)", "Contract": "Contract Type"},
    )
    st.plotly_chart(fig_contract, use_container_width=True)
