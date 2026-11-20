import streamlit as st
from api_client import predict_single

st.title("Single Customer Churn Prediction")
st.markdown("Enter a customer's profile below to calculate real-time churn risk and key drivers.")

with st.form("single_customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        customer_id = st.text_input("Customer ID (Optional)", placeholder="e.g., CUST-1001")
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female"])

    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
        dependents = st.selectbox("Dependents", options=["No", "Yes"])

    with col3:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=20000.0,
            value=float(tenure * monthly_charges),
            step=10.0,
        )
        contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"])

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True, type="primary")

if submitted:
    payload = {
        "customer_id": customer_id.strip() if customer_id.strip() else None,
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
        "gender": gender,
        "Dependents": dependents,
        "Contract": contract,
    }

    try:
        with st.spinner("Scoring customer via FastAPI..."):
            result = predict_single(payload)

        record = {**payload, **result}
        st.session_state["predictions"].append(record)

        st.divider()
        st.subheader(f"Prediction Result for `{result['customer_id']}`")

        m1, m2, m3 = st.columns(3)
        prob_pct = result["churn_probability"] * 100
        m1.metric("Churn Probability", f"{prob_pct:.1f}%")
        m2.metric("Risk Tier", result["risk_tier"])
        m3.metric("Model Decision", "Likely to Churn" if result["churn_prediction"] == 1 else "Likely to Stay")

        if result["risk_tier"] == "High":
            st.error("High Churn Risk: Immediate retention outreach recommended (e.g., contract upgrade offer).")
        elif result["risk_tier"] == "Medium":
            st.warning("Medium Churn Risk: Monitor engagement and consider targeted loyalty perks.")
        else:
            st.success("Low Churn Risk: Customer profile indicates strong retention stability.")

        st.subheader("Top Factors Driving This Score")
        for reason in result.get("top_reasons", []):
            st.write(f"- **{reason['feature']}** — {reason['direction']} (`impact: {reason['impact']:+.4f}`)")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
