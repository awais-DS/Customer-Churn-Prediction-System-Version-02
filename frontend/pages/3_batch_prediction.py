import pandas as pd
import streamlit as st
from api_client import predict_csv

st.title("Batch Customer Prediction")
st.markdown("Upload a CSV file containing customer records to score them in bulk and download the results.")

with st.expander("Required CSV Format & Sample Template"):
    st.markdown(
        "Your CSV should contain the model's 6 feature columns (`tenure`, `MonthlyCharges`, `TotalCharges`, "
        "`gender`, `Dependents`, `Contract`) and an optional `customer_id` column. "
        "Invalid or duplicate rows are automatically skipped and reported without stopping valid rows."
    )
    sample_df = pd.DataFrame(
        [
            {
                "customer_id": "CUST-101",
                "tenure": 3,
                "MonthlyCharges": 85.5,
                "TotalCharges": 256.5,
                "gender": "Female",
                "Dependents": "No",
                "Contract": "Month-to-month",
            },
            {
                "customer_id": "CUST-102",
                "tenure": 48,
                "MonthlyCharges": 55.0,
                "TotalCharges": 2640.0,
                "gender": "Male",
                "Dependents": "Yes",
                "Contract": "Two year",
            },
        ]
    )
    st.dataframe(sample_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download Sample CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_churn_customers.csv",
        mime="text/csv",
    )

uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

if uploaded_file is not None:
    if st.button("Run Batch Prediction", type="primary", use_container_width=True):
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)

            with st.spinner("Scoring batch via FastAPI..."):
                batch_result = predict_csv(uploaded_file)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows Received", batch_result["total_rows_received"])
            m2.metric("Successfully Scored", batch_result["total_processed"])
            m3.metric("Skipped Rows", batch_result["skipped_count"])
            m4.metric("High-Risk Customers", batch_result["high_risk_count"])

            warnings_list = batch_result.get("warnings", [])
            if warnings_list:
                st.warning(f"{len(warnings_list)} row(s) were skipped due to validation issues or duplicates:")
                st.dataframe(pd.DataFrame(warnings_list), use_container_width=True, hide_index=True)

            predictions_list = batch_result.get("predictions", [])
            if predictions_list:
                input_lookup = {}
                if "customer_id" in uploaded_df.columns:
                    for _, r in uploaded_df.dropna(subset=["customer_id"]).iterrows():
                        cid = str(r["customer_id"]).strip()
                        if cid not in input_lookup:
                            input_lookup[cid] = r.to_dict()

                merged_records = []
                for pred in predictions_list:
                    cid = pred["customer_id"]
                    base_features = input_lookup.get(cid, {})
                    top_driver = pred["top_reasons"][0]["feature"] if pred.get("top_reasons") else None
                    merged = {
                        **base_features,
                        **pred,
                        "primary_churn_driver": top_driver,
                    }
                    merged_records.append(merged)

                st.session_state["predictions"].extend(merged_records)

                results_df = pd.DataFrame(merged_records)
                display_cols = [
                    c
                    for c in [
                        "customer_id",
                        "churn_probability",
                        "churn_prediction",
                        "risk_tier",
                        "primary_churn_driver",
                        "Contract",
                        "tenure",
                        "MonthlyCharges",
                    ]
                    if c in results_df.columns
                ]

                st.subheader("Scored Customer Results")
                st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)

                st.download_button(
                    label="Download Scored Results CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="scored_churn_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.error("No valid rows could be scored from this file.")

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
