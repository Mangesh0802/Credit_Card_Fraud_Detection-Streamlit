import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import get_prediction, batch_predict, feature_cols

st.set_page_config(layout="wide", page_title="Real-time Fraud Detection Dashboard")

st.title("💳 Real-time Credit Card Fraud Detector")
st.markdown("Use this dashboard to simulate a single transaction or batch process a file to check for fraud.")

tab1, tab2, tab3 = st.tabs([
    "🚀 Single Transaction Simulation",
    "📈 Model Performance",
    "📥 Batch Processor (Bonus)"
])

# -------------------- TAB 1: SINGLE TRANSACTION --------------------
with tab1:
    col_input, col_output = st.columns([1, 1.2])

    with col_input:
        st.header("Transaction Input Form")

        with st.form("transaction_input_form"):
            st.subheader("Key Features")

            transaction_data = {}
            transaction_data['V1'] = st.slider('V1 (PCA Feature 1)', -30.0, 2.0, -10.0, 0.1)
            transaction_data['V2'] = st.slider('V2 (PCA Feature 2)', -15.0, 20.0, 10.0, 0.1)
            transaction_data['V3'] = st.slider('V3 (PCA Feature 3)', -30.0, 1.0, -25.0, 0.1)

            st.subheader("Transaction Details")
            transaction_data['Amount'] = st.number_input('Transaction Amount ($)', 0.0, 2000.0, 10.0, 1.0)
            transaction_data['Time'] = st.number_input('Time (Seconds elapsed)', 0.0, 172792.0, 10000.0, 100.0)

            st.subheader("Other Anonymized Features")
            for i in range(4, 29):
                transaction_data[f'V{i}'] = 0.0

            submitted = st.form_submit_button("Predict Fraud Status")

    with col_output:
        st.header("Prediction Result")

        if submitted:
            with st.spinner('Analyzing transaction...'):
                result = get_prediction(transaction_data)

            pred = result['prediction']
            conf = result['confidence']

            if pred == 1:
                st.error("🚨 FRAUD DETECTED")
                color = "red"
            else:
                st.success("✅ Legitimate Transaction")
                color = "green"

            st.metric(
                label="Model Confidence (P(Fraud))",
                value=f"{conf:.4f}",
                delta_color="off"
            )

            # --- Basic feature importance placeholder (no SHAP) ---
            if 'feature_importance' in result:
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': result['feature_importance']
                }).sort_values(by='Importance', ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
                ax.set_title("Top Feature Importance (Model-based)")
                ax.set_xlabel("Feature Importance")
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.info("Feature importance visualization not available in this version.")

# -------------------- TAB 2: MODEL PERFORMANCE --------------------
with tab2:
    st.header("Model Reliability and Evaluation Metrics")
    st.markdown("These charts demonstrate the overall performance of the model on unseen **test data**.")

    col_matrix, col_prc = st.columns(2)
    with col_matrix:
        st.subheader("Confusion Matrix")
        st.image("https://i.imgur.com/8QxM2o4.png", caption="Mock Confusion Matrix for Imbalanced Data")
        st.markdown("""
        - **Goal:** Maximize True Positives (catching fraud).
        - **Metric Focus:** High **Recall** (True Positive Rate).
        """)

    with col_prc:
        st.subheader("Precision-Recall Curve (PRC)")
        st.image("https://i.imgur.com/M6L5c7k.png", caption="Mock Precision-Recall Curve")
        st.markdown("""
        - **Goal:** Maintain high Precision as Recall increases.
        - **Threshold:** The optimal point on this curve determines the final prediction threshold.
        """)

# -------------------- TAB 3: BATCH PROCESSING --------------------
with tab3:
    st.header("Process Transactions in Bulk")
    st.info("Upload a CSV file containing transactions to get predictions for all rows.")

    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Data Preview")
            st.dataframe(df_upload.head())

            df_results = batch_predict(df_upload.copy())

            st.subheader("Batch Prediction Results")
            st.dataframe(df_results)

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df_results)

            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name='fraud_predictions.csv',
                mime='text/csv',
            )

        except ValueError as e:
            st.error(f"Error processing file: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
