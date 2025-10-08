import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


from model import get_prediction, batch_predict, feature_cols 

st.set_page_config(layout="wide", page_title="Real-time Fraud Detection Dashboard")

st.title("💳 Real-time Credit Card Fraud Detector")
st.markdown("Use this dashboard to simulate a single transaction or batch process a file to check for fraud.")


tab1, tab2, tab3 = st.tabs(["🚀 Single Transaction Simulation", "📈 Model Performance (XAI)", "📥 Batch Processor (Bonus)"])


with tab1:
    col_input, col_output = st.columns([1, 1.2])

    with col_input:
        st.header("Transaction Input Form")
        
       
        with st.form("transaction_input_form"):
            
           
            st.subheader("Key Features")
           
            
            transaction_data = {}
            
            transaction_data['V1'] = st.slider('V1 (PCA Feature 1)', min_value=-30.0, max_value=2.0, value=-10.0, step=0.1)
            transaction_data['V2'] = st.slider('V2 (PCA Feature 2)', min_value=-15.0, max_value=20.0, value=10.0, step=0.1)
            transaction_data['V3'] = st.slider('V3 (PCA Feature 3)', min_value=-30.0, max_value=1.0, value=-25.0, step=0.1)

            st.subheader("Transaction Details")
            transaction_data['Amount'] = st.number_input('Transaction Amount ($)', min_value=0.0, max_value=2000.0, value=10.0, step=1.0)
            transaction_data['Time'] = st.number_input('Time (Seconds elapsed)', min_value=0.0, max_value=172792.0, value=10000.0, step=100.0)

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

            st.markdown(f"### Explanation (SHAP Values)")
            st.info("The plot below shows how much each feature contributed to the model's final confidence score.")

            
            
            
            shap_df = pd.DataFrame({
                'Feature': feature_cols,
                'SHAP Value': result['shap_values']
            }).sort_values(by='SHAP Value', key=abs, ascending=False).head(10)
            
           
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['red' if v > 0 else 'blue' for v in shap_df['SHAP Value']]
            ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
            ax.set_title("Feature Contribution to Prediction (SHAP)")
            ax.set_xlabel("SHAP Value (Impact on Model Output)")
            ax.invert_yaxis() 
            st.pyplot(fig)
            
            

with tab2:
    st.header("Model Reliability and Evaluation Metrics")
    st.markdown("These charts demonstrate the overall performance of the model on unseen **test data**.")
    
    st.info("To create this tab, you need to run your model on a test set, calculate metrics, and save the results for plotting here (e.g., using `st.pyplot(confusion_matrix_plot)`).")

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
        - **Threshold:** The optimal point on this curve determines the final prediction threshold (e.g., not 0.5).
        """)


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