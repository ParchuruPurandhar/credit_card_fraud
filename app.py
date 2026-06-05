import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("credit_model.pkl")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")

st.write(
    "Upload a CSV file containing transaction records."
)
st.info("""
Expected CSV format:

Time, V1, V2, ..., V28, Amount

Do NOT include the Class column.
""")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    try:
        predictions = model.predict(df)

        df["Prediction"] = predictions

        df["Prediction"] = df["Prediction"].map(
            {
                0: "Legitimate",
                1: "Fraud"
            }
        )

        st.subheader("Prediction Results")
        st.dataframe(df)

        fraud_count = (df["Prediction"] == "Fraud").sum()
        legit_count = (df["Prediction"] == "Legitimate").sum()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fraud Transactions", fraud_count)

        with col2:
            st.metric("Legitimate Transactions", legit_count)

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="fraud_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
