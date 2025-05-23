import streamlit as st
import pandas as pd
import joblib

st.title("Credit Card Fraud Detection")

@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('fraud_model.pkl')
    return scaler, model

scaler, model = load_models()

uploaded_file = st.file_uploader("Upload CSV file for prediction", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)
    
    data['Amount'] = scaler.transform(data[['Amount']])
    
    expected_features = model.get_booster().feature_names
    data = data[expected_features]

    # Predict probabilities
    probs = model.predict_proba(data)[:, 1]
    
    st.write("Fraud probabilities for each transaction:")
    st.write(probs)

    # Threshold slider
    threshold = st.slider("Select classification threshold", 0.0, 1.0, 0.5, 0.01)

    preds = (probs >= threshold).astype(int)
    st.write(f"Predictions with threshold {threshold}:")
    st.write(preds)
