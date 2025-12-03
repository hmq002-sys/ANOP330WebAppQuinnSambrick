import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Reunion Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Bucknell Reunion Attendance Predictor")
st.write("Predict whether alumni will accept reunion invitations")

@st.cache_resource
def load_model():
    model_data = pickle.load(open("model.pkl", "rb"))
    scaler_data = pickle.load(open("scaler.pkl", "rb"))
    features = pickle.load(open("features.pkl", "rb"))
    return model_data, scaler_data, features

model_data, scaler_data, features = load_model()
st.success("✅ Model loaded successfully!")

col1, col2 = st.columns(2)

with col1:
    years = st.number_input("Reunion Years Out", 0, 50, 5)
    class_year = st.number_input("Bucknell Class Year", 1900, 2024, 2000)

with col2:
    peer = st.radio("Peer-to-Peer Contact", ["No", "Yes"])
    parent = st.radio("Current Parent", ["No", "Yes"])

if st.button("Predict Attendance", type="primary"):
    peer_num = 1 if peer == "Yes" else 0
    parent_num = 1 if parent == "Yes" else 0
    
    # Scale the input
    input_values = np.array([[years, class_year, peer_num, parent_num]], dtype=float)
    scaled = (input_values - scaler_data['mean']) / scaler_data['std']
    
    # Calculate probability (simulated model)
    prob = (np.dot(scaled, model_data['coefficients']) + model_data['intercept'])[0]
    prob = 1 / (1 + np.exp(-prob))  # Sigmoid
    
    st.divider()
    st.subheader("Prediction Result")
    st.metric("Acceptance Probability", f"{prob:.1%}")
    
    if prob > 0.5:
        st.success(f"✅ Likely to Accept")
    else:
        st.warning(f"❌ Unlikely to Accept")

st.caption("Demo Model | Streamlit App")
