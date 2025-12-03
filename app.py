import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Reunion Predictor", layout="centered")

st.title("🎓 Bucknell Reunion Attendance Predictor")
st.write("Predict whether alumni will accept reunion invitations")

# Load model
@st.cache_resource
def load_model():
    return (
        pickle.load(open("model.pkl", "rb")),
        pickle.load(open("scaler.pkl", "rb")),
        pickle.load(open("features.pkl", "rb"))
    )

model, scaler, features = load_model()

st.success("✅ Model loaded successfully!")

# Inputs
col1, col2 = st.columns(2)
with col1:
    years = st.number_input("Reunion Years Out", 0, 50, 5)
    class_year = st.number_input("Class Year", 1900, 2024, 2000)
with col2:
    peer = st.radio("Peer Contact?", ["No", "Yes"])
    parent = st.radio("Current Parent?", ["No", "Yes"])

# Predict
if st.button("Predict Attendance", type="primary"):
    # Convert inputs
    peer_num = 1 if peer == "Yes" else 0
    parent_num = 1 if parent == "Yes" else 0
    
    # Create input array
    input_data = [[years, class_year, peer_num, parent_num]]
    input_df = pd.DataFrame(input_data, columns=features)
    
    # Scale and predict
    scaled = scaler.transform(input_df)
    prob = model.predict_proba(scaled)[0][1]
    
    # Display
    st.divider()
    st.subheader("Prediction Result")
    st.metric("Acceptance Probability", f"{prob:.1%}")
    
    if prob > 0.5:
        st.success(f"✅ **Likely to Accept** (Probability: {prob:.1%})")
    else:
        st.warning(f"❌ **Unlikely to Accept** (Probability: {prob:.1%})")

st.caption("Model: Random Forest | Features: 4 variables")
