
import streamlit as st
import pandas as pd
import pickle

st.title("🎓 Reunion Predictor")

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

reunion_years = st.number_input("Reunion Years Out", 0, 50, 5)
class_year = st.number_input("Class Year", 1900, 2024, 2000)
peer = st.selectbox("Peer Contact", [0, 1])
parent = st.selectbox("Current Parent", [0, 1])

if st.button("Predict"):
    input_data = [[reunion_years, class_year, peer, parent]]
    input_df = pd.DataFrame(input_data, columns=features)
    scaled = scaler.transform(input_df)
    prob = model.predict_proba(scaled)[0][1]
    st.write(f"Probability: {prob:.1%}")
