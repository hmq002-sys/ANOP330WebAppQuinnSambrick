import streamlit as st
import pandas as pd
import pickle

st.title("Test App")

try:
    model = pickle.load(open("model.pkl", "rb"))
    st.success("Model loaded!")
    st.write("Features:", list(model.feature_names_in_))
except:
    st.error("Could not load model")

st.write("Hello World")
