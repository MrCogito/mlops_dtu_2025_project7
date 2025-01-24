import streamlit as st
import requests

st.title("Flood Prediction")

st.write("""
Upload a CSV with columns: "event_id", "precipitation", 
and this app will send it to a FastAPI backend for inference.
""")

backend_url = st.text_input("Backend URL", value="http://localhost:8000/predict_csv")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    st.write("Preview of uploaded CSV (first few lines):")
    st.write(uploaded_file.getvalue().decode("utf-8").split("\n")[:5])

    if st.button("Run Prediction"):
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(backend_url, files=files, timeout=30)
            response.raise_for_status()  # raise an exception for 4xx/5xx errors
            data = response.json()

            st.success("Predictions received!")
            st.write(data)
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")