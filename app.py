import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="Hospital AI Dashboard", layout="wide")

# ---------------- LOGIN ----------------

if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Hospital Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

    st.stop()

# ---------------- DASHBOARD ----------------

st.title("🏥 AI Hospital Forecasting Dashboard")

# Model selection
model_choice = st.selectbox(
    "Select Prediction Model",
    ["Linear Regression", "Random Forest"]
)

# Dataset upload
uploaded_file = st.file_uploader(
    "Upload Hospital Dataset (Optional)",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("HDHI_Admission_data.csv")

# Convert date
data["D.O.A"] = pd.to_datetime(data["D.O.A"], errors="coerce")
data = data.dropna(subset=["D.O.A"])

# Daily patient counts
daily_counts = data.groupby("D.O.A").size().reset_index(name="Patient_Count")

# Load model
if model_choice == "Linear Regression":
    model = pickle.load(open("model.pkl", "rb"))
else:
    model = pickle.load(open("rf_model.pkl", "rb"))

# Date input
selected_date = st.date_input("Select Date")

if st.button("Predict"):

    date_ordinal = pd.Timestamp(selected_date).toordinal()

    prediction = model.predict([[date_ordinal]])

    predicted_value = max(0, int(prediction[0]))

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Patients", predicted_value)

    with col2:
        icu = int(predicted_value * 0.2)
        st.metric("Estimated ICU Occupancy", icu)

    if predicted_value > 50:
        st.warning("⚠ High Patient Load Expected")
    else:
        st.success("Normal Patient Load")

# ---------------- GRAPH ----------------

st.subheader("📈 Last 30 Days Patient Trend")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=daily_counts["D.O.A"].tail(30),
    y=daily_counts["Patient_Count"].tail(30),
    mode='lines+markers',
    name='Patient Trend'
))

fig.update_layout(
    title="Last 30 Days Patient Trend",
    xaxis_title="Date",
    yaxis_title="Patient Count"
)

st.plotly_chart(fig)

# ---------------- LOGOUT ----------------

if st.button("Logout"):
    st.session_state.login = False
    st.rerun()

st.caption("Developed as Advanced Healthcare AI Forecasting System")