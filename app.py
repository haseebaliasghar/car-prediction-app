import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="Car Evaluation Prediction App", layout="centered")
st.title("🚗 Car Acceptability Prediction")
st.write("Select the car features and SVM kernel to predict the acceptability of a car.")

# -----------------------------
# Load models and label encoder
# -----------------------------
models = {
    "Linear": pickle.load(open("car_svm_linear.pkl", "rb")),
    "Polynomial": pickle.load(open("car_svm_poly.pkl", "rb")),
    "RBF": pickle.load(open("car_svm_rbf.pkl", "rb"))
}

label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# -----------------------------
# Define dropdowns
# -----------------------------
st.sidebar.header("Select Car Attributes")

buying   = st.sidebar.selectbox("Buying Price",  ["vhigh", "high", "med", "low"])
maint    = st.sidebar.selectbox("Maintenance Cost", ["vhigh", "high", "med", "low"])
doors    = st.sidebar.selectbox("Number of Doors", ["2", "3", "4", "5more"])
persons  = st.sidebar.selectbox("Persons Capacity", ["2", "4", "more"])
lug_boot = st.sidebar.selectbox("Luggage Boot Size", ["small", "med", "big"])
safety   = st.sidebar.selectbox("Safety Level", ["low", "med", "high"])

# Kernel selector
selected_kernel = st.sidebar.selectbox(
    "Select SVM Kernel",
    ["Linear", "Polynomial", "RBF"]
)

# -----------------------------
# Encode input features exactly like training
# -----------------------------
# List of all one-hot columns generated during training
# (from pd.get_dummies with drop_first=True)
feature_columns = [
    'buying_high', 'buying_med', 'buying_vhigh',
    'maint_high', 'maint_med', 'maint_vhigh',
    'doors_3', 'doors_4', 'doors_5more',
    'persons_4', 'persons_more',
    'lug_boot_med', 'lug_boot_small',
    'safety_low', 'safety_med'
]

# Create a DataFrame with zeros
input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

# Fill 1s based on selected values
if buying != "low":
    input_df[f"buying_{buying}"] = 1
if maint != "low":
    input_df[f"maint_{maint}"] = 1
if doors != "2":
    input_df[f"doors_{doors}"] = 1
if persons != "2":
    input_df[f"persons_{persons}"] = 1
if lug_boot != "big":
    input_df[f"lug_boot_{lug_boot}"] = 1
if safety != "high":
    input_df[f"safety_{safety}"] = 1

# -----------------------------
# Predict
# -----------------------------
if st.sidebar.button("Predict"):
    model = models[selected_kernel]
    pred_num = model.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]

    st.subheader("Prediction Result")
    st.success(f"The car is predicted as: **{pred_label.upper()}**")

    # If model supports probability, show confidence
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df).max() * 100
        st.write(f"Model confidence: **{prob:.2f}%**")
