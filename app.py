# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Car Acceptability Prediction", layout="centered")
st.title("Car Acceptability Prediction App")
st.write("Predict the acceptability of a car based on its features.")

# ---------------------------------------------------------
# LOAD MODELS AND ENCODER
# ---------------------------------------------------------
kernels = ["linear", "poly", "rbf"]
models = {}
for k in kernels:
    models[k] = pickle.load(open(f"car_svm_{k}.pkl", "rb"))

label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------
st.sidebar.header("Select Car Features")

buying = st.sidebar.selectbox("Buying Price", ["low", "med", "high", "vhigh"])
maint = st.sidebar.selectbox("Maintenance Cost", ["low", "med", "high", "vhigh"])
doors = st.sidebar.selectbox("Number of Doors", ["2", "3", "4", "5more"])
persons = st.sidebar.selectbox("Capacity (Persons)", ["2", "4", "more"])
lug_boot = st.sidebar.selectbox("Luggage Boot Size", ["small", "med", "big"])
safety = st.sidebar.selectbox("Estimated Safety", ["low", "med", "high"])

selected_kernel = st.sidebar.selectbox("Select SVM Kernel", kernels)

# ---------------------------------------------------------
# CREATE INPUT DATAFRAME
# ---------------------------------------------------------
# Start with all zeros
input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

# Fill 1s based on user selections (drop_first=True logic)
if buying != "low":
    input_df[f"buying_{buying}"] = 1
if maint != "low":
    input_df[f"maint_{maint}"] = 1
if doors != "2":
    input_df[f"doors_{doors}"] = 1
if persons != "2":
    input_df[f"persons_{persons}"] = 1
if lug_boot != "small":
    input_df[f"lug_boot_{lug_boot}"] = 1
if safety != "low":
    input_df[f"safety_{safety}"] = 1

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.sidebar.button("Predict"):
    model = models[selected_kernel]
    pred_num = model.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]

    st.subheader("Prediction Result")
    st.write(f"Predicted car acceptability: **{pred_label}**")
