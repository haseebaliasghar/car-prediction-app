import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------------------------------------------------
# CONFIGURATION & SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Car Safety Prediction", page_icon="🚗")

st.title("🚗 Car Safety Evaluation App")
st.write("Predict the safety class of a car based on its features using your trained SVM models.")

# ---------------------------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------------------------
# We use st.cache_resource so we don't reload files on every interaction
@st.cache_resource
def load_artifacts():
    try:
        # Load Encoders and Column structure
        feature_cols = pickle.load(open("feature_columns.pkl", "rb"))
        label_enc = pickle.load(open("label_encoder.pkl", "rb"))
        
        # Load Models
        models = {
            "Linear Kernel": pickle.load(open("car_svm_linear.pkl", "rb")),
            "Polynomial Kernel": pickle.load(open("car_svm_poly.pkl", "rb")),
            "RBF Kernel": pickle.load(open("car_svm_rbf.pkl", "rb"))
        }
        return feature_cols, label_enc, models
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.error("Please ensure .pkl files are in the same directory as app.py")
        return None, None, None

feature_columns, label_encoder, trained_models = load_artifacts()

if not trained_models:
    st.stop() # Stop execution if files aren't found

# ---------------------------------------------------------
# SIDEBAR - USER INPUTS
# ---------------------------------------------------------
st.sidebar.header("User Input Features")

# Helper function to match the exact strings used in your CSV
def user_input_features():
    # 1. Buying Price
    buying = st.sidebar.selectbox("Buying Price", ["vhigh", "high", "med", "low"])
    
    # 2. Maintenance Cost
    maint = st.sidebar.selectbox("Maintenance Cost", ["vhigh", "high", "med", "low"])
    
    # 3. Number of Doors
    # Note: In your dataset, '5more' is a string
    doors = st.sidebar.selectbox("Number of Doors", ["2", "3", "4", "5more"])
    
    # 4. Capacity (Persons)
    # Note: 'more' is a string
    persons = st.sidebar.selectbox("Capacity (Persons)", ["2", "4", "more"])
    
    # 5. Luggage Boot Size
    lug_boot = st.sidebar.selectbox("Luggage Boot Size", ["small", "med", "big"])
    
    # 6. Safety
    safety = st.sidebar.selectbox("Safety Level", ["low", "med", "high"])

    # Create a dictionary
    data = {
        'buying': buying,
        'maint': maint,
        'doors': doors,
        'persons': persons,
        'lug_boot': lug_boot,
        'safety': safety
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display User Selection
st.subheader("Current Car Configuration")
st.dataframe(input_df)

# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
# 1. One-Hot Encode the user input
# drop_first=True must be set to match your training logic
input_encoded = pd.get_dummies(input_df, drop_first=True)

# 2. ALIGN COLUMNS (Crucial Step)
# The user input might generate fewer columns than the training set 
# (e.g., if 'buying' is 'low', get_dummies won't generate columns for 'high' or 'med').
# We use .reindex() to force the columns to match the training signature.
input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
st.subheader("Model Prediction")

# Select Model
model_choice = st.selectbox("Select SVM Kernel", list(trained_models.keys()))
selected_model = trained_models[model_choice]

# Button to Predict
if st.button("Evaluate Car"):
    
    # Make Prediction
    prediction_index = selected_model.predict(input_final)[0]
    
    # Convert numerical label back to string (unacc, acc, etc.)
    prediction_label = label_encoder.inverse_transform([prediction_index])[0]
    
    # Styling the result
    color_map = {
        "unacc": "red",
        "acc": "blue",
        "good": "orange",
        "vgood": "green"
    }
    
    readable_map = {
        "unacc": "Unacceptable",
        "acc": "Acceptable",
        "good": "Good",
        "vgood": "Very Good"
    }

    result_color = color_map.get(prediction_label, "black")
    result_text = readable_map.get(prediction_label, prediction_label)

    st.markdown(f"### Prediction: :{result_color}[{result_text}]")
    
    # Optional: Show probabilities if model supports it (SVM default usually doesn't unless probability=True was set)
    # Since standard SVC() was used without probability=True, we cannot show percentages easily.
    
    st.info(f"Using **{model_choice}**. The raw class label is '{prediction_label}'.")
