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
    # Define mappings from UI Display -> Backend Value
    buying_map = {
        "Low": "low",
        "Medium": "med",
        "High": "high",
        "Very High": "vhigh"
    }
    
    maint_map = {
        "Low": "low",
        "Medium": "med",
        "High": "high",
        "Very High": "vhigh"
    }
    
    doors_map = {
        "2": "2",
        "3": "3",
        "4": "4",
        "5 or more": "5more"
    }
    
    persons_map = {
        "2": "2",
        "4": "4",
        "More than 4": "more"
    }
    
    lug_boot_map = {
        "Small": "small",
        "Medium": "med",
        "Big": "big"
    }
    
    safety_map = {
        "Low": "low",
        "Medium": "med",
        "High": "high"
    }

    # 1. Buying Price
    buying_ui = st.sidebar.selectbox("Buying Price", list(buying_map.keys()))
    
    # 2. Maintenance Cost
    maint_ui = st.sidebar.selectbox("Maintenance Cost", list(maint_map.keys()))
    
    # 3. Number of Doors
    doors_ui = st.sidebar.selectbox("Number of Doors", list(doors_map.keys()))
    
    # 4. Capacity (Persons)
    persons_ui = st.sidebar.selectbox("Capacity (Persons)", list(persons_map.keys()))
    
    # 5. Luggage Boot Size
    lug_boot_ui = st.sidebar.selectbox("Luggage Boot Size", list(lug_boot_map.keys()))
    
    # 6. Safety
    safety_ui = st.sidebar.selectbox("Safety Level", list(safety_map.keys()))

    # Create a dictionary using the mapped backend values
    data = {
        'buying': buying_map[buying_ui],
        'maint': maint_map[maint_ui],
        'doors': doors_map[doors_ui],
        'persons': persons_map[persons_ui],
        'lug_boot': lug_boot_map[lug_boot_ui],
        'safety': safety_map[safety_ui]
    }
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display User Selection
st.subheader("Current Car Configuration")
# We display the raw input dataframe here, which now contains the backend codes.
# If you want to show the pretty names here too, we would need a separate display dataframe.
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
