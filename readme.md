# Car Evaluation Prediction App

This is a Streamlit web app that predicts the acceptability of a car based on its features using pre-trained SVM models.

## Features:
- Choose car attributes from dropdowns:
  - Buying price
  - Maintenance cost
  - Number of doors
  - Persons capacity
  - Luggage boot size
  - Safety level
- Select which SVM kernel to use for prediction:
  - Linear, Polynomial, RBF
- See prediction result and model confidence.

## Run locally:
1. Install dependencies:
   pip install -r requirements.txt
2. Run Streamlit app:
   streamlit run app.py
