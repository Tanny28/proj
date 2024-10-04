# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Page title
st.title("AI-Powered Climate Change Prediction Platform")

# Sidebar for uploading data
st.sidebar.header("Upload Climate Data")
uploaded_file = st.sidebar.file_uploader("Upload your climate dataset in CSV format", type=["csv"])

# Check if the user has uploaded a dataset
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    
    # Load the data into a dataframe
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Climate Dataset")
    st.dataframe(data.head())

    # Select features for prediction
    st.sidebar.subheader("Select Features for Prediction")
    features = st.sidebar.multiselect("Choose features from the dataset", options=data.columns)

    # Select the target variable
    target = st.sidebar.selectbox("Select the Target Variable", options=data.columns)

    if features and target:
        st.write(f"### Selected Features: {features}")
        st.write(f"### Target Variable: {target}")

        # Split the data into train and test sets
        X = data[features]
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build a RandomForest Model for Climate Prediction
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        st.write(f"### Root Mean Squared Error (RMSE) of the model: {rmse:.2f}")

        # Plot actual vs predicted
        st.write("### Actual vs Predicted Values")
        plt.figure(figsize=(10,5))
        plt.plot(y_test.values, label="Actual")
        plt.plot(y_pred, label="Predicted", linestyle='--')
        plt.legend()
        st.pyplot(plt.figure())

else:
    st.warning("Please upload a CSV dataset to start")

# Display an input form for prediction based on live data
st.sidebar.subheader("Predict Climate Indicator")
user_inputs = {}
if features:
    for feature in features:
        user_inputs[feature] = st.sidebar.number_input(f"Input {feature}")

    # Convert user input into a dataframe for prediction
    if st.sidebar.button("Predict Climate Indicator"):
        input_data = pd.DataFrame([user_inputs])
        prediction = model.predict(input_data)
        st.sidebar.write(f"Predicted {target}: {prediction[0]:.2f}")

# Footer for credits and deployment details
st.markdown("""
---
**Deployed on:** Vultr Cloud Platform  
**Developed by:** AI Climate Tracker Team
""")
