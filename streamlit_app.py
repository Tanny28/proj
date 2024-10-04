import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import nbformat

# Title of the app
st.title("AI Climate Prediction App")

# Load the Jupyter Notebook and extract the model and data
notebook_file = 'your_notebook.ipynb'  # Specify the path to your Jupyter Notebook here
notebook_content = nbformat.read(notebook_file, as_version=4)
code_cells = [cell['source'] for cell in notebook_content.cells if cell.cell_type == 'code']

# Execute all code cells to set up the environment
for code in code_cells:
    exec(code, globals())

# Check if the data variable and model are defined
if 'data' in globals() and 'model' in globals():
    st.success("Model and data loaded successfully!")

    # Display the dataset
    st.write("### Uploaded Climate Dataset")
    st.dataframe(data.head())

    # Select features for prediction
    st.sidebar.subheader("Select Features for Prediction")
    features = st.sidebar.multiselect("Choose features from the dataset", options=data.columns)

    # Select the target variable
    target = st.sidebar.selectbox("Select the Target Variable", options=data.columns)

    # Ensure both features and target are selected before proceeding
    if features and target:
        st.write(f"### Selected Features: {features}")
        st.write(f"### Target Variable: {target}")

        # Prepare data for prediction
        X = data[features]
        y = data[target]

        # Evaluate the model
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        st.write(f"### Root Mean Squared Error (RMSE) of the model: {rmse:.2f}")

        # Plot actual vs predicted
        st.write("### Actual vs Predicted Values")
        plt.figure(figsize=(10,5))
        plt.plot(y.values, label="Actual")
        plt.plot(y_pred, label="Predicted", linestyle='--')
        plt.legend()
        st.pyplot(plt.figure())

        # Input form for prediction based on live data
        st.sidebar.subheader("Predict Climate Indicator")
        user_inputs = {}
        for feature in features:
            user_inputs[feature] = st.sidebar.number_input(f"Input {feature}")

        # Predict when the user clicks the button
        if st.sidebar.button("Predict Climate Indicator"):
            input_data = pd.DataFrame([user_inputs])
            prediction = model.predict(input_data)
            st.sidebar.write(f"Predicted {target}: {prediction[0]:.2f}")

else:
    st.error("Failed to load model or data. Please check your Jupyter Notebook.")
