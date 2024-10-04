import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import nbformat

# Title of the app
st.title("AI Climate Prediction App")

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload your climate data (CSV or Jupyter Notebook)", type=["csv", "ipynb"])

# Initialize data variable
data = None

# Check if the user has uploaded a dataset
if uploaded_file is not None:
    st.success("File uploaded successfully!")

    # Determine the file type and load the data
    if uploaded_file.type == "text/csv":
        try:
            # Load the CSV data into a dataframe
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

    elif uploaded_file.type == "application/x-ipynb+json":
        try:
            # Load the Jupyter Notebook and extract the data
            notebook_content = nbformat.read(uploaded_file, as_version=4)
            # Look for code cells in the notebook
            code_cells = [cell['source'] for cell in notebook_content.cells if cell.cell_type == 'code']
            
            # Assume the last code cell has the dataframe creation statement
            if code_cells:
                # Execute the last cell to get the dataframe (this is a simplistic approach)
                exec(code_cells[-1], globals())
                data = globals().get('data')  # Replace 'data' with your actual dataframe variable name
                
                if data is None:
                    st.error("The last code cell did not define a variable named 'data'. Please check your notebook.")
                else:
                    st.success("Data loaded from Jupyter Notebook!")
            else:
                st.error("No code cells found in the Jupyter Notebook.")

        except Exception as e:
            st.error(f"Error reading Jupyter Notebook: {e}")

    if data is not None:
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

            # Split the data into train and test sets
            X = data[features]
            y = data[target]

            # Check if y is a single column and convert to 1D array if needed
            if len(y.shape) > 1:
                y = y.values.ravel()  # Convert to 1D array
            
            # Ensure that the target variable is numeric
            if not np.issubdtype(y.dtype, np.number):
                st.error("The target variable must be numeric.")
            else:
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
                plt.plot(y_test, label="Actual")
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
    st.warning("Please upload a CSV or Jupyter Notebook dataset to start")
