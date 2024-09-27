import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# Title and Description
st.title("AI-powered Climate Change Tracker")
st.write("""
This app tracks climate change by analyzing and visualizing global climate data.
It also uses predictive models to forecast future climate trends.
""")

# Upload Dataset
st.sidebar.header("Upload your Climate Data CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load Data
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Overview")
    st.write(data.head())

    # Select Columns for Visualization
    st.sidebar.header("Visualization Settings")
    x_axis = st.sidebar.selectbox('Select X-axis:', data.columns)
    y_axis = st.sidebar.selectbox('Select Y-axis:', data.columns)

    # Data Visualization
    st.subheader("Climate Data Visualization")
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Matrix")
    corr = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # Predictive Model
    st.subheader("Predict Future Trends with Linear Regression")
    selected_feature = st.selectbox('Select feature for prediction (X-axis):', data.columns)
    if selected_feature:
        # Preparing Data for Linear Regression
        X = np.array(data[selected_feature].dropna()).reshape(-1, 1)
        y = np.array(data[y_axis].dropna())

        if len(X) > 0 and len(y) > 0:
            # Train Linear Regression Model
            model = LinearRegression()
            model.fit(X, y)

            # Predict Future Data (e.g., next 10 years)
            future_years = np.array(list(range(int(max(X)) + 1, int(max(X)) + 11))).reshape(-1, 1)
            future_preds = model.predict(future_years)

            # Plot Prediction
            st.write(f"Predicted values for {y_axis} based on {selected_feature}:")
            fig, ax = plt.subplots()
            sns.lineplot(x=X.flatten(), y=y, label='Historical Data', ax=ax)
            sns.lineplot(x=future_years.flatten(), y=future_preds, label='Predicted Data', ax=ax)
            st.pyplot(fig)
else:
    st.write("Please upload a dataset to proceed.")

# Conclusion
st.sidebar.header("Conclusion")
st.sidebar.write("This climate tracker provides insights into climate trends and predictions for the future.")

st.write("### Key Takeaways:")
st.write("""
- Visualize trends in global climate data.
- Predict future changes using linear regression models.
- Get insights into the correlation between different climate variables.
""")
st.write("Made for the Hackethon Purpose ")
