import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from io import BytesIO

# Title of the Streamlit app
st.title('Machine Learning Model Deployment with XGBoost')

# File uploader to upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file", type="xlsx")

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Predictions')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

if uploaded_file is not None:
    # Load the data from Excel file
    df = pd.read_excel(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Assuming the last column is the target and the rest are features
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train an XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and display metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write("Model Quality Metrics:")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

    # Create a DataFrame with the actual and predicted values
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.write("Predicted vs Actual:")
    st.write(results_df)

    # Provide download link for the predictions
    excel_data = to_excel(results_df)
    st.download_button(label='Download Predictions as Excel', data=excel_data, file_name='predictions.xlsx')
else:
    st.write("Please upload an Excel file.")
