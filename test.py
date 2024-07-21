import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt
import pickle
import xgboost as xgb

@st.cache_data
def load_data(uploaded_file, sheet_name, header):
    try:
        data = pd.read_excel(uploaded_file, header=header, sheet_name=sheet_name, engine='openpyxl')
        data.columns = data.columns.str.strip()
        for col in data.columns:
            if data[col].dtype == 'O':
                data[col] = data[col].str.strip()
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model(uploaded_file):
    try:
        model = pd.read_pickle(uploaded_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def concatenate_dataframes(df1, df2):
    return pd.concat([df1, df2.set_index(df1.index)], axis=1)

def dataframe_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.00'})
        worksheet.set_column('A:A', None, format1)
    processed_data = output.getvalue()
    return processed_data

def train_model(data, features_to_drop, target):
    eta = 0.08
    lambda_param = 5
    X = data.drop(features_to_drop, axis=1)
    y = data[target]
    model = XGBRegressor(booster='gblinear', eta=eta, reg_lambda=lambda_param)
    model.fit(X, y)
    predictions = model.predict(X)

    st.download_button("Download Model", data=pickle.dumps(model), file_name="model.pkl")
    return X, y, predictions

def display_results():
    X, y, predictions = train_model(sub_data, features_to_drop, target)
    subset = sub_data.drop(features_to_drop, axis=1)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.scatter(y, predictions)
    st.pyplot(fig)
    
    st.write("Mean Absolute Percentage Error")
    st.write(mt.mean_absolute_percentage_error(y, predictions))
    st.write("R-squared")
    st.write(mt.r2_score(y, predictions))
    
    results_df = pd.DataFrame({'Actual': y, 'Predicted': predictions})
    combined_df = concatenate_dataframes(subset, results_df)
    st.dataframe(combined_df)
    
    excel_data = dataframe_to_excel(combined_df)
    st.download_button(label='Download Data', data=excel_data, file_name='predictions.xlsx')

st.set_page_config(page_title='Predictive Model for Cement Compression Strength', layout="wide")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Data', 'Data Description', 'Graphs', 'Train Model', 'Apply Model'])

st.sidebar.write("****Upload Excel Data File****")
uploaded_file = st.sidebar.file_uploader("Upload File Here")

if uploaded_file:
    sheet_name = st.sidebar.selectbox("Which sheet contains the data?", pd.ExcelFile(uploaded_file).sheet_names)
    header_row = st.sidebar.number_input("Which row contains the column names?", min_value=0, max_value=100)

    data = load_data(uploaded_file, sheet_name, header_row)
    
    if data is not None:
        with tab1:
            st.write('### 1. Loaded Data')
            st.dataframe(data, use_container_width=True)

        with tab2:
            st.write('### 2. Data Description')
            description_option = st.radio("Select what you want to see from the data", 
                                          ["Dimensions", "Variable Descriptions", "Descriptive Statistics", "Column Value Counts"])
            
            if description_option == 'Variable Descriptions':
                field_descriptions = data.dtypes.reset_index().rename(columns={'index': 'Field Name', 0: 'Field Type'})
                st.dataframe(field_descriptions, use_container_width=True)
            
            elif description_option == 'Descriptive Statistics':
                descriptive_stats = data.describe(include='all').round(2).fillna('')
                st.dataframe(descriptive_stats, use_container_width=True)
            
            elif description_option == 'Column Value Counts':
                column_to_investigate = st.selectbox("Select Column to Investigate", data.select_dtypes('object').columns)
                value_counts = data[column_to_investigate].value_counts().reset_index().rename(columns={'index': 'Value', column_to_investigate: 'Count'})
                st.dataframe(value_counts, use_container_width=True)
            
            else:
                st.write('###### Data Dimensions:', data.shape)

        with tab3:
            selected_mill = st.selectbox("Select Mill", data['Mill'].unique())
            cement_type = st.selectbox("Select Cement Type", data['Cement Type'].unique())
            graph_type = st.selectbox("Select Graph Type", ['Boxplot', 'Histogram', 'Trend'])
            filtered_data = data[(data['Cement Type'] == cement_type) & (data['Mill'] == selected_mill)]
            
            st.write('### 3. Graphical Exploration')
            if graph_type == "Boxplot":
                fig, axs = plt.subplots(2, 2)
                fig.set_size_inches(10, 6)
                axs[0, 0].boxplot(filtered_data['R1D'])
                axs[0, 0].set_title("1 Day")
                axs[0, 1].boxplot(filtered_data['R3D'])
                axs[0, 1].set_title("3 Days")
                axs[1, 0].boxplot(filtered_data['R7D'])
                axs[1, 0].set_title("7 Days")
                axs[1, 1].boxplot(filtered_data['R28D'])
                axs[1, 1].set_title("28 Days")
                st.pyplot(fig)
            elif graph_type == "Histogram":
                fig, axs = plt.subplots(2, 2)
                fig.set_size_inches(10, 6)
                axs[0, 0].hist(filtered_data['R1D'])
                axs[0, 0].set_title("1 Day")
                axs[0, 1].hist(filtered_data['R3D'])
                axs[0, 1].set_title("3 Days")
                axs[1, 0].hist(filtered_data['R7D'])
                axs[1, 0].set_title("7 Days")
                axs[1, 1].hist(filtered_data['R28D'])
                axs[1, 1].set_title("28 Days")
                st.pyplot(fig)
            elif graph_type == "Trend":
                fig, axs = plt.subplots(2, 2)
                fig.set_size_inches(10, 8)
                axs[0, 0].plot(filtered_data['Date'], filtered_data['R1D'])
                axs[0, 0].set_title("1 Day")
                axs[0, 0].tick_params(axis='x', labelrotation=30, labelsize=8)
                axs[0, 1].plot(filtered_data['Date'], filtered_data['R3D'])
                axs[0, 1].set_title("3 Days")
                axs[0, 1].tick_params(axis='x', labelrotation=30, labelsize=8)
                axs[1, 0].plot(filtered_data['Date'], filtered_data['R7D'])
                axs[1, 0].set_title("7 Days")
                axs[1, 0].tick_params(axis='x', labelrotation=30, labelsize=8)
                axs[1, 1].plot(filtered_data['Date'], filtered_data['R28D'])
                axs[1, 1].set_title("28 Days")
                axs[1, 1].tick_params(axis='x', labelrotation=30, labelsize=8)
                st.pyplot(fig)

        with tab4:
            selected_mill_model = st.selectbox("Select Mill to Model", data['Mill'].unique())
            selected_cement_type_model = st.selectbox("Select Cement Type to Model", data['Cement Type'].unique())
            prediction_age = st.selectbox("Age to Predict", ["1 Day", "3 Days", "7 Days", "28 Days"])
            
            sub_data = data[(data['Cement Type'] == selected_cement_type_model) & (data['Mill'] == selected_mill_model)]
            
            if prediction_age == "1 Day":
                features_to_drop = ['Date', 'Cement Type', 'Mill', 'R1D', 'R3D', 'R7D', 'R28D']
                target = 'R1D'
                display_results()
                
            elif prediction_age == "3 Days":
                features_to_drop = ['Date', 'Cement Type', 'Mill', 'R3D', 'R7D', 'R28D']
                target = 'R3D'
                display_results()
                
            elif prediction_age == "7 Days":
                features_to_drop = ['Date', 'Cement Type', 'Mill', 'R7D', 'R28D']
                target = 'R7D'
                display_results()
                
            elif prediction_age == "28 Days":
                features_to_drop = ['Date', 'Cement Type', 'Mill', 'R28D']
                target = 'R28D'
                display_results()

        with tab5:
            model_file = st.file_uploader("Upload Model")
            if model_file:
                loaded_model = load_model(model_file)
                if loaded_model:
                    st.write("Model loaded successfully")
                    prediction_data_file = st.file_uploader("Upload Production Data")
                    if prediction_data_file:
                        prediction_data = load_data(prediction_data_file, 'Sheet1', 0)
                        if prediction_data is not None:
                            st.write("Predicting...")
                            y_pred = loaded_model.get_booster().predict(xgb.DMatrix(prediction_data))
                            prediction_results = pd.DataFrame({'Predictions': y_pred})
                            combined_results = concatenate_dataframes(prediction_data, prediction_results)
                            st.dataframe(combined_results)
                            
                            excel_data = dataframe_to_excel(combined_results)
                            st.download_button(label='Download Results', data=excel_data, file_name='results.xlsx')
