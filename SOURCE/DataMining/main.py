import streamlit as st
import pandas as pd
import pickle
import base64
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree
import networkx as nx

# Read dataset
df = pd.read_csv("data/Global_Superstore.csv")

# Load Model Random Forest (Best performance)
rf = pickle.load(open('model/RandomForest.sav', 'rb'))


# Load dictionaries
def load_score_dict(file_path, index_col, score_col):
    df = pd.read_csv(file_path)
    return df.set_index(index_col)[score_col].to_dict()

city_score_dict = load_score_dict("data/city_score.csv", "City", "City_Score")
state_score_dict = load_score_dict("data/state_score.csv", "State", "State_Score")
country_score_dict = load_score_dict("data/country_score.csv", "Country", "Country_Score")


# Function to preprocess features
def preprocess_features(features_df):
    # Copy features to avoid modifying the original DataFrame
    features = features_df.copy()

    # Convert other categorical columns to numerical scores
    features['City_Score'] = features['City'].map(city_score_dict).fillna(0)
    features['Country_Score'] = features['Country'].map(country_score_dict).fillna(0)
    features['State_Score'] = features['State'].map(state_score_dict).fillna(0)
    
    # Drop original categorical columns
    features.drop(['City', 'Country', 'State'], axis=1, inplace=True)

    return features

# Function to classify movie success
def classify_order(features, model):
    # Make prediction
    prediction = model.predict(features)

    if prediction[0] == 1:
        return 'The order is profitable'
    else:
        return 'The order is unprofitable'
    
def get_score_category(score):
    if score == 4:
        return 'Very High Profit'
    elif score == 3:
        return 'High Profit'
    elif score == 2:
        return 'Average Profit'
    elif score == 1:
        return 'Low Profit'
    elif score == 0:
        return 'Loss Profit'
    else:
        return 'Unknown'

# Streamlit UI
title_html = f"""<h1 style="font-size: 40px; font-weight: bold; color: #B85042; font-family: 'Times New Roman', serif; text-shadow: 2px 2px 4px #000000;">Predict order profit classification</h1>"""

st.markdown(title_html, unsafe_allow_html=True)
# Custom CSS to style the button
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

Discount = st.number_input('Discount', format="%f", help="Enter the discount of order", placeholder="Enter the discount of the order")
City = st.text_input('City', help="Enter the city name", placeholder="Enter the city name of the order")
State = st.text_input('State', help="Enter the state name", placeholder="Enter the state name of the order")
Country = st.text_input('Country', help="Enter the country name", placeholder="Enter the country name of the order")

features = {
    'Discount': Discount,
    'City': City,
    'Country': Country,
    'State': State
}
       
features_df = pd.DataFrame([features])


if st.button('Get Result'):
    st.subheader('User Input The Order Information !')
    st.write(features_df)

    # Preprocess features
    features_preprocessed = preprocess_features(features_df)

    # Perform method classification
    rf_prediction = classify_order(features_preprocessed, rf)
    st.write('Profit forecast results:', rf_prediction)

    # Đánh giá các thành phố, bang, quốc gia
    city_score_category = get_score_category(features_preprocessed["City_Score"].iloc[0])
    state_score_category = get_score_category(features_preprocessed["State_Score"].iloc[0])
    country_score_category = get_score_category(features_preprocessed["Country_Score"].iloc[0])

    # Hiển thị thông báo về ý nghĩa của điểm số
    st.write(f'City {features_df["City"].iloc[0]} has a {city_score_category}')
    st.write(f'State {features_df["State"].iloc[0]} has a {state_score_category}')
    st.write(f'Country {features_df["Country"].iloc[0]} has a {country_score_category}')
    
# File upload

title1_html = f"""<h1 style="font-size: 30px; font-weight: bold; color: #FF0000; font-family: 'Times New Roman', serif ">Upload CSV for Batch Prediction</h1>"""

st.markdown(title1_html, unsafe_allow_html=True)
uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Ensure the input data has all the necessary columns
    required_columns = ['Discount', 'City', 'State', 'Country']
    if not all(col in data.columns for col in required_columns):
        st.error(f"Uploaded CSV must contain the following columns: {', '.join(required_columns)}")
        raise ValueError("Uploaded CSV is missing required columns.")
    else:
        # Preprocess the data
        preprocessed_data = preprocess_features(data)
        
        # Predict using the loaded models
        rf_prediction = rf.predict(preprocessed_data)
        
        # Create a DataFrame for the output
        output_data = preprocessed_data.copy()
        output_data['Predicted_Label_RF'] = ['Profitable' if pred == 1 else 'Unprofitable' for pred in rf_prediction]
        
        # Convert DataFrame to CSV
        output_csv = output_data.to_csv(index=False)
        
        # Create a download button for the output CSV
        st.download_button(
            label="Download Predictions",
            data=output_csv,
            file_name='ModelPredictions.csv',
            mime='text/csv'
        )