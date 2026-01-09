# --- Deploy with Streamlit ---
import numpy as np
import pandas as pd
import pickle
import kagglehub
import os
import plotly.express as px
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# load model 
with open('treemodel.pkl', 'rb') as file:
    treemodel = pickle.load(file)

# load encoders
with open('shape_le.pkl', 'rb') as file:
    shape_le = pickle.load(file)
with open('color_le.pkl', 'rb') as file:
    color_le = pickle.load(file)
with open('taste_le.pkl', 'rb') as file:
    taste_le = pickle.load(file)
with open('target_le.pkl', 'rb') as file:
    target_le = pickle.load(file)

path = kagglehub.dataset_download("pranavkapratwar/fruit-classification")
for file in os.listdir(path):
    if file.endswith(".csv"):
        dataset_path = os.path.join(path, file)
        break
print("Using dataset file:", dataset_path)
df = pd.read_csv(dataset_path)

target_column = 'fruit_name'
X = df.drop(target_column, axis=1)
Y = df[target_column]

# streamlit run fruitclass.py
import streamlit as st

# Page configuration
st.set_page_config(layout="wide")

# trained-model - treemodel
# feature importance
importances_df = pd.DataFrame({
    'Variables': X.columns,
    'Feature Importance Score': treemodel.feature_importances_
})

# Sidebar setup
image_sidebar = Image.open('apple.png')
st.sidebar.image(image_sidebar, use_container_width=True)
st.sidebar.header('Fruit Features')

# Feature selection on sidebar
def get_user_input():
    shape = st.sidebar.selectbox('Shape', shape_le.classes_)
    color = st.sidebar.selectbox('Color', color_le.classes_)
    taste = st.sidebar.selectbox('Taste', taste_le.classes_)
    size = st.sidebar.number_input('Size (cm)', min_value=0.0, max_value=30.0, step=0.1, value=10.0)
    weight = st.sidebar.number_input('Weight (g)', min_value=0.0, max_value=3300.0, step=0.1, value=1000.0)
    price = st.sidebar.number_input('Average Price (RM)', min_value=0.0, max_value=8.0, step=0.1, value=5.0)

    user_data = {
        'shape': shape_le.transform([shape])[0],
        'color': color_le.transform([color])[0],
        'taste': taste_le.transform([taste])[0],
        'size (cm)': size,
        'weight (g)': weight,
        'avg_price (₹)': price * 22.15
    }
    
    return user_data

# Big one in the middle
image_banner = Image.open('apple.png')
st.image(image_banner, use_container_width=True)

# Centered title
st.markdown("<h1 style='text-align: center;'>Fruit Price Prediction App</h1>", unsafe_allow_html=True)

# Split layout into two columns
left_col, right_col = st.columns(2)

# Left column: Feature Importance Interactive Bar Chart
with left_col:
    st.header("Feature Importance")
    
    # Sort feature importance DataFrame by 'Feature Importance Score'
    importances_sorted = importances_df.sort_values(by='Feature Importance Score', ascending=True)
    
    # Create interactive bar chart with Plotly
    fig = px.bar(
        importances_sorted,
        x='Feature Importance Score',
        y='Variables',
        orientation='h',
        title="Feature Importance",
        labels={'Feature Importance Score': 'Importance', 'Variable': 'Feature'},
        text='Feature Importance Score',
        color_discrete_sequence=['#48a3b4']
    )
    fig.update_layout(
        xaxis_title="Feature Importance Score",
        yaxis_title="Variable",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Right column: Prediction Interface
with right_col:
    st.header("Predict Fruit Name")
    
    # User inputs from sidebar
    user_data = get_user_input()

    # Transform the input into the required format
    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        return np.array([list(input_data.values())])

    # Feature list (same order as used during model training)
    features = X.columns.tolist()

    # Predict button
    if st.button("Predict"):
        input_array = prepare_input(user_data, features)
        prediction = treemodel.predict(input_array)
        st.subheader("Predicted Fruit Name")
        pred_fruit = target_le.inverse_transform(prediction)[0]
        st.write(pred_fruit)