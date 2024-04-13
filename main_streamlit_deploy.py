#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


import streamlit as st
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Function to generate truck availability status
def generate_truck_availability(truck_numbers):
    return {truck: random.choice([0, 1]) for truck in truck_numbers}

# Function to load input data from CSV
def load_input_data_from_csv(file_path):
    input_data = pd.read_csv(file_path)
    return input_data.to_dict(orient='records')[0]

# Loading the model and label encoders
data = pd.read_csv("Combined_Dataset.csv")
data2 = data.iloc[:18000].copy()
truck_numbers = data2['Truck_Number'].unique()
truck_availability_status = generate_truck_availability(truck_numbers)
data2['Truck_Availability'] = data2['Truck_Number'].map(truck_availability_status)
data2 = data2[data2['Truck_Availability'] == 1].copy()

relevant_columns = ['HSD_Requirement', 'LDO_Requirement', 'FO_Requirement', 
                    'LSHS_Requirement', 'SKO_Requirement', 'MS_Requirement', 
                    'Truck_Capacity', 'Weather', 'Traffic', 'Local_Infrastructure', 
                    'Night_Driving', 'Truck_Number']
truck_availability_df = pd.DataFrame(truck_availability_status.items(), columns=['Truck_Number', 'Truck_Availability'])

clusters = data2['Cluster_ID'].unique()
models = {}
label_encoders = {}  # Initialize label encoders dictionary

for cluster in clusters:
    cluster_data = data2[data2['Cluster_ID'] == cluster]
    if len(cluster_data) == 0:  
        continue
    cluster_data = cluster_data[relevant_columns]  
    
    # Initialize label encoder for each cluster
    label_encoders[cluster] = {}
    for column in ['Weather', 'Traffic', 'Local_Infrastructure', 'Night_Driving']:    
        label_encoders[cluster][column] = LabelEncoder()
        cluster_data[column] = label_encoders[cluster][column].fit_transform(cluster_data[column])
    
    X = cluster_data.drop(columns=['Truck_Number'])
    y = cluster_data['Truck_Number']
    model = RandomForestClassifier()
    model.fit(X, y)
    models[cluster] = model

# Function to predict truck allocation
def predict_truck_allocation(input_data, cluster_id):
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical features using appropriate label encoder
    for column in ['Weather', 'Traffic', 'Local_Infrastructure', 'Night_Driving']:
        input_df[column] = label_encoders[cluster_id][column].transform([input_data[column]])   
        
    model = models.get(cluster_id)
    if model is None:
        return "No Truck is Currently Available"    
    
    # Ensure input data has the same columns as used during model training
    input_df = input_df.reindex(columns=relevant_columns[:-1], fill_value=0)  
    
    predicted_truck = model.predict(input_df)
    return predicted_truck[0]

# Function to find cluster ID
def find_cluster_id(input_data):
    customer_id = input_data['Customer_ID']
    customer_data = data[data['Customer_ID'] == customer_id]
    if len(customer_data) == 0:
        print("Customer ID not found in the dataset")
        return None
    cluster_id = customer_data.iloc[0]['Cluster_ID']
    return cluster_id

# Streamlit App
def main():
    st.title('Truck Allocation Predictor')

    # Input form for user input
    st.sidebar.header('User Input')
    truck_capacity = st.sidebar.number_input('Truck Capacity')
    weather = st.sidebar.selectbox('Weather', ['Sunny', 'Rainy', 'Drizzle'])
    traffic = st.sidebar.slider('Traffic', min_value=0, max_value=5)
    local_infrastructure = st.sidebar.selectbox('Local Infrastructure', ['Good', 'Average', 'Poor'])
    night_driving = st.sidebar.selectbox('Night Driving', ['Yes', 'No'])
    customer_id = st.sidebar.text_input('Customer ID')

    # Load input data from Final_Fuel_Requirement_Predictions.csv
    input_data = load_input_data_from_csv("Final_Fuel_Requirement_Predictions.csv")

    if st.sidebar.button('Allocate Truck'):
        # Update input data with user input
        input_data.update({
            'Truck_Capacity': truck_capacity,
            'Weather': weather,
            'Traffic': traffic,
            'Local_Infrastructure': local_infrastructure,
            'Night_Driving': night_driving,
            'Customer_ID': customer_id
        })
        cluster_id = find_cluster_id(input_data)
        if cluster_id is not None:
            predicted_truck = predict_truck_allocation(input_data, cluster_id)
            st.success(f"Allocated Truck Number for Customer_ID '{input_data['Customer_ID']}' is '{predicted_truck}'")

if __name__ == '__main__':
    main()


# In[ ]:




