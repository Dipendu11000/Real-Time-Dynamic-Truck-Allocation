#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random

def generate_truck_availability(truck_numbers):
    return {truck: random.choice([0, 1]) for truck in truck_numbers}
def load_input_data_from_csv(file_path):
    input_data = pd.read_csv(file_path)
    return input_data.to_dict(orient='records')[0]

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


# In[ ]:





# In[2]:


clusters = data2['Cluster_ID'].unique()
models = {}
for cluster in clusters:
    cluster_data = data2[data2['Cluster_ID'] == cluster]
    if len(cluster_data) == 0:  
        continue
    cluster_data = cluster_data[relevant_columns]  
    label_encoders = {}
    for column in ['Weather', 'Traffic', 'Local_Infrastructure', 'Night_Driving']:    
        label_encoders[column] = LabelEncoder()
        cluster_data[column] = label_encoders[column].fit_transform(cluster_data[column])
    X = cluster_data.drop(columns=['Truck_Number'])
    y = cluster_data['Truck_Number']
    model = RandomForestClassifier()
    model.fit(X, y)
    models[cluster] = model

def predict_truck_allocation(input_data, cluster_id):
    input_df = pd.DataFrame([input_data])
    for column in ['Weather', 'Traffic', 'Local_Infrastructure', 'Night_Driving']:    
        input_df[column] = label_encoders[column].transform([input_data[column]])   
    model = models.get(cluster_id)
    if model is None:
        return "No Truck is Currently Available"    
    predicted_truck = model.predict(input_df.drop(columns=['Customer_ID']))
    return predicted_truck[0]

def find_cluster_id(input_data):
    customer_id = input_data['Customer_ID']
    customer_data = data[data['Customer_ID'] == customer_id]
    if len(customer_data) == 0:
        print("Customer ID not found in the dataset")
        return None
    cluster_id = customer_data.iloc[0]['Cluster_ID']
    return cluster_id


# In[6]:


# Streamlit App
def main():
    st.title('Truck Allocation Predictor')
    st.sidebar.header('User Input')
    input_data = load_input_data_from_csv("Final_Fuel_Requirement_Predictions.csv")
    st.write("Input Data from CSV:", input_data) 
    try:
        truck_capacity = st.sidebar.number_input('Truck Capacity', value=input_data['Truck_Capacity'])
        weather = st.sidebar.selectbox('Weather', ['Sunny', 'Rainy', 'Drizzle'], index=input_data['Weather'])
        traffic = st.sidebar.slider('Traffic', min_value=0, max_value=5, value=input_data['Traffic'])
        local_infrastructure = st.sidebar.selectbox('Local Infrastructure', ['Good', 'Average', 'Poor'], index=input_data['Local_Infrastructure'])
        night_driving = st.sidebar.selectbox('Night Driving', ['Yes', 'No'], index=input_data['Night_Driving'])
        customer_id = input_data['Customer_ID']
        input_data = {
            'Truck_Capacity': truck_capacity,
            'Weather': weather,
            'Traffic': traffic,
            'Local_Infrastructure': local_infrastructure,
            'Night_Driving': night_driving,
            'Customer_ID': customer_id
        }
        if st.sidebar.button('Predict'):
            cluster_id = find_cluster_id(input_data)
            if cluster_id is not None:
                predicted_truck = predict_truck_allocation(input_data, cluster_id)
                st.write(f"Allocated Truck Number for Customer_ID '{input_data['Customer_ID']}' is '{predicted_truck}'")
    except KeyError as e:
        st.error(f"KeyError: {e}. Please check if the keys in the CSV file match the keys used in the code.")
if __name__ == '__main__':
    main()


# In[ ]:




