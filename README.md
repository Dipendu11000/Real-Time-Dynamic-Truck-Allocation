Dynamic Truck Allocation Model_main file.ipynb  --> This file does the Dynamic Real Time Truck Allocation taking into consideration the various aspects trained over the available datasets(synthesized). The input 
parameters that it takes into consideration are 'Traffic', 'Local_Infrastructure', 'Night_Driving’, 'Customer_ID', Weather', 'Truck_Capacity', Time_Series_Fuel_Requirement_Prediction_Data (This time series prediction is done to get the predicted different types of fuel over the next week's required prediction. This prediction result is then given into the Dynamic Truck Allocation Model_main file as a parameter to allot the truck for that instance.
Edge cases taken into consideration - It might happen that a truck is allocated by our prediction model but that truck is already busy somewhere at that point of time. So, to avoid this, we have considered the real time Truck Availability Status (which in real life we will get from sensors, but here every time the code runs, it produces a differnet random pattern for the availability of each truck).
Also, each cluster is having their own set of unique Truck IDs , so our model will detect the CustomerID and then match it with the respective cluster and will shown the reccommandation for only the Trucks in their cluster , and not from any other cluster.
Data Generation IMS VTS TAS External Conditions.pynb  --> This file generates the datasets in a synchonous manner such that there are exactly 2000 customer IDs and 100 clusters, and each cluster will be having a certain amount of allocated trucks with their specified Truck ID. 







