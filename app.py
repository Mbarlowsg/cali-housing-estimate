import streamlit as st
import pandas as pd
import numpy as np
import pickle

df = pd.DataFrame({'flower': ['iris'],
                   'price': ['$12']})

st.header('California Housing Price Estimator')

st.write("Please enter the properties of the house below:")

income = st.number_input('Income - in block group')
house_age = st.number_input('House Age - in block group')
rooms = st.number_input('Number of Rooms')
bedrooms = st.number_input('Number of Bedrooms')
area_population = st.number_input('Population in area')
num_members = st.number_input('Number of people in home')
lat = st.number_input('Latitude of house')
long = st.number_input('Longitude of house')


X = np.array([[income, house_age, rooms, bedrooms, area_population, num_members, lat, long]])


with open('housing_model.pkl', 'rb') as f:
    model = pickle.load(f)


prediction = model.predict(X)

print(prediction)
print(type(prediction))


st.write(prediction)


