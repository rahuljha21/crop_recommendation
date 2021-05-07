import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.ensemble import  RandomForestClassifier
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


st.write(""" #Crop Recommendation using Nutrient Level in Soil
 
This web app predicts the best crop based on soil nutrient """)
st.sidebar.header('User Input Parameter')


def user():
  N=st.sidebar.slider('Nitrogen',1,140)
  P=st.sidebar.slider('Phosphorus',0,145)
  K=st.sidebar.slider('Potassium',1,200)
  temperature=st.sidebar.slider('Temperature(celsuis)',9,45)
  humidity=st.sidebar.slider('Humidity',10,100)
  ph=st.sidebar.slider('Nitrogen',3.5,10.0)
  rainfall=st.sidebar.slider('Rainfall',20,300)
  data={'N':N,'P':P,'K':K,'temperature':temperature,'humidity':humidity,'ph':ph,'rainfall':rainfall}
  data=pd.DataFrame(data,index=[0])
  return data

data=user()
st.subheader('User Input Parameter')
st.write(data)
model=pd.read_csv("https://raw.githubusercontent.com/dphi-official/Datasets/master/crop_recommendation/train_set_label.csv")
x=model.drop('crop',axis=1)
y=model['crop']


encode=LabelEncoder()
encode.fit(y)
y=encode.transform(y)
models=RandomForestClassifier()
models.fit(x,y)

test=models.predict(data)

test=encode.inverse_transform(test)
st.subheader('The crop you should grow ')
st.write(test)
