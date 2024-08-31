import streamlit as st
import pandas as pd
import numpy as np
st.title("🤖Machine Learning App")

st.info("Tis app generates the results and visualizations for the datasets")

with st.expander("Dataset working on"):
  st.write("**Raw Data**")
  data=pd.read_csv("https://raw.githubusercontent.com/prithvirajjadhav2266/MachineLearning/main/penguin_cleaned.csv")
  data

  st.write("** Independent Variables : X**")
  X=data.drop("species",axis=1)
  X

  st.write("**Dependent Variable : y**")
  y=data.species
  y
#the species is the target variable (Y)  and remaining variables are the input variables (X)

with st.expander("Data Visualizers"):
  #taking some of the variables that are important
  st.scatter_chart(data=data,x="bill_length_mm",y="body_mass_g",color="species")

#data preparations
with st.sidebar:
  st.header("Input Features")
  #island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex
  island=st.selectbox("Island",("Biscoe","Dream","Torgersen"))
  gender=st.selectbox("Gender",("Male","Female"))
  bill_length_mm=st.slider("Bill length (in mm)",32.1,59.6,43.9)
  bill_depth_mm=st.slider("Bill depth (in mm)",13.1,21,5,17,2)
  flipper_lwngth_mm=st.slider("Flipper length (in mm)",172.0,231.0,201.0)
  body_mass_g=st.slider("body mass (in g)",2700.0,6300.0,4207.0)












