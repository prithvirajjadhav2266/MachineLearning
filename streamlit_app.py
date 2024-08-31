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
