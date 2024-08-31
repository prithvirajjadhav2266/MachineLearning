import streamlit as st
import pandas as pd
import numpy as np
st.title("🤖Machine Learning App")

st.info("Tis app generates the results and visualizations for the datasets")

data=pd.read_csv("https://github.com/prithvirajjadhav2266/MachineLearning/blob/main/penguin_cleaned.csv")

data
