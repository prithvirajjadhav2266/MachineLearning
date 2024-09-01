import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("🤖 Machine Learning App")

st.info("This app generates results and visualizations for datasets")

# Load and display dataset
with st.expander("Dataset Working On"):
    st.write("**Raw Data**")
    data = pd.read_csv("https://raw.githubusercontent.com/prithvirajjadhav2266/MachineLearning/main/penguin_cleaned.csv")
    st.write(data)

    st.write("**Independent Variables : X**")
    X_raw = data.drop("species", axis=1)
    st.write(X_raw)

    st.write("**Dependent Variable : y**")
    y_raw = data['species']
    st.write(y_raw)

# Data visualizations using Streamlit's built-in visualizers
with st.expander("Data Visualizers"):
    # Scatter Plot for Bill Length vs. Body Mass
    st.write("### Scatter Plot: Bill Length vs. Body Mass")
    st.plotly_chart(
        {
            "data": [
                {
                    "x": data["bill_length_mm"],
                    "y": data["body_mass_g"],
                    "type": "scatter",
                    "mode": "markers",
                    "marker": {"color": data["species"].apply(lambda x: {'Adelie': 'blue', 'Chinstrap': 'red', 'Gentoo': 'green'}[x])},
                    "text": data["species"],
                }
            ],
            "layout": {"title": "Bill Length vs Body Mass Colored by Species"},
        }
    )

    # Line Chart of Body Mass by Species
    st.write("### Line Chart: Body Mass by Species")
    body_mass_by_species = data.groupby("species")["body_mass_g"].mean()
    st.line_chart(body_mass_by_species)

    # Bar Chart for Species Distribution
    st.write("### Bar Chart: Species Distribution")
    species_distribution = data['species'].value_counts()
    st.bar_chart(species_distribution)

    # Area Chart for Flipper Length by Species
    st.write("### Area Chart: Flipper Length by Species")
    flipper_length_by_species = data.groupby("species")["flipper_length_mm"].mean()
    st.area_chart(flipper_length_by_species)

# Data preparations
with st.sidebar:
    st.header("Input Features")
    island = st.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
    gender = st.selectbox("Gender", ("Male", "Female"))
    bill_length_mm = st.slider("Bill length (in mm)", 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider("Bill depth (in mm)", 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider("Flipper length (in mm)", 172.0, 231.0, 201.0)
    body_mass_g = st.slider("Body mass (in g)", 2700.0, 6300.0, 4207.0)

# Creating the input features DataFrame 
values = {
    "island": island,
    "bill_length_mm": bill_length_mm,
    "bill_depth_mm": bill_depth_mm,
    "flipper_length_mm": flipper_length_mm,
    "body_mass_g": body_mass_g,
    "sex": gender
}

input_data = pd.DataFrame(values, index=[0])
input_penguins = pd.concat([input_data, X_raw], axis=0)

with st.expander("Input Features"):
    st.write("**Input Features**")
    st.write(input_data)

    st.write("**Combined Dataset**")
    st.write(input_penguins)

# Data encoding
encode = ["island", "sex"]
data_penguins = pd.get_dummies(input_penguins, columns=encode)

X = data_penguins[1:]  # Features for model training
input_row = data_penguins[:1]  # Input row for prediction

# Encoding target variable
target_mapper = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
y = y_raw.apply(lambda x: target_mapper[x])

with st.expander("Data Preparations"):
    st.write("**Encoded Input Penguin (X)**")
    st.write(input_row)

    st.write("**Encoded y**")
    st.write(y)

# Model training
model = RandomForestClassifier()
model.fit(X, y)

# Apply model to make predictions
prediction = model.predict(input_row)
prediction_probability = model.predict_proba(input_row)

data_prediction_probability = pd.DataFrame(prediction_probability, columns=["Adelie", "Chinstrap", "Gentoo"])

# Display the prediction species probabilities
st.subheader("Predicted Species Probabilities")
st.dataframe(data_prediction_probability, hide_index=True)

# Display the predicted species
penguin_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
st.success(f"Predicted Species: {penguin_species[prediction][0]}")
