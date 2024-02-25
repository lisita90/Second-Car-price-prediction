import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load the data
data = pd.read_csv('cleaned car.csv')

# Perform one-hot encoding for categorical variables
data_encoded = pd.get_dummies(data, columns=['fuel_type', 'company', 'name'])

# Split the data into features and target variable
X = data_encoded.drop('Price', axis=1)
y = data_encoded['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app
st.title('Car Price Prediction')

# Dropdown menu for selecting company
company = st.selectbox('Company', data['company'].unique())

# Filter names based on selected company
filtered_names = data[data['company'] == company]['name'].unique()
name = st.selectbox('Name', filtered_names)

year = st.number_input('Year', min_value=2000, max_value=2025, value=2010)
kms_driven = st.number_input('Kilometers Driven', min_value=0, value=0)
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])

# Add a submit button
submit_button = st.button('Submit')

if submit_button:
    # Encode user input
    user_input = {
        'company': [company],
        'name': [name],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    }
    user_input_df = pd.DataFrame(user_input)

    # Perform one-hot encoding for user input
    user_input_encoded = pd.get_dummies(user_input_df, columns=['fuel_type', 'company', 'name'])

    # Ensure the columns in user_input_encoded match the columns used for training
    missing_cols = set(X_train.columns) - set(user_input_encoded.columns)
    for col in missing_cols:
        user_input_encoded[col] = 0

    # Reorder the columns to match the order used during training
    user_input_encoded = user_input_encoded[X_train.columns]

    # Make prediction
    prediction = model.predict(user_input_encoded)

    # Display prediction in dollars
    st.subheader('Predicted Price')
    st.write('$', round(prediction[0] / 82, 2))
