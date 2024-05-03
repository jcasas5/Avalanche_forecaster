import streamlit as st
import tensorflow as tf
import numpy as np
import joblib  # Import joblib for loading scikit-learn models


# Function to load the model
@st.cache_data
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def load_model2(model_path):
    model = joblib.load(model_path)
    return model

# Load your trained model
model = load_model('final_weights.h5')


# Streamlit app title
st.title('Avalanche Probability Prediction at Davos Ski Resort')

# Input fields for weather variables
temp = st.number_input('Temperature (°C)', value=0.0)
humidity = st.number_input('Humidity (%)', value=0.0)
precip = st.number_input('Precipitation (mm)', value=0.0)
snow = st.number_input('Snowfall (cm)', value=0.0)
snowdepth = st.number_input('Snow Depth (cm)', value=0.0)
windspeed = st.number_input('Wind Speed (km/h)', value=0.0)
# For precipitation type, consider using select boxes for ease of use
preciptype = st.selectbox(
    'Precipitation Type',
    ('None', 'Rain', 'Rain,Snow', 'Snow')
)

# Encode precipitation type into binary features as expected by the model
preciptype_rain = 1 if preciptype == 'Rain' else 0
preciptype_rain_snow = 1 if preciptype == 'Rain,Snow' else 0
preciptype_snow = 1 if preciptype == 'Snow' else 0

# Button to predict avalanche probability
if st.button('Predict Probability'):
    # Organize the input data as expected by the model
    input_data = np.array([[temp, humidity, precip, snow, snowdepth, windspeed,
                            preciptype_rain, preciptype_rain_snow, preciptype_snow]])

    # Predicting the probability of avalanche
    probability = model.predict(input_data)

    # Displaying the prediction
    st.write(f'Probability of Avalanche: {probability[0][0]*100:.2f}%')


