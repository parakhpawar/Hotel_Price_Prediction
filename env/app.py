

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the hotel dataset
hotelDataset = pd.read_excel("hotels.xlsx")

# Create a Flask app
app = Flask(__name__)

# Define a route for the homepage
@app.route('/')
def home():
    return "Welcome to the Hotel Price Predictor!"

# Define a route for the hotel price predictor
@app.route('/predict_price')
def predict_price():
    # Get the hotel name from the request parameters
    hotel_name = request.args.get('hotel_name')
    
    # Get the row for the specified hotel name
    hotel_data = hotelDataset[hotelDataset['Hotel_name'] == hotel_name]

    # Check if any data was found for the specified hotel name
    if len(hotel_data) == 0:
        return jsonify({'error': f"No data found for {hotel_name}."})
    else:
        # Extract the features for the specified hotel name
        features = hotel_data[['Hotel star rating', 'Distance', 'Rooms', 'Squares']]

        # Load the trained model
        regressor = LinearRegression()
    regressor.fit(features, hotel_data['Price(BAM)'])

    # Predict the price for the specified hotel
    predicted_price = regressor.predict(features)

    return jsonify({'predicted_price': predicted_price[0]}) 


if __name__ == '__main__':
 app.run(debug=True)