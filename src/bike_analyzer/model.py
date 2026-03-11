import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class BikeModel:
    """Handles all statistical analysis and machine learning."""

    def __init__(self, data):
        self.df = data
        self.model = LinearRegression()

    def get_weather_stats(self):
        """Calculates mean rentals by weather condition."""
        return self.df.groupby('weathersit')['cnt'].mean()

    def calculate_correlation(self):
        """Returns the correlation between temperature and rentals."""
        return self.df['temp'].corr(self.df['cnt'])

    def train_prediction_model(self):
        """Trains a simple linear regression model."""
        X = self.df[['temp', 'hum', 'windspeed']]
        y = self.df['cnt']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        return "Model trained successfully!"

    def predict_demand(self, temp, hum, wind):
        """Predicts rentals based on input conditions."""
        return self.model.predict([[temp, hum, wind]])[0]