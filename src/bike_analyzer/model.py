import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class BikeModel:
    """Handles all statistical analysis and machine learning."""

    def __init__(self, data):
        self.df = data
        self.features = ['hr', 'temp', 'hum', 'windspeed', 'workingday']
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def get_test_predictions(self):
        """Returns actual and predicted values for the test set."""
        X = self.df[self.features]
        y = self.df['cnt']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred = self.model.predict(X_test)
        return y_test, y_pred

    def get_weather_stats(self):
        """Calculates mean rentals by weather condition."""
        return self.df.groupby('weathersit')['cnt'].mean()

    def calculate_correlation(self):
        """Returns the correlation between temperature and rentals."""
        return self.df['temp'].corr(self.df['cnt'])

    def train_prediction_model(self):
        """Trains a linear regression model with temporal and weather features."""
        features = ['hr', 'temp', 'hum', 'windspeed', 'workingday']
        X = self.df[features]
        y = self.df['cnt']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        return "✅ Model trained with Temporal & Weather features!"

    def predict_demand(self, hr, temp, hum, wind, workingday):
        """Predicts rentals based on 5 features."""
        import pandas as pd
        query = pd.DataFrame([[hr, temp, hum, wind, workingday]],
                             columns=['hr', 'temp', 'hum', 'windspeed', 'workingday'])
        return self.model.predict(query)[0]

    def evaluate_model(self):
        """Returns the R-squared score using the updated feature set."""
        features = ['hr', 'temp', 'hum', 'windspeed', 'workingday']
        X = self.df[features]
        y = self.df['cnt']
        return self.model.score(X, y)

    def get_test_predictions(self):
        """Returns actual and predicted values for the test set."""
        X = self.df[self.features]
        y = self.df['cnt']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = self.model.predict(X_test)
        return y_test, y_pred