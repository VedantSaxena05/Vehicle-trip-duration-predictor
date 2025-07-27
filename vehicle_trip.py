import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For saving/loading models

dt = pd.read_csv(r"C:\\Users\\Lanovo\\OneDrive\\Desktop\\Vehicle Trip Duration Predictor\\vehicle_trip_duration_simulated.csv")
dt['pickup_datetime'] = pd.to_datetime(dt['pickup_datetime'])

dt['hour'] = dt['pickup_datetime'].dt.hour
dt['day_of_week'] = dt['pickup_datetime'].dt.dayofweek
dt['is_weekend'] = dt['day_of_week'].isin([5,6]).astype(int)

X = dt[['distance_km', 'hour', 'day_of_week', 'is_weekend']]
y = dt['duration_minutes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

joblib.dump(model, "vehicle_trip_model.pkl")
