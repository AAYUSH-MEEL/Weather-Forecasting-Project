import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = "GlobalWeatherRepository.csv"  # Update with correct path
df = pd.read_csv(file_path)

# Convert 'last_updated' column to datetime
df["last_updated"] = pd.to_datetime(df["last_updated"])

# Select city with sufficient data
city = "Berlin"
df_city = df[df["location_name"] == city].copy()

# Ensure data is sorted by time
df_city = df_city.sort_values("last_updated")

# Set time index
df_city.set_index("last_updated", inplace=True)

# Select temperature for time series forecasting
temperature_series = df_city["temperature_celsius"]

# Train ARIMA model
model = ARIMA(temperature_series, order=(5,1,0))
model_fit = model.fit()

# Forecast next 7 days
forecast_steps = 7
forecast = model_fit.forecast(steps=forecast_steps)

# Evaluate model
actual_values = temperature_series[-forecast_steps:]
mae = mean_absolute_error(actual_values, forecast[:len(actual_values)])
rmse = np.sqrt(mean_squared_error(actual_values, forecast[:len(actual_values)]))

# Plot results
plt.figure(figsize=(10,5))
plt.plot(temperature_series[-30:], label="Actual Temperature", color='blue')
plt.plot(forecast, label="Forecasted Temperature", color='red', linestyle='dashed')
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title(f"Temperature Forecast for {city}")
plt.legend()
plt.show()

# Print model performance
print(f"Mean Absolute Error (MAE): {mae:.2f}°C")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}°C")
