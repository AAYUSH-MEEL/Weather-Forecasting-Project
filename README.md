# Weather-Forecasting-Project
# Weather Trend Forecasting

## Project Overview
This project analyzes global weather data and forecasts future temperature trends using time series modeling techniques. The dataset is sourced from Kaggle's **Global Weather Repository** and contains **59,243 records with 41 features** related to weather conditions worldwide.

## Dataset
- **Source:** [Kaggle - Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code)
- **Key Features:** Temperature, humidity, wind speed, precipitation, air quality
- **Selected City for Forecasting:** Berlin (305 records available)

## Steps Performed
### 1. **Data Preprocessing**
- Converted **'last_updated'** column to datetime format.
- Handled missing values (none found).
- Normalized and cleaned numerical data.

### 2. **Exploratory Data Analysis (EDA)**
- Visualized **temperature and precipitation** trends.
- Identified patterns in data distribution.

### 3. **Time Series Forecasting**
- Trained an **ARIMA(5,1,0) model** on Berlin's temperature data.
- Forecasted next **7 days** of temperature.
- Evaluated model performance:
  - **Mean Absolute Error (MAE):** 1.18°C
  - **Root Mean Squared Error (RMSE):** 1.40°C

## Deliverables
- **Python Script:** Includes data processing, EDA, and forecasting model.
- **Report:** Summary of methodology, results, and insights.
- **Forecast Visualization:** Graphs illustrating predicted temperature trends.

## How to Run the Code
1. Clone this repository.
2. Install dependencies using:
   ```sh
   pip install pandas numpy matplotlib seaborn statsmodels
   ```
3. Run the Python script to preprocess data, perform EDA, and generate forecasts:
   ```sh
   python weather_forecast.py
   ```

## Future Enhancements
- Use deep learning models (**LSTMs, CNNs**) for improved accuracy.
- Extend forecasting to multiple cities.
- Incorporate additional meteorological factors.

## Author
**AAYUSH MEEL** - Artificial intelligence Enthusiast 🚀
