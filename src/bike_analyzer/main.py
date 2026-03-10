import pandas as pd
import os
import sys


def main():
    """
    Main entry point for the Bike Sharing Analysis project.
    Focuses on environmental impact on total rental counts.
    """
    print("--- 🚲 Bike Sharing Analysis: Weather & Demand ---")

    # Define the data path relative to this script's location
    # Going up 3 levels: src/bike_analyzer/__main__.py -> src/bike_analyzer -> src -> project_root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, "hour.csv")

    # Check if data file exists before processing
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)

    # Load the dataset
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.\n")

    # 1. Analyze Average Rentals per Weather Situation
    # Weather categories: 1: Clear, 2: Mist, 3: Light Rain/Snow, 4: Heavy Rain
    weather_group = df.groupby('weathersit')['cnt'].mean()
    print("Average hourly rentals by weather condition:")
    print(weather_group)

    # 2. Calculate Correlation between Temperature and Demand
    # 'temp' is normalized temperature in Celsius
    temp_correlation = df['temp'].corr(df['cnt'])
    print(f"\nCorrelation between Temp and Total Rentals: {temp_correlation:.4f}")

    # 3. Quick Insight
    if temp_correlation > 0.3:
        print("Insight: There is a significant positive correlation between temperature and bike demand.")


if __name__ == "__main__":
    main()