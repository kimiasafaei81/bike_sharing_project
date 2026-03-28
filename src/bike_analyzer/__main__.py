import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from bike_analyzer.model import BikeModel
try:
    from bike_analyzer.plotter import BikePlotter
except ImportError:
    print("❌ Error: Could not find BikePlotter class. Make sure plotter.py exists.")
    sys.exit(1)

def main():
    print("--- 🚲 Bike Sharing Analysis: Weather & Demand ---")


    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
    data_path = os.path.join(base_dir, "hour.csv")

    if not os.path.exists(data_path):
        data_path = "hour.csv"

    try:
        df = pd.read_csv(data_path)
        print(f"✅ Dataset loaded successfully from: {data_path}\n")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        sys.exit(1)


    weather_group = df.groupby('weathersit')['cnt'].mean()
    print("Average hourly rentals by weather condition:")
    print(weather_group)


    viz_manager = BikePlotter(df, save_dir=os.getcwd())

    print("\n📊 Generating and Saving Plots (PNG)...")
    try:
        p1 = viz_manager.plot_weather_impact()
        p2 = viz_manager.plot_hourly_trend()
        p3 = viz_manager.plot_workingday_comparison()

        print(f"✅ Saved: {p1}")
        print(f"✅ Saved: {p2}")
        print(f"✅ Saved: {p3}")
    except AttributeError as e:
        print(f"❌ Plotting Error: {e}")
        print("Tip: Make sure all methods (like plot_hourly_trend) are defined in plotter.py")

    print("\n🤖 Training Model...")
    analyst = BikeModel(df)
    print(analyst.train_prediction_model())

    accuracy = analyst.evaluate_model()
    print(f"📊 Model Accuracy (R²): {accuracy:.4f}")
    print("📈 Generating Accuracy Plot...")
    y_actual, y_predicted = analyst.get_test_predictions()
    p4 = viz_manager.plot_actual_vs_predicted(y_actual, y_predicted)
    print(f"✅ Accuracy check saved: {p4}")

    #Interactive Prediction
    print("\n🔮 --- Manual Prediction ---")
    try:
        hr = int(input("Enter hour (0-23): "))
        temp = float(input("Enter normalized temperature (0-1): "))
        hum = float(input("Enter humidity (0-1): "))
        wind = float(input("Enter windspeed (0-1): "))
        workingday = int(input("Working day? (1 for Yes, 0 for No): "))

        prediction = analyst.predict_demand(hr, temp, hum, wind, workingday)
        print(f"\n✨ Predicted Bike Demand: {int(prediction)} bikes")

    except ValueError:
        print("❌ Invalid input! Please enter numbers only.")

if __name__ == "__main__":
    main()