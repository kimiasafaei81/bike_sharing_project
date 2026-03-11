import sys
from .data_manager import BikeData
from .model import BikeModel
from .plotter import BikePlotter


def main():
    print("--- 🚲 Bike Sharing Analysis: Professional OOP Version ---")

    try:
        # 1. Initialize and Load Data
        data_engine = BikeData("hour.csv")
        df = data_engine.load_data()

        # 2. Analysis & Machine Learning
        analyst = BikeModel(df)

        # Weather Stats
        print("\nAverage rentals by weather condition:")
        print(analyst.get_weather_stats())

        # ML Training
        print(f"\n{analyst.train_prediction_model()}")

        # Example Prediction (Temp=0.6, Hum=0.5, Wind=0.1)
        pred = analyst.predict_demand(0.6, 0.5, 0.1)
        print(f"Prediction for given conditions: {int(pred)} bikes")

        # 3. Visualization
        # Using data_engine.base_dir to save the plot in the root folder
        visualizer = BikePlotter(df, data_engine.base_dir)
        path = visualizer.plot_weather_impact()
        print(f"\n✅ Visualization generated and saved to: {path}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()