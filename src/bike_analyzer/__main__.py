import sys
from .data_manager import BikeData
from .model import BikeModel
from .plotter import BikePlotter


def main():
    print("--- 🚲 Bike Sharing Analysis")

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
        #pred = analyst.predict_demand(0.6, 0.5, 0.1)
        #print(f"Prediction for given conditions: {int(pred)} bikes")

        # 3. Interactive Prediction
        print("\n--- 🧠 Predictive System ---")
        print("Please enter the conditions to predict hourly rentals:")

        while True:
            try:
                print("\n--- Input Data ---")
                u_hr = int(input("👉 Hour (0-23): "))
                u_temp = float(input("👉 Normalized Temp (0.0-1.0): "))
                u_hum = float(input("👉 Normalized Humidity (0.0-1.0): "))
                u_wind = float(input("👉 Normalized Windspeed (0.0-1.0): "))
                u_work = int(input("👉 Working Day? (1 for Yes, 0 for No/Weekend): "))

                user_pred = analyst.predict_demand(u_hr, u_temp, u_hum, u_wind, u_work)

                print(f"\n🔮 Prediction Result: {int(user_pred)} bikes/hour")

                accuracy = analyst.evaluate_model()
                print(f"📊 New Model Accuracy (R²): {accuracy:.4f}")
                break

            except ValueError:
                print("❌ Invalid input! Please use numbers.")



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