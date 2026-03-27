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
        pred = analyst.predict_demand(0.6, 0.5, 0.1)
        print(f"Prediction for given conditions: {int(pred)} bikes")

        # 3. Interactive Prediction
        print("\n--- 🧠 Predictive System ---")
        print("Please enter the conditions to predict hourly rentals:")

        try:
            # getting input
            u_temp = float(input("Enter Normalized Temperature (0.0 to 1.0): "))
            u_hum = float(input("Enter Normalized Humidity (0.0 to 1.0): "))
            u_wind = float(input("Enter Normalized Windspeed (0.0 to 1.0): "))

            # prediction
            user_pred = analyst.predict_demand(u_temp, u_hum, u_wind)

            print(f"\n🔮 Prediction Results:")
            print(f"For Temp:{u_temp}, Hum:{u_hum}, Wind:{u_wind}")
            print(f"👉 Estimated Hourly Rentals: {int(user_pred)} bikes")

        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")

        # Display Accuracy
        accuracy = analyst.evaluate_model()
        print(f"\n📊 Model Accuracy (R² Score): {accuracy:.4f}")
        print(f"Interpretation: {accuracy * 100:.1f}% of variance explained.")



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