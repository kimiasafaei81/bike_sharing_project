import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    """
    Main entry point for the Bike Sharing Analysis project.
    Focuses on environmental impact on total rental counts.
    """
    print("--- 🚲 Bike Sharing Analysis: Weather & Demand ---")

    # 1. Define paths using os.path
    # base_dir is the root of your project
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, "hour.csv")

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)

    # Load the dataset
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully.\n")

    # 2. Analysis
    weather_group = df.groupby('weathersit')['cnt'].mean()
    print("Average hourly rentals by weather condition:")
    print(weather_group)

    temp_correlation = df['temp'].corr(df['cnt'])
    print(f"\nCorrelation between Temp and Total Rentals: {temp_correlation:.4f}")

    if temp_correlation > 0.3:
        print("Insight: Significant positive correlation between temperature and demand.")

    # 3. Visualization
    print("\nGenerating visualization...")
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='temp', y='cnt', scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'})
    plt.title('Impact of Temperature on Bike Rentals')
    plt.xlabel('Normalized Temperature')
    plt.ylabel('Total Rentals')

    # 4. Save the plot correctly using os.path
    # We use base_dir which is already defined above
    plot_path = os.path.join(base_dir, "weather_impact.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()