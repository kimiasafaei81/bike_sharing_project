import matplotlib.pyplot as plt
import seaborn as sns
import os


class BikePlotter:
    def __init__(self, data, save_dir="."):
        self.df = data
        self.save_dir = save_dir
        sns.set_theme(style="whitegrid")

    def plot_weather_impact(self):
        """1. Regression Plot: Temp vs Count"""
        plt.figure(figsize=(10, 6))
        sns.regplot(data=self.df, x='temp', y='cnt', scatter_kws={'alpha': 0.1}, line_kws={'color': 'red'})
        plt.title('Impact of Temperature on Bike Rentals')

        path = os.path.join(self.save_dir, "weather_impact.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_hourly_trend(self):
        """2. Bar Plot: Hourly Trend"""
        plt.figure(figsize=(12, 6))
        sns.barplot(data=self.df, x='hr', y='cnt', palette='viridis')
        plt.title('Average Hourly Bike Rentals')

        path = os.path.join(self.save_dir, "hourly_trend.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_workingday_comparison(self):
        """3. Box Plot: Working Day vs Weekend"""
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.df, x='workingday', y='cnt', palette='Set2')
        plt.title('Rentals: Working Day (1) vs Weekend/Holiday (0)')

        path = os.path.join(self.save_dir, "workingday_comparison.png")
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    def plot_actual_vs_predicted(self, y_true, y_pred):
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title('Prediction Accuracy: Actual vs Predicted')
        path = os.path.join(self.save_dir, "accuracy_check.png")
        plt.savefig(path)
        plt.close()
        return path