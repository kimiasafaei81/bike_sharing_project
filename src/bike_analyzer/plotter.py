import matplotlib.pyplot as plt
import seaborn as sns
import os


class BikePlotter:
    """Handles all graphical visualizations for the bike project."""

    def __init__(self, df, save_dir):
        self.df = df
        self.save_dir = save_dir

    def plot_weather_impact(self):
        """Creates and saves a regression plot for temperature vs demand."""
        plt.figure(figsize=(10, 6))
        sns.regplot(
            data=self.df,
            x='temp',
            y='cnt',
            scatter_kws={'alpha': 0.1},
            line_kws={'color': 'red'}
        )
        plt.title('Impact of Temperature on Bike Rentals')
        plt.xlabel('Normalized Temperature')
        plt.ylabel('Total Rentals')

        # Define the full path for the output image
        save_path = os.path.join(self.save_dir, "weather_impact.png")
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory
        return save_path