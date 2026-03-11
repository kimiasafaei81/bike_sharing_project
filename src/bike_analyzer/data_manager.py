import pandas as pd
import os


class BikeData:
    """Handles data loading and basic preprocessing."""

    def __init__(self, file_name="hour.csv"):
        # Find the root path (3 levels up from this file)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.file_path = os.path.join(self.base_dir, file_name)
        self.df = None

    def load_data(self):
        """Loads the CSV file into a DataFrame."""
        if os.path.exists(self.file_path):
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Data loaded from {self.file_path}")
            return self.df
        else:
            raise FileNotFoundError(f"❌ Could not find {self.file_path}")

    def get_summary(self):
        """Returns basic stats of the dataset."""
        if self.df is not None:
            return self.df.describe()
        return "No data loaded."