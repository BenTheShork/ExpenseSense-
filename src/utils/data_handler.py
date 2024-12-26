import pandas as pd
from pathlib import Path

class DataHandler:
    def __init__(self, storage_file='data/expense_data.pkl'):
        self.storage_file = Path(storage_file)
        self.data = pd.DataFrame()
        self._ensure_data_directory()

    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        self.storage_file.parent.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load data from pickle file"""
        try:
            self.data = pd.read_pickle(self.storage_file)
            self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"Loaded {len(self.data)} existing records")
        except FileNotFoundError:
            self.data = pd.DataFrame(columns=['date', 'amount', 'category'])
            print("No existing data found, starting fresh")
        return self.data

    def save_data(self, data=None):
        """Save data to pickle file"""
        if data is not None:
            self.data = data
        try:
            self.data.to_pickle(self.storage_file)
            print(f"Saved {len(self.data)} records")
        except Exception as e:
            print(f"Error saving data: {str(e)}")

    def import_csv(self, filename):
        """Import data from CSV file"""
        try:
            df = pd.read_csv(filename)
            df['date'] = pd.to_datetime(df['date'])
            self.data = df
            self.save_data()
            return True, "Data imported successfully"
        except Exception as e:
            return False, f"Error importing CSV: {str(e)}"

    def export_csv(self, filename):
        """Export data to CSV file"""
        try:
            self.data.to_csv(filename, index=False)
            return True, "Data exported successfully"
        except Exception as e:
            return False, f"Error exporting CSV: {str(e)}"

    def add_expense(self, date, amount, category):
        """Add new expense record"""
        new_data = pd.DataFrame({
            'date': [date],
            'amount': [amount],
            'category': [category]
        })
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        self.save_data()
        return True

    def get_monthly_totals(self):
        """Get monthly expense totals"""
        return self.data.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum()

    def get_category_totals(self):
        """Get total expenses by category"""
        return self.data.groupby('category')['amount'].sum().sort_values(ascending=True)

    def get_monthly_category_data(self, selected_month):
        """Get expense data for specific month"""
        selected_period = pd.Period(selected_month)
        return self.data[pd.to_datetime(self.data['date']).dt.to_period('M') == selected_period]