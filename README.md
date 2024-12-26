# ExpensePredictor - Advanced Personal Finance Tracker 💰📊

Welcome to ExpensePredictor, a sophisticated Python application designed to help you understand, track, and predict your financial expenses with precision and insight. Leveraging machine learning and data visualization, this tool transforms your spending data into actionable financial intelligence. 💸🔮

## Features 🌟

* **Expense Logging** 📝: Easily add, import, and export your financial transactions with a user-friendly interface.

* **Machine Learning Predictions** 🤖: Utilize advanced XGBoost models to predict future expenses across multiple categories, providing you with forward-looking financial insights.

* **Comprehensive Visualization** 📈: 
  - Trend Analysis: View your monthly spending patterns
  - Category Breakdown: Understand where your money is going
  - Prediction Visualization: See projected expenses with confidence intervals

* **Monthly Overview** 🗓️: Dive deep into monthly spending with interactive pie charts and detailed category-wise breakdowns.

* **Multi-Category Support** 📊: Track expenses across 12 different categories:
  - Housing
  - Transportation
  - Food
  - Utilities
  - Healthcare
  - Insurance
  - Entertainment
  - Shopping
  - Education
  - Savings
  - Debt Payments
  - Other

* **Prediction Confidence Scoring** 📊: Each prediction comes with a confidence metric, helping you understand the reliability of future expense estimates.

## Technical Highlights 🛠️

* **Machine Learning**: Uses XGBoost for robust expense prediction
* **Data Processing**: Powered by Pandas for efficient data manipulation
* **Visualization**: Matplotlib and Seaborn for creating insightful graphics
* **UI**: Tkinter for a responsive, modern desktop application

## Prerequisites 🖥️

* Python 3.8+
* Libraries:
  - tkinter
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - tkcalendar

## Getting Started 🚀

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/ExpensePredictor.git
   cd ExpensePredictor
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Application**
   ```bash
   python expense_predictor.py
   ```

## Usage Tips 💡

* Add expenses manually or import from CSV
* Use the prediction feature to forecast future spending
* Explore trends and category-wise spending in the visualization tabs
* Export your data for further analysis or record-keeping

## Data Generation 🎲

### `data_generator.py` - Synthetic Expense Data Creation

The `data_generator.py` script provides a powerful tool for generating realistic synthetic financial data. Key features include:

* **Comprehensive Category Coverage**: Generates expenses across 12 different categories
* **Realistic Variance**: Incorporates natural spending variations
* **Inflation Modeling**: Applies yearly inflation to expense amounts
* **Seasonal Adjustments**: Accounts for seasonal variations (e.g., higher utility costs in winter)
* **Time Span**: Generates 10 years of financial data (2014-2024)

#### Generation Characteristics:
* **Frequency Variations**: Different expense categories have unique generation frequencies
  - Monthly expenses (Housing, Utilities, Insurance)
  - Weekly expenses (Transportation, Entertainment)
  - Biweekly expenses (Shopping)
  - Daily expenses (Food)

* **Inflation Simulation**: Each category's base amounts increase gradually over time
* **Randomization**: Uses numpy's random generation for natural variability
* **Output**: Produces a `slovenia_expenses.csv` file with comprehensive transaction data

#### How to Use
1. Run the script to generate synthetic financial data
2. Import the generated CSV into the ExpensePredictor application
3. Use the synthetic data for testing, training, or demonstration purposes
