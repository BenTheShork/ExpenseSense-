import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def is_winter_month(month):
    return month in [11, 12, 1, 2, 3]

def apply_yearly_inflation(base_amount, year, start_year, inflation_rate=0.023):
    years_passed = year - start_year
    return base_amount * (1 + inflation_rate) ** years_passed

# Generate dates for 10 years
start_date = datetime(2014, 1, 1)
end_date = datetime(2024, 1, 1)

def get_housing_cost(date, base_summer=200, base_winter=260, yearly_increase=4):
    years_passed = date.year - start_date.year
    base = base_winter if is_winter_month(date.month) else base_summer
    return base + (yearly_increase * years_passed)

categories = {
    'Housing': {
        'frequency': 'monthly'
    },
    'Transportation': {
        'base_range': (50, 90),
        'frequency': 'weekly'
    },
    'Food': {
        'base_range': (200, 300),
        'frequency': 'daily'
    },
    'Utilities': {
        'base_summer': (70, 100),
        'base_winter': (120, 180),
        'frequency': 'monthly'
    },
    'Healthcare': {
        'base_range': (15, 40),
        'frequency': 'monthly'
    },
    'Insurance': {
        'base_range': (40, 60),
        'frequency': 'monthly'
    },
    'Entertainment': {
        'base_range': (40, 120),
        'frequency': 'weekly'
    },
    'Shopping': {
        'base_range': (80, 160),
        'frequency': 'biweekly'
    },
    'Education': {
        'base_range': (20, 60),
        'frequency': 'monthly'
    },
    'Savings': {
        'base_range': (80, 150),
        'frequency': 'monthly'
    },
    'Debt Payments': {
        'base_range': (0, 40),
        'frequency': 'monthly'
    },
    'Other': {
        'base_range': (20, 80),
        'frequency': 'weekly'
    }
}

data = []
current_date = start_date
while current_date < end_date:
    month = current_date.month
    year = current_date.year
    
    # Housing with yearly increase and seasonal variation
    if current_date.day == 1:
        amount = get_housing_cost(current_date)
        data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'amount': round(amount, 2),
            'category': 'Housing'
        })
    
    # Other categories with inflation adjustment
    for category, params in categories.items():
        if category == 'Housing':
            continue
            
        if params['frequency'] == 'monthly' and current_date.day == 1:
            if category == 'Utilities':
                base_range = params['base_winter'] if is_winter_month(month) else params['base_summer']
                min_amount = apply_yearly_inflation(base_range[0], year, start_date.year)
                max_amount = apply_yearly_inflation(base_range[1], year, start_date.year)
                amount = np.random.uniform(min_amount, max_amount)
            else:
                min_amount = apply_yearly_inflation(params['base_range'][0], year, start_date.year)
                max_amount = apply_yearly_inflation(params['base_range'][1], year, start_date.year)
                amount = np.random.uniform(min_amount, max_amount)
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'amount': round(amount, 2),
                'category': category
            })
        
        elif params['frequency'] == 'weekly' and current_date.weekday() == 0:
            min_amount = apply_yearly_inflation(params['base_range'][0], year, start_date.year)
            max_amount = apply_yearly_inflation(params['base_range'][1], year, start_date.year)
            amount = np.random.uniform(min_amount/4, max_amount/4)
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'amount': round(amount, 2),
                'category': category
            })
        
        elif params['frequency'] == 'biweekly' and current_date.day in [1, 15]:
            min_amount = apply_yearly_inflation(params['base_range'][0], year, start_date.year)
            max_amount = apply_yearly_inflation(params['base_range'][1], year, start_date.year)
            amount = np.random.uniform(min_amount/2, max_amount/2)
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'amount': round(amount, 2),
                'category': category
            })
        
        elif params['frequency'] == 'daily':
            if np.random.random() < 0.7:
                min_amount = apply_yearly_inflation(params['base_range'][0], year, start_date.year)
                max_amount = apply_yearly_inflation(params['base_range'][1], year, start_date.year)
                amount = np.random.uniform(min_amount/30, max_amount/30)
                data.append({
                    'date': current_date.strftime('%Y-%m-%d'),
                    'amount': round(amount, 2),
                    'category': category
                })
    
    current_date += timedelta(days=1)

# Create DataFrame and sort by date
df = pd.DataFrame(data)
df = df.sort_values('date')

# Monthly total validation
monthly_totals = df.groupby([df['date'].str[:7]])['amount'].sum()
print("Monthly expense ranges:", monthly_totals.min(), "-", monthly_totals.max(), "€")

# Save to CSV
df.to_csv('slovenia_expenses.csv', index=False)

print(f"Generated {len(df)} transactions from {start_date.date()} to {end_date.date()}")