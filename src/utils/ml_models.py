import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class ExpensePredictor:
    def __init__(self, categories):
        self.categories = categories
        self.category_models = {}
        self.trend_factors = {}
        self.scaler = StandardScaler()

    def prepare_data(self, data):
        """Prepare data for ML model with trend analysis"""
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Ensure required columns exist
        required_columns = ['date', 'category', 'amount']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            try:
                data['date'] = pd.to_datetime(data['date'])
            except:
                raise ValueError("Cannot convert 'date' column to datetime")

        if len(data) < 6:
            return None

        df = data.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Group by year, month, and category
        monthly_expenses = df.groupby(['year', 'month', 'category'])['amount'].sum().reset_index()
        
        # Create pivot table
        pivot_df = monthly_expenses.pivot_table(
            index=['year', 'month'],
            columns='category',
            values='amount',
            fill_value=0
        ).reset_index()
        
        # Ensure all categories are present
        for category in self.categories:
            if category not in pivot_df.columns:
                pivot_df[category] = 0
        
        return pivot_df[['year', 'month']], pivot_df[self.categories]

    def _calculate_trend_factors(self, y):
        """Calculate trend factors for each category"""
        trend_factors = {}
        for category in self.categories:
            # Skip if all zeros
            if y[category].eq(0).all():
                trend_factors[category] = 0.02
                continue

            # Calculate year-over-year and recent trend
            category_data = y[category]
            
            # Last 3 years yearly growth
            try:
                yearly_data = category_data.groupby(category_data.index.get_level_values('year')).mean()
                # Explicitly specify dtype as float64
                yearly_growth_rates = pd.Series(dtype=float) if len(yearly_data) <= 1 else yearly_data.pct_change().dropna()
            except:
                # Explicitly specify dtype as float64
                yearly_growth_rates = pd.Series(dtype=float)
            
            # Recent monthly trend (last 12 months)
            recent_data = category_data.tail(12)
            monthly_trend = recent_data.pct_change().mean()
            
            # Combine trends with more weight to recent trend
            if len(yearly_growth_rates) > 0:
                avg_yearly_growth = yearly_growth_rates.mean()
                trend_factor = (0.4 * avg_yearly_growth + 0.6 * monthly_trend)
            else:
                trend_factor = monthly_trend
            
            # Bound the trend factor between -15% and +15%
            trend_factor = max(min(trend_factor, 0.15), -0.15)
            
            # Default to 2% if no significant trend detected
            trend_factors[category] = trend_factor if abs(trend_factor) > 0.01 else 0.02
    
        return trend_factors

    def train_models(self, data):
        """Train prediction models"""
        try:
            result = self.prepare_data(data)
            if result is None:
                return False, {}

            X, y = result
            X_scaled = self.scaler.fit_transform(X)
            confidence_scores = {}
            
            # Calculate trend factors
            self.trend_factors = self._calculate_trend_factors(y)
            
            active_categories = [cat for cat in self.categories 
                               if not y[cat].eq(0).all()]
            
            for category in self.categories:
                if category not in active_categories:
                    confidence_scores[category] = 0
                    continue
                    
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y[category], test_size=0.2, random_state=42
                )
                
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.2,
                    max_depth=20,
                    random_state=42
                )
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                
                confidence = self._calculate_confidence(y_train, y_val, val_pred)
                confidence_scores[category] = confidence
                self.category_models[category] = model
            
            return True, confidence_scores
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, {}

    def predict_future_expenses(self, last_date, months_ahead):
        """Predict future expenses with trend adjustment"""
        if not isinstance(last_date, pd.Timestamp):
            last_date = pd.Timestamp(last_date)

        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=months_ahead, 
            freq='M'
        )
        
        future_X = pd.DataFrame({
            'year': future_dates.year,
            'month': future_dates.month
        })
        
        future_X_scaled = self.scaler.transform(future_X)
        predictions = pd.DataFrame(index=future_dates)
        
        for category in self.categories:
            if category in self.category_models:
                model = self.category_models[category]
                base_pred = model.predict(future_X_scaled)
                
                # Apply trend factor to prediction
                trend_factor = self.trend_factors.get(category, 0.02)
                adjusted_pred = base_pred * (1 + trend_factor)
                
                predictions[category] = adjusted_pred
                print(f"Predicted {category} with trend factor {trend_factor}")
            else:
                predictions[category] = 0
                
        return predictions

    def _calculate_confidence(self, y_train, y_val, val_pred):
        """Calculate confidence score for predictions"""
        data_points = len(y_train)
        if data_points == 0:
            return 0

        # Consistency score
        std_dev = np.std(y_train)
        mean_val = np.mean(y_train)
        cv = std_dev / mean_val if mean_val != 0 else float('inf')
        consistency_score = 1 / (1 + cv**2)
        
        # Quantity score
        min_points = 6
        optimal_points = 12
        quantity_score = min(1.0, (data_points - min_points) / 
                           (optimal_points - min_points))
        
        # Accuracy score
        mae = np.mean(np.abs(y_val - val_pred))
        accuracy_score = 1 - min(1, mae / (mean_val if mean_val != 0 else 1))
        
        # Combined score
        confidence = (
            consistency_score * 0.4 +
            quantity_score * 0.4 +
            accuracy_score * 0.2
        ) * 100
        
        return max(min(confidence, 95), 40)