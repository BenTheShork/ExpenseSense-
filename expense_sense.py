import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import calendar
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
import joblib
from tkcalendar import DateEntry    

class ExpensePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Expense Predictor")
        self.root.geometry("1800x1200")
        
        # Initialize variables
        self.category_models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.data = pd.DataFrame()
        
        # Define categories and storage
        self.categories = [
            'Housing', 'Transportation', 'Food', 'Utilities', 
            'Healthcare', 'Insurance', 'Entertainment', 'Shopping',
            'Education', 'Savings', 'Debt Payments', 'Other'
        ]
        self.storage_file = 'expense_data.pkl'
        
        # Load data and setup GUI
        self.load_data()
        self.setup_gui()
        self.update_plots()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


    def load_data(self):
        try:
            self.data = pd.read_pickle(self.storage_file)
            # Convert date column back to datetime
            self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"Loaded {len(self.data)} existing records")
        except FileNotFoundError:
            self.data = pd.DataFrame(columns=['date', 'amount', 'category'])
            print("No existing data found, starting fresh")
    
    def save_data(self):
        try:
            self.data.to_pickle(self.storage_file)
            print(f"Saved {len(self.data)} records")
        except Exception as e:
            print(f"Error saving data: {str(e)}")
        
    def setup_gui(self):
        style = ttk.Style()
        
        # Style configurations
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10), padding=5, background='#0078D4')
        style.configure('TEntry', font=('Segoe UI', 10), padding=5)
        style.configure('Heading.TLabel', font=('Segoe UI', 12, 'bold'), padding=10)
        
        # Container setup
        main_container = ttk.Frame(self.root, padding="20")
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Input section
        input_section = ttk.Frame(main_container, padding="10")
        input_section.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        ttk.Label(input_section, text="Add New Expense", style='Heading.TLabel').grid(row=0, column=0, columnspan=6, sticky="w")
        
        # Input fields
        row = 1
        ttk.Label(input_section, text="Amount (€):").grid(row=row, column=0, padx=(0, 5))
        self.amount_var = tk.StringVar()
        amount_entry = ttk.Entry(input_section, textvariable=self.amount_var, width=15)
        amount_entry.grid(row=row, column=1, padx=5)
        
        ttk.Label(input_section, text="Date:").grid(row=row, column=2, padx=5)
        self.date_picker = DateEntry(input_section, width=12, background='#0078D4',
                                foreground='white', borderwidth=2,
                                locale='en_US', date_pattern='yyyy-mm-dd')
        self.date_picker.grid(row=row, column=3, padx=5)
        self.date_picker.set_date(datetime.now())
        
        ttk.Label(input_section, text="Category:").grid(row=row, column=4, padx=5)
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(input_section, textvariable=self.category_var,
                                        values=self.categories, width=20)
        self.category_combo.grid(row=row, column=5, padx=5)
        
        # Buttons
        row += 1
        buttons_frame = ttk.Frame(input_section, padding="10")
        buttons_frame.grid(row=row, column=0, columnspan=6, pady=10)
        
        ttk.Button(buttons_frame, text="Add Expense", command=self.add_expense).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Import CSV", command=self.import_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        
        # Prediction controls
        pred_frame = ttk.Frame(buttons_frame)
        pred_frame.pack(side=tk.LEFT, padx=20)
        ttk.Button(pred_frame, text="Predict", command=self.predict_expenses).pack(side=tk.LEFT, padx=5)
        
        # Notebook setup
        self.notebook = ttk.Notebook(main_container)
        self.notebook.grid(row=2, column=0, sticky="nsew", pady=10)
        
        # Create tabs
        self.trends_frame = ttk.Frame(self.notebook)
        self.category_frame = ttk.Frame(self.notebook)
        self.prediction_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.trends_frame, text='Trends')
        self.notebook.add(self.category_frame, text='Categories')
        self.notebook.add(self.prediction_frame, text='Predictions')
        
        # Make container expandable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize plots
        self.setup_plots()

        ttk.Label( text="Months to predict:").grid(row=row, column=5, padx=5)
        self.prediction_months = tk.StringVar(value="3")
        months_spinbox = ttk.Spinbox(from_=1, to=24, width=5, 
                                    textvariable=self.prediction_months)
        months_spinbox.grid(row=row, column=6, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        ttk.Button(buttons_frame, text="Monthly Overview", 
               command=self.show_monthly_overview).pack(side=tk.LEFT, padx=5)

   
    def show_monthly_overview(self):
        if self.data.empty:
            messagebox.showerror("Error", "No data available")
            return

        overview_window = tk.Toplevel(self.root)
        overview_window.title("Monthly Category Overview")
        overview_window.geometry("1200x800")

        # Create main frame with left and right sections
        main_frame = ttk.Frame(overview_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for treeview
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right frame for pie chart
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Month selector
        ttk.Label(left_frame, text="Select Month:").pack(anchor=tk.W)
        month_var = tk.StringVar()
        periods = pd.to_datetime(self.data['date']).dt.to_period('M')
        month_strings = [str(period) for period in sorted(periods.unique())]
        
        combo = ttk.Combobox(left_frame, textvariable=month_var, values=month_strings, state='readonly')
        combo.pack(anchor=tk.W, pady=(0, 10))
        if month_strings:
            combo.set(month_strings[-1])

        # Create Treeview with scrollbar
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        tree = ttk.Treeview(tree_frame, columns=("category", "amount", "percentage"), show="headings", height=15)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        # Configure columns
        tree.heading("category", text="Category")
        tree.heading("amount", text="Amount (€)")
        tree.heading("percentage", text="% of Total")
        tree.column("category", width=150)
        tree.column("amount", width=150, anchor=tk.E)
        tree.column("percentage", width=100, anchor=tk.E)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create pie chart
        pie_figure, pie_ax = plt.subplots(figsize=(8, 8))
        pie_canvas = FigureCanvasTkAgg(pie_figure, master=right_frame)
        pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def update_view(*args):
            selected_month = month_var.get()
            if not selected_month:
                return

            tree.delete(*tree.get_children())
            pie_ax.clear()
            
            selected_period = pd.Period(selected_month)
            month_data = self.data[pd.to_datetime(self.data['date']).dt.to_period('M') == selected_period]
            
            total = month_data['amount'].sum()
            by_category = month_data.groupby('category')['amount'].sum().sort_values(ascending=False)

            # Update tree
            for category in self.categories:
                amount = by_category.get(category, 0)
                percentage = (amount / total * 100) if total > 0 else 0
                if amount > 0:  # Only show categories with expenses
                    tree.insert("", tk.END, values=(
                        category,
                        f"€{amount:,.2f}",
                        f"{percentage:.1f}%"
                    ))

            # Update pie chart
            non_zero_categories = by_category[by_category > 0]
            if not non_zero_categories.empty:
                wedges, texts, autotexts = pie_ax.pie(
                    non_zero_categories.values,
                    labels=non_zero_categories.index,
                    autopct='%1.1f%%',
                    startangle=90
                )
                
                # Make percentage labels more readable
                for autotext in autotexts:
                    autotext.set_fontsize(8)
                for text in texts:
                    text.set_fontsize(8)
                
                pie_ax.set_title(f'Expense Distribution - {selected_month}')
            
            pie_figure.tight_layout()
            pie_canvas.draw()
            tree.update_idletasks()

        month_var.trace('w', update_view)

        # Summary frame at bottom
        summary_frame = ttk.Frame(left_frame)
        summary_frame.pack(fill=tk.X, pady=(10, 0))
        summary_label = ttk.Label(summary_frame, text="")
        summary_label.pack(anchor=tk.W)

        def update_summary(*args):
            selected_month = month_var.get()
            if not selected_month:
                return

            selected_period = pd.Period(selected_month)
            month_data = self.data[pd.to_datetime(self.data['date']).dt.to_period('M') == selected_period]
            total = month_data['amount'].sum()
            transactions = len(month_data)

            summary_label.config(
                text=f"Total: €{total:,.2f} | Transactions: {transactions}")

        month_var.trace('w', update_summary)
        
        # Initial updates
        update_view()
        update_summary()

        def update_summary(*args):
            selected_month = month_var.get()
            if not selected_month:
                return

            selected_period = pd.Period(selected_month)
            month_data = self.data[pd.to_datetime(self.data['date']).dt.to_period('M') == selected_period]
            total = month_data['amount'].sum()
            transactions = len(month_data)

            summary_label.config(
                text=f"Total: €{total:,.2f} | Transactions: {transactions}")

            month_var.trace('w', update_summary)

    def on_closing(self):
        """Handle window close event"""
        self.save_data()
        self.root.quit()
        self.root.destroy()

    def setup_plots(self):
        # Increase figure sizes
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 12
        })
        
        # Trends plot
        self.trends_figure, self.trends_ax = plt.subplots(figsize=(15, 8))
        self.trends_canvas = FigureCanvasTkAgg(self.trends_figure, master=self.trends_frame)
        self.trends_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Category plot
        self.category_figure, self.category_ax = plt.subplots(figsize=(15, 8))
        self.category_canvas = FigureCanvasTkAgg(self.category_figure, master=self.category_frame)
        self.category_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Prediction plot
        self.prediction_figure, self.prediction_ax = plt.subplots(figsize=(15, 8))
        self.prediction_canvas = FigureCanvasTkAgg(self.prediction_figure, master=self.prediction_frame)
        self.prediction_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def add_expense(self):
        try:
            amount = float(self.amount_var.get())
            date = pd.to_datetime(self.date_picker.get_date())
            category = self.category_var.get()
            
            if not category:
                messagebox.showerror("Error", "Please select a category")
                return
                
            new_data = pd.DataFrame({
                'date': [date],
                'amount': [amount],
                'category': [category]
            })
            
            self.data = pd.concat([self.data, new_data], ignore_index=True)
            self.save_data()  # Save after adding new expense
            self.update_plots()
            self.amount_var.set("")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
    
    def prepare_data_for_ml(self):
        if len(self.data) < 6:
            return None
            
        df = self.data.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        monthly_expenses = df.groupby(['year', 'month', 'category'])['amount'].sum().reset_index()
        
        pivot_df = monthly_expenses.pivot_table(
            index=['year', 'month'],
            columns='category',
            values='amount',
            fill_value=0
        ).reset_index()
        
        for category in self.categories:
            if category not in pivot_df.columns:
                pivot_df[category] = 0
            return (pivot_df[['year', 'month']], pivot_df[self.categories])
            
        df = self.data.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Create time-based features
        df['days_in_month'] = df['date'].dt.days_in_month
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
        
        # Calculate rolling statistics
        monthly_expenses = df.groupby(['year', 'month', 'category'])['amount'].sum().reset_index()
        
        # Pivot the categories and ensure all categories exist
        pivot_df = monthly_expenses.pivot_table(
            index=['year', 'month'],
            columns='category',
            values='amount',
            fill_value=0
        ).reset_index()
        
        # Add missing categories with zero values
        for category in self.categories:
            if category not in pivot_df.columns:
                pivot_df[category] = 0
        
        # Prepare features and target
        features = pivot_df[['year', 'month']]
        targets = pivot_df[self.categories]
        
        return features, targets
    
    def train_models(self):
        result = self.prepare_data_for_ml()
        if result is None:
            return False, {}
        X, y = result
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            confidence_scores = {}
            
            # Only train models for categories with data
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
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=42
                )
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                
                data_points = len(y_train)
                
                # Enhanced confidence calculation
                std_dev = np.std(y_train)
                mean_val = np.mean(y_train)
                cv = std_dev / mean_val if mean_val != 0 else float('inf')
                consistency_score = 1 / (1 + cv**2)
                
                # Adjusted quantity scoring
                min_points = 6
                optimal_points = 12
                quantity_score = min(1.0, (data_points - min_points) / 
                                (optimal_points - min_points))
                
                # Improved accuracy scoring
                mae = np.mean(np.abs(y_val - val_pred))
                accuracy_score = 1 - min(1, mae / (mean_val if mean_val != 0 else 1))
                
                confidence = (
                    consistency_score * 0.4 +
                    quantity_score * 0.4 +    # Increased weight for data quantity
                    accuracy_score * 0.2
                ) * 100
                
                # Adjusted confidence range
                confidence = max(min(confidence, 95), 40)  # Raised minimum to 40%
                confidence_scores[category] = confidence
                self.category_models[category] = model
            
            return True, confidence_scores
            
        except Exception as e:
            print(f"Error training models: {str(e)}")
            return False, {}
        
    def predict_expenses(self):
        success, confidence_scores = self.train_models()
        if success is False:
            messagebox.showerror("Error", "Not enough data for prediction (minimum 6 months required)")
            return
            
        try:
            months_ahead = int(self.prediction_months.get())
            last_date = self.data['date'].max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                    periods=months_ahead, freq='M')
            
            future_X = pd.DataFrame({
                'year': future_dates.year,
                'month': future_dates.month
            })
            
            # Scale features
            future_X_scaled = self.scaler.transform(future_X)
            
            # Make predictions for each category
            predictions = pd.DataFrame(index=future_dates)
            for category in self.categories:
                if category in self.category_models:
                    model = self.category_models[category]
                    pred = model.predict(future_X_scaled)
                    predictions[category] = pred
                else:
                    predictions[category] = 0
            
            self.plot_predictions(predictions)
            self.show_prediction_summary(predictions)
            
            # Show confidence scores
            confidence_window = tk.Toplevel(self.root)
            confidence_window.title("Prediction Confidence Scores")
            confidence_window.geometry("300x400")
            
            text = tk.Text(confidence_window, wrap=tk.WORD, padx=10, pady=10)
            text.pack(fill=tk.BOTH, expand=True)
            
            text.insert(tk.END, f"Predictions for next {months_ahead} months\n\n")
            text.insert(tk.END, "Confidence by Category:\n\n")
            for category, confidence in confidence_scores.items():
                text.insert(tk.END, f"{category}: {confidence:.1f}%\n")
            text.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {str(e)}")
    
    def plot_predictions(self, predictions):
        self.prediction_ax.clear()
        
        historical = self.data.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().reset_index()
        self.prediction_ax.plot(historical['date'], historical['amount'], 
                            label='Historical', color='blue', marker='o', linewidth=2, picker=5)
        
        total_predictions = predictions.sum(axis=1)
        pred_line = self.prediction_ax.plot(predictions.index, total_predictions, 
                            label='Predicted', color='red', linestyle='--', 
                            marker='s', linewidth=2, markersize=8, picker=5)
        
        # Store predictions data for tooltip access
        self.current_predictions = predictions
        self.current_total_predictions = total_predictions
        
        def on_pick(event):
            artist = event.artist
            ind = event.ind[0]
            
            if artist.get_label() == 'Historical':
                date = historical['date'].iloc[ind]
                value = historical['amount'].iloc[ind]
                month_data = self.data[self.data['date'].dt.to_period('M') == date.to_period('M')]
                category_amounts = month_data.groupby('category')['amount'].sum()
                
                text = f"Date: {date.strftime('%B %Y')}\n"
                text += f"Total: €{value:,.2f}\n\n"
                text += "Category Breakdown:\n"
                for cat, amt in category_amounts.items():
                    text += f"{cat}: €{amt:,.2f}\n"
            else:
                # Handle prediction points
                dates = predictions.index
                if ind < len(dates):
                    date = dates[ind]
                    value = total_predictions.iloc[ind]
                    
                    text = f"Predicted for {date.strftime('%B %Y')}:\n"
                    text += f"Total: €{value:,.2f}\n\n"
                    text += "Category Breakdown:\n"
                    for cat in predictions.columns:
                        text += f"{cat}: €{predictions.loc[date, cat]:,.2f}\n"
            
            x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()
            tooltip = tk.Toplevel()
            tooltip.wm_geometry(f"+{x+10}+{y+10}")
            tooltip.wm_overrideredirect(True)
            
            label = tk.Label(tooltip, text=text, justify=tk.LEFT, 
                            background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                            font=("Arial", 10), padx=5, pady=5)
            label.pack()
            
            def hide_tooltip(event=None):
                tooltip.destroy()
            
            tooltip.bind('<Leave>', hide_tooltip)
            tooltip.after(3000, hide_tooltip)
        
        self.prediction_figure.canvas.mpl_connect('pick_event', on_pick)
        
        # Formatting remains the same
        self.prediction_ax.grid(True, linestyle='--', alpha=0.7)
        self.prediction_ax.set_xlabel('Date', fontsize=12)
        self.prediction_ax.set_ylabel('Monthly Total (€)', fontsize=12)
        self.prediction_ax.set_title('Monthly Expense Predictions', fontsize=14, pad=20)
        self.prediction_ax.legend(loc='upper right', frameon=True)
        self.prediction_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
        plt.setp(self.prediction_ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        self.prediction_ax.margins(x=0.05)
        self.prediction_figure.tight_layout()
        self.prediction_canvas.draw()
    
    def show_prediction_summary(self, predictions):
        summary = predictions.sum().round(2)
        
        # Create summary window
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Prediction Summary")
        summary_window.geometry("400x500")
        
        # Add summary text
        text_widget = tk.Text(summary_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        text_widget.insert(tk.END, "Predicted Monthly Expenses by Category:\n\n")
        for category in self.categories:
            avg_prediction = predictions[category].mean()
            text_widget.insert(tk.END, f"{category}: ${avg_prediction:.2f}\n")
        
        text_widget.insert(tk.END, f"\nTotal Monthly Average: ${predictions.sum(axis=1).mean():.2f}")
        text_widget.config(state=tk.DISABLED)
    
    def update_plots(self):
        if self.data.empty:
            return
            
        self.trends_ax.clear()
        monthly_total = self.data.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum()
        self.trends_ax.plot(monthly_total.index, monthly_total.values, marker='o', linewidth=2)
        self.trends_ax.grid(True, linestyle='--', alpha=0.7)
        self.trends_ax.set_xlabel('Date', fontsize=12)
        self.trends_ax.set_ylabel('Total Expenses (€)', fontsize=12)
        self.trends_ax.set_title('Monthly Expense Trends', fontsize=14, pad=20)
        self.trends_ax.tick_params(axis='x', rotation=45)
        self.trends_figure.tight_layout()
        self.trends_canvas.draw()
        
        self.category_ax.clear()
        category_totals = self.data.groupby('category')['amount'].sum().sort_values(ascending=True)
        bars = category_totals.plot(kind='barh', ax=self.category_ax)
        self.category_ax.grid(True, linestyle='--', alpha=0.7, axis='x')
        self.category_ax.set_title('Total Expenses by Category', fontsize=14, pad=20)
        self.category_ax.set_xlabel('Total Amount (€)', fontsize=12)
        
        for i, v in enumerate(category_totals):
            self.category_ax.text(v, i, f'€{v:,.0f}', fontsize=10, va='center', fontweight='bold')
        
        self.category_figure.tight_layout()
        self.category_canvas.draw()

    
    def import_csv(self):
        filename = filedialog.askopenfilename(
            title="Import CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if filename:
            try:
                df = pd.read_csv(filename)
                df['date'] = pd.to_datetime(df['date'])
                self.data = df
                self.save_data()  # Save after importing
                self.update_plots()
                messagebox.showinfo("Success", "Data imported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Error importing CSV: {str(e)}")
    
    def export_csv(self):
        if self.data.empty:
            messagebox.showerror("Error", "No data to export")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Export CSV",
            filetypes=[("CSV files", "*.csv")],
            defaultextension=".csv"
        )
        if filename:
            try:
                self.data.to_csv(filename, index=False)
                messagebox.showinfo("Success", "Data exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting CSV: {str(e)}")
    
    def show_analytics(self):
        if self.data.empty:
            messagebox.showerror("Error", "No data available for analysis")
            return
        
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Expense Analytics")
        analysis_window.geometry("600x800")
        
        text_widget = tk.Text(analysis_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        analysis = self.generate_analytics()
        text_widget.insert(tk.END, analysis)
        text_widget.config(state=tk.DISABLED)
    
    def generate_analytics(self):
        analysis = "Expense Analysis Report\n"
        analysis += "=" * 50 + "\n\n"
        
        # Overall statistics
        total_spent = self.data['amount'].sum()
        avg_monthly = self.data.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().mean()
        
        analysis += f"Total Expenses: ${total_spent:,.2f}\n"
        analysis += f"Average Monthly Expenses: ${avg_monthly:,.2f}\n\n"
        
        # Category analysis
        analysis += "Category Breakdown\n"
        analysis += "-" * 20 + "\n"
        category_stats = self.data.groupby('category').agg({
            'amount': ['sum', 'mean', 'count']
        })
        
        for category in self.categories:
            if category in category_stats.index:
                stats = category_stats.loc[category]
                total = stats['amount']['sum']
                avg = stats['amount']['mean']
                count = stats['amount']['count']
                percentage = (total / total_spent) * 100
                
                analysis += f"{category}:\n"
                analysis += f"  Total: ${total:,.2f}\n"
                analysis += f"  Average: ${avg:,.2f}\n"
                analysis += f"  Number of transactions: {count}\n"
                analysis += f"  Percentage of total: {percentage:.1f}%\n\n"
        
        # Trend analysis
        monthly_changes = self.data.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum().pct_change() * 100
        avg_monthly_change = monthly_changes.mean()
        
        analysis += "Trend Analysis\n"
        analysis += "-" * 20 + "\n"
        analysis += f"Average monthly change: {avg_monthly_change:+.1f}%\n"
        
        if len(monthly_changes) >= 3:
            last_3_months = monthly_changes.tail(3)
            analysis += "\nLast 3 months trend:\n"
            for date, change in last_3_months.items():
                analysis += f"{date.strftime('%Y-%m')}: {change:+.1f}%\n"
        
        return analysis

if __name__ == "__main__":
    root = tk.Tk()
    app = ExpensePredictor(root)
    root.mainloop()