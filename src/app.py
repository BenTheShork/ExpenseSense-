# src/app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

from utils.data_handler import DataHandler
from utils.ml_models import ExpensePredictor
from utils.visualizations import Visualizer
from gui.components import (
    ExpenseEntryFrame,
    MonthlyOverviewWindow,
    PredictionSummaryWindow
)
from gui.components import PredictionDetailWindow

class ExpensePredictorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced Expense Predictor")
        self.root.geometry("1800x1200")
        self.detail_window = None
        # Define categories
        self.categories = [
            'Housing', 'Transportation', 'Food', 'Utilities', 
            'Healthcare', 'Insurance', 'Entertainment', 'Shopping',
            'Education', 'Savings', 'Debt Payments', 'Other'
        ]
        
        # Initialize components
        self.data_handler = DataHandler()
        self.ml_predictor = ExpensePredictor(self.categories)
        self.visualizer = Visualizer()
        
        # Load data
        self.data = self.data_handler.load_data()
        
        # Setup GUI
        self.setup_gui()
        self.update_plots()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        # Style configuration
        self.setup_styles()
        
        # Main container
        main_container = ttk.Frame(self.root, padding="20")
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Expense entry section
        self.expense_entry = ExpenseEntryFrame(
            main_container,
            self.categories,
            self.add_expense
        )
        self.expense_entry.grid(row=0, column=0, sticky="ew")
        
        # Button frame
        self.setup_buttons()
        
        # Notebook for plots
        self.setup_notebook(main_container)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def setup_styles(self):
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', 
                       font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10), 
                       padding=5)
        style.configure('Heading.TLabel',
                       font=('Segoe UI', 12, 'bold'),
                       padding=10)

    def setup_buttons(self):
        buttons_frame = ttk.Frame(self.expense_entry, padding="10")
        buttons_frame.grid(row=2, column=0, columnspan=6)
        
        # First row of buttons
        main_buttons_frame = ttk.Frame(buttons_frame)
        main_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(main_buttons_frame, text="Add Expense",
                  command=self.add_expense).pack(side=tk.LEFT, padx=5)
        ttk.Button(main_buttons_frame, text="Import CSV",
                  command=self.import_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(main_buttons_frame, text="Export CSV",
                  command=self.export_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(main_buttons_frame, text="Monthly Overview",
                  command=self.show_monthly_overview).pack(side=tk.LEFT, padx=5)
        
        # Second row with prediction controls
        pred_frame = ttk.Frame(buttons_frame)
        pred_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(pred_frame, text="Months to predict:").pack(side=tk.LEFT, padx=5)
        self.prediction_months = tk.StringVar(value="3")
        spinbox = ttk.Spinbox(pred_frame, from_=1, to=24,
                             width=5,
                             textvariable=self.prediction_months)
        spinbox.pack(side=tk.LEFT, padx=5)
        ttk.Button(pred_frame, text="Predict",
                  command=self.predict_expenses).pack(side=tk.LEFT, padx=5)

    def setup_notebook(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=2, column=0, sticky="nsew", pady=10)
        
        # Create frames
        self.trends_frame = ttk.Frame(self.notebook)
        self.category_frame = ttk.Frame(self.notebook)
        self.prediction_frame = ttk.Frame(self.notebook)
        
        # Add frames to notebook
        self.notebook.add(self.trends_frame, text='Trends')
        self.notebook.add(self.category_frame, text='Categories')
        self.notebook.add(self.prediction_frame, text='Predictions')
        
        # Create plots
        self.setup_plots()

    def setup_plots(self):
        # Set up style
        self.visualizer.setup_style()
        
        # Create figures
        self.trends_figure, self.trends_ax, self.trends_canvas = \
            self.visualizer.create_figure(self.trends_frame)
        self.category_figure, self.category_ax, self.category_canvas = \
            self.visualizer.create_figure(self.category_frame)
        self.prediction_figure, self.prediction_ax, self.prediction_canvas = \
            self.visualizer.create_figure(self.prediction_frame)

    def update_plots(self):
        if self.data.empty:
            return
        
        # Update trends plot
        monthly_total = self.data_handler.get_monthly_totals()
        self.visualizer.plot_trends(self.trends_ax, monthly_total)
        self.trends_figure.tight_layout()
        self.trends_canvas.draw()
        
        # Update category plot
        category_totals = self.data_handler.get_category_totals()
        self.visualizer.plot_categories(self.category_ax, category_totals)
        self.category_figure.tight_layout()
        self.category_canvas.draw()

    def add_expense(self):
        expense_data = self.expense_entry.get_expense_data()
        if expense_data:
            date, amount, category = expense_data
            if self.data_handler.add_expense(date, amount, category):
                self.update_plots()

    def predict_expenses(self):
        success, confidence_scores = \
            self.ml_predictor.train_models(self.data)
        
        if not success:
            messagebox.showerror(
                "Error",
                "Not enough data for prediction (minimum 6 months required)"
            )
            return
            
        try:
            months_ahead = int(self.prediction_months.get())
            predictions = self.ml_predictor.predict_future_expenses(
                self.data['date'].max(),
                months_ahead
            )
            
            # Plot predictions with raw data
            historical = self.data.groupby(
                pd.Grouper(key='date', freq='M'))['amount'].sum().reset_index()
            
            self.visualizer.plot_predictions(
                self.prediction_ax,
                historical,
                predictions,
                self.data,  # Pass the raw data
                self.show_prediction_detail
            )
            self.prediction_figure.tight_layout()
            self.prediction_canvas.draw()
            
            # Show summary
            PredictionSummaryWindow(
                self.root,
                predictions,
                confidence_scores
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction: {str(e)}")

    def show_prediction_detail(self, date, data, is_prediction):
        """Show detailed view of clicked data point"""
        if self.detail_window is not None and self.detail_window.winfo_exists():
            self.detail_window.destroy()
            
        self.detail_window = PredictionDetailWindow(self.root, date, data, is_prediction)

    def show_monthly_overview(self):
        if self.data.empty:
            messagebox.showerror("Error", "No data available")
            return
        MonthlyOverviewWindow(self.root, self.data, self.categories)

    def import_csv(self):
        filename = filedialog.askopenfilename(
            title="Import CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        if filename:
            success, message = self.data_handler.import_csv(filename)
            if success:
                self.data = self.data_handler.data
                self.update_plots()
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)

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
            success, message = self.data_handler.export_csv(filename)
            if success:
                messagebox.showinfo("Success", message)
            else:
                messagebox.showerror("Error", message)

    def on_closing(self):
        """Handle window close event"""
        self.data_handler.save_data()
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = ExpensePredictorApp()
    app.run()