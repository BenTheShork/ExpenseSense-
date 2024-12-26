import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ExpenseEntryFrame(ttk.Frame):
    def __init__(self, parent, categories, add_expense_callback):
        super().__init__(parent, padding="10")
        self.categories = categories
        self.add_expense_callback = add_expense_callback
        self.setup_gui()

    def setup_gui(self):
        # Header
        ttk.Label(self, text="Add New Expense", style='Heading.TLabel')\
            .grid(row=0, column=0, columnspan=6, sticky="w")
        
        # Input fields
        ttk.Label(self, text="Amount (€):")\
            .grid(row=1, column=0, padx=(0, 5))
        self.amount_var = tk.StringVar()
        amount_entry = ttk.Entry(self, textvariable=self.amount_var, width=15)
        amount_entry.grid(row=1, column=1, padx=5)
        
        ttk.Label(self, text="Date:").grid(row=1, column=2, padx=5)
        self.date_picker = DateEntry(self, width=12, background='#0078D4',
                                   foreground='white', borderwidth=2,
                                   locale='en_US', date_pattern='yyyy-mm-dd')
        self.date_picker.grid(row=1, column=3, padx=5)
        self.date_picker.set_date(datetime.now())
        
        ttk.Label(self, text="Category:").grid(row=1, column=4, padx=5)
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(self, textvariable=self.category_var,
                                         values=self.categories, width=20)
        self.category_combo.grid(row=1, column=5, padx=5)

        # Add key binding for Enter key
        amount_entry.bind('<Return>', lambda e: self.try_add_expense())
        self.category_combo.bind('<Return>', lambda e: self.try_add_expense())

    def try_add_expense(self):
        """Attempt to add expense and clear inputs on success"""
        expense_data = self.get_expense_data()
        if expense_data:
            self.add_expense_callback(*expense_data)
            self.amount_var.set("")  # Clear amount
            self.date_picker.set_date(datetime.now())  # Reset date to today
            self.category_var.set("")  # Clear category

    def get_expense_data(self):
        try:
            amount = float(self.amount_var.get())
            date = pd.to_datetime(self.date_picker.get_date())
            category = self.category_var.get()
            
            if not category:
                raise ValueError("Please select a category")
                
            return date, amount, category
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return None

class MonthlyOverviewWindow(tk.Toplevel):
    def __init__(self, parent, data, categories):
        super().__init__(parent)
        self.title("Monthly Category Overview")
        self.geometry("1200x800")
        
        self.data = data
        self.categories = categories
        
        # Create main frame with left and right sections
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame for treeview
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right frame for pie chart
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Month selector
        ttk.Label(left_frame, text="Select Month:").pack(anchor=tk.W)
        self.month_var = tk.StringVar()
        periods = pd.to_datetime(self.data['date']).dt.to_period('M')
        month_strings = [str(period) for period in sorted(periods.unique())]
        
        self.combo = ttk.Combobox(left_frame, textvariable=self.month_var,
                                values=month_strings, state='readonly')
        self.combo.pack(anchor=tk.W, pady=(0, 10))
        if month_strings:
            self.combo.set(month_strings[-1])

        # Create treeview with scrollbar
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(tree_frame, 
                                columns=("category", "amount", "percentage"), 
                                show="headings", height=15)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Configure columns
        self.tree.heading("category", text="Category")
        self.tree.heading("amount", text="Amount (€)")
        self.tree.heading("percentage", text="% of Total")
        self.tree.column("category", width=150)
        self.tree.column("amount", width=150, anchor=tk.E)
        self.tree.column("percentage", width=100, anchor=tk.E)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create pie chart
        self.pie_figure, self.pie_ax = plt.subplots(figsize=(8, 8))
        self.pie_canvas = FigureCanvasTkAgg(self.pie_figure, master=right_frame)
        self.pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Summary frame at bottom
        self.summary_frame = ttk.Frame(left_frame)
        self.summary_frame.pack(fill=tk.X, pady=(10, 0))
        self.summary_label = ttk.Label(self.summary_frame, text="")
        self.summary_label.pack(anchor=tk.W)

        # Bind update functions to month selection
        self.month_var.trace('w', self.update_view)
        
        # Initial update
        self.update_view()
    def update_view(self, *args):
        selected_month = self.month_var.get()
        if not selected_month:
            return

        self.tree.delete(*self.tree.get_children())
        self.pie_ax.clear()
        
        # Convert selected month to period
        selected_period = pd.Period(selected_month)
        month_data = self.data[pd.to_datetime(self.data['date']).dt.to_period('M') == selected_period]
        
        total = month_data['amount'].sum()
        by_category = month_data.groupby('category')['amount'].sum().sort_values(ascending=False)

        # Update tree
        for category in self.categories:
            amount = by_category.get(category, 0)
            percentage = (amount / total * 100) if total > 0 else 0
            if amount > 0:  # Only show categories with expenses
                self.tree.insert("", tk.END, values=(
                    category,
                    f"€{amount:,.2f}",
                    f"{percentage:.1f}%"
                ))

        # Update pie chart
        non_zero_categories = by_category[by_category > 0]
        if not non_zero_categories.empty:
            wedges, texts, autotexts = self.pie_ax.pie(
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
            
            self.pie_ax.set_title(f'Expense Distribution - {selected_month}')
        
        self.pie_figure.tight_layout()
        self.pie_canvas.draw()

        # Update summary
        transactions = len(month_data)
        self.summary_label.config(
            text=f"Total: €{total:,.2f} | Transactions: {transactions}"
        )

class PredictionSummaryWindow(tk.Toplevel):
    def __init__(self, parent, predictions, confidence_scores=None):
        super().__init__(parent)
        self.title("Prediction Summary")
        self.geometry("400x500")
        
        self.predictions = predictions
        self.confidence_scores = confidence_scores
        self.setup_gui()

    def setup_gui(self):
        text_widget = tk.Text(self, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        text_widget.insert(tk.END, "Predicted Monthly Expenses by Category:\n\n")
        
        for category in self.predictions.columns:
            avg_prediction = self.predictions[category].mean()
            text_widget.insert(tk.END, f"{category}: €{avg_prediction:.2f}\n")
            if self.confidence_scores:
                confidence = self.confidence_scores.get(category, 0)
                text_widget.insert(tk.END, f"Confidence: {confidence:.1f}%\n\n")
        
        total_avg = self.predictions.sum(axis=1).mean()
        text_widget.insert(tk.END, f"\nTotal Monthly Average: €{total_avg:.2f}")
        text_widget.config(state=tk.DISABLED)

class PredictionDetailWindow(tk.Toplevel):
    def __init__(self, parent, date, data, is_prediction=False):
        super().__init__(parent)
        self.title(f"{'Predicted' if is_prediction else 'Historical'} Data Details")
        self.geometry("600x800")
        
        # Convert date to pandas Timestamp if it isn't already
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        
        # Create main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = f"{'Predicted' if is_prediction else 'Historical'} Expenses for {date.strftime('%B %Y')}"
        ttk.Label(main_frame, text=title, style='Heading.TLabel').pack(pady=(0, 10))
        
        # Create treeview
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        tree = ttk.Treeview(tree_frame, columns=("category", "amount", "percentage"), 
                           show="headings", height=15)
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Configure columns
        tree.heading("category", text="Category")
        tree.heading("amount", text="Amount (€)")
        tree.heading("percentage", text="% of Total")
        tree.column("category", width=200)
        tree.column("amount", width=150, anchor=tk.E)
        tree.column("percentage", width=100, anchor=tk.E)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Process and display data
        if is_prediction:
            # For prediction data (dictionary format)
            total = sum(data.values())
            for category, amount in data.items():
                percentage = (amount / total * 100) if total > 0 else 0
                tree.insert("", tk.END, values=(
                    category,
                    f"€{amount:,.2f}",
                    f"{percentage:.1f}%"
                ))
        else:
            # For historical data (DataFrame format)
            # Filter data for the specific month
            month_data = data[data['date'].dt.to_period('M') == date.to_period('M')]
            by_category = month_data.groupby('category')['amount'].sum()
            total = by_category.sum()
            
            for category, amount in by_category.items():
                percentage = (amount / total * 100) if total > 0 else 0
                tree.insert("", tk.END, values=(
                    category,
                    f"€{amount:,.2f}",
                    f"{percentage:.1f}%"
                ))
        
        # Add total at bottom
        summary_frame = ttk.Frame(main_frame)
        summary_frame.pack(fill=tk.X, pady=(10, 0))
        total_amount = total if total > 0 else 0
        ttk.Label(summary_frame, 
                 text=f"Total: €{total_amount:,.2f}",
                 style='Heading.TLabel').pack(anchor=tk.W)
        
        # Add note for predictions
        if is_prediction:
            note_frame = ttk.Frame(main_frame)
            note_frame.pack(fill=tk.X, pady=(10, 0))
            ttk.Label(note_frame,
                     text="Note: These are predicted values based on historical patterns",
                     foreground='gray').pack(anchor=tk.W)
        
        # Add close button
        ttk.Button(main_frame, text="Close", command=self.destroy).pack(pady=(10, 0))