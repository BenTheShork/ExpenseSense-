import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Visualizer:
    @staticmethod
    def setup_style():
        """Set up plotting style"""
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 12
        })

    @staticmethod
    def create_figure(master, figsize=(15, 8)):
        """Create a new figure with canvas"""
        figure, ax = plt.subplots(figsize=figsize)
        canvas = FigureCanvasTkAgg(figure, master=master)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        return figure, ax, canvas

    @staticmethod
    def plot_trends(ax, monthly_total):
        """Plot monthly trends"""
        ax.clear()
        ax.plot(monthly_total.index, monthly_total.values, 
                marker='o', linewidth=2)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Total Expenses (€)', fontsize=12)
        ax.set_title('Monthly Expense Trends', fontsize=14, pad=20)
        ax.tick_params(axis='x', rotation=45)

    @staticmethod
    def plot_categories(ax, category_totals):
        """Plot category breakdown"""
        ax.clear()
        category_totals.plot(kind='barh', ax=ax)
        ax.grid(True, linestyle='--', alpha=0.7, axis='x')
        ax.set_title('Total Expenses by Category', fontsize=14, pad=20)
        ax.set_xlabel('Total Amount (€)', fontsize=12)
        
        for i, v in enumerate(category_totals):
            ax.text(v, i, f'€{v:,.0f}', 
                   fontsize=10, va='center', fontweight='bold')

    @staticmethod
    def plot_predictions(ax, historical, predictions, raw_data, click_callback=None):
        """Plot predictions with historical data and click handling"""
        ax.clear()
        
        # Plot historical data
        hist_line = ax.plot(historical['date'], historical['amount'],
                          label='Historical', color='blue', 
                          marker='o', linewidth=2, picker=True, pickradius=5)[0]
        
        # Plot predictions
        total_predictions = predictions.sum(axis=1)
        pred_line = ax.plot(predictions.index, total_predictions,
                          label='Predicted', color='red', 
                          linestyle='--', marker='s', 
                          linewidth=2, markersize=8, picker=True, pickradius=5)[0]
        
        if click_callback:
            # Store last click time to prevent multiple rapid clicks
            last_click = {'time': 0}
            
            def on_pick(event):
                current_time = time.time()
                if current_time - last_click['time'] < 0.5:  # 500ms debounce
                    return
                last_click['time'] = current_time
                
                artist = event.artist
                ind = event.ind[0]
                
                if artist == hist_line:
                    date = historical['date'].iloc[ind]
                    # Pass the raw data for proper historical view
                    click_callback(date, raw_data, False)
                else:
                    try:
                        xdata = artist.get_xdata()
                        clicked_date = pd.Timestamp(xdata[ind])
                        closest_date = predictions.index[predictions.index.get_indexer([clicked_date], method='nearest')[0]]
                        data_dict = {cat: predictions.loc[closest_date, cat] 
                                   for cat in predictions.columns}
                        click_callback(closest_date, data_dict, True)
                    except Exception as e:
                        print(f"Error processing prediction click: {str(e)}")
            
            # Remove any existing pick event handlers
            for cid in ax.figure.canvas.callbacks.callbacks.get(event_name := 'pick_event', {}).copy():
                ax.figure.canvas.mpl_disconnect(cid)
            
            # Add new handler
            ax.figure.canvas.mpl_connect('pick_event', on_pick)
        
    @staticmethod
    def plot_monthly_pie(ax, data, selected_month):
        """Plot pie chart for monthly expenses"""
        ax.clear()
        
        by_category = data.groupby('category')['amount'].sum()
        non_zero_categories = by_category[by_category > 0]
        
        if not non_zero_categories.empty:
            wedges, texts, autotexts = ax.pie(
                non_zero_categories.values,
                labels=non_zero_categories.index,
                autopct='%1.1f%%',
                startangle=90
            )
            
            # Make labels more readable
            for autotext in autotexts:
                autotext.set_fontsize(8)
            for text in texts:
                text.set_fontsize(8)
            
            ax.set_title(f'Expense Distribution - {selected_month}')