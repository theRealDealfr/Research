import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics():
    csv_path = 'results_controlled_baselines/phase2_baseline_summary.csv'
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Make sure you run this script from the Research folder.")
        return
    
    # Simplifying the model names for the plot
    models = ['Frozen WavLM', 'WavLM + Age', 'WavLM + AAFF']
    
    # 4 Key metrics we want to visualize and compare
    metrics = {
        'Test Accuracy': df['Test Acc'].values,
        'Test Macro F1': df['Test Macro F1'].values,
        'Male Macro F1': df['Male Macro F1'].values,
        'Female Macro F1': df['Female Macro F1'].values
    }
    
    x = np.arange(len(models))
    width = 0.2  # Width of the bars
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting each metric grouped by model
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = width * i
        bars = ax.bar(x + offset, values, width, label=metric_name)
        # Add exact values on top of the bars for clarity
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)

    # Adding labels and title
    ax.set_ylabel('Performance Score (0 to 1)')
    ax.set_title('Model Performance Comparison (Baseline vs Age vs AAFF)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Set y-axis limits to zoom in on the differences (since all are above 0.8)
    ax.set_ylim(0.75, 1.0)
    
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300)
    print("Plot successfully saved as 'metrics_comparison.png'")
    plt.show()

if __name__ == "__main__":
    plot_metrics()
