import pandas as pd
import matplotlib.pyplot as plt

# Load the classification report
report_df = pd.read_csv('Data/classification_report.csv', index_col=0)

# Move the index into a column for easy filtering
report_df.reset_index(inplace=True)
report_df.rename(columns={'index': 'class'}, inplace=True)

# Now filter rows where class is '0' or '1'
class_metrics = report_df[report_df['class'].isin(['0', '1'])].copy()

# Metrics to plot
metrics = ['precision', 'recall', 'f1-score']
x_labels = ['Fake (0)', 'Real (1)']
x_pos = range(len(x_labels))
bar_width = 0.25

# Plotting
plt.figure(figsize=(9, 6))
for i, metric in enumerate(metrics):
    plt.bar(
        [p + i * bar_width for p in x_pos],
        class_metrics[metric].astype(float),
        width=bar_width,
        label=metric.capitalize()
    )

# Formatting
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Classification Metrics by Class')
plt.xticks([p + bar_width for p in x_pos], x_labels)
plt.ylim(0.9, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save and show
plt.tight_layout()
plt.savefig('Data/classification_metrics_by_class.png')
plt.show()
