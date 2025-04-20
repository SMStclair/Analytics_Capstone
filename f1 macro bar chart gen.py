import pandas as pd
import matplotlib.pyplot as plt

# Load F1 Macro Scores CSV
f1_scores = pd.read_csv('Data/f1_macro_scores.csv')

# Plot
plt.figure(figsize=(8, 5))
plt.bar(f1_scores['Fold'], f1_scores['F1_Macro_Score'], color='skyblue')
plt.ylim(0.94, 0.96)
plt.title('Cross-Validation F1 Macro Scores')
plt.ylabel('F1 Macro Scores')
plt.xlabel('Folds')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save chart
plt.tight_layout()
plt.savefig('Data/f1_macro_scores_bar.png')
plt.show()
