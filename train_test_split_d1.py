import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import json

# 1. Load the TF-IDF matrix and labels
X = load_npz('Data/text_tfidf_matrix.npz')
df = pd.read_csv('Data/best_cleaned.csv')
y = df['label'].values

# 2. Train-Test Split with Stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3. Define Cross-Validation Strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Define Model
model = LogisticRegression(class_weight='balanced', max_iter=1000)

# 5. Cross-Validation on Training Set
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
print(f"Cross-Validation F1 Macro Avg: {scores.mean():.4f}")
print(f"Cross-Validation F1 Scores: {scores}")

# Save Cross-Validation F1 Scores to CSV
fold_df = pd.DataFrame({
    'Fold': [f'Fold {i+1}' for i in range(len(scores))],
    'F1_Macro_Score': scores
})
fold_df.to_csv('Data/f1_macro_scores.csv', index=False)

# 6. Train on Full Training Set
model.fit(X_train, y_train)

# 7. Final Evaluation on Test Set
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
print("\nTest Set Evaluation:")
print(classification_report(y_test, y_pred))

# Save Classification Report to CSV
# CSV
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('Data/classification_report.csv')
