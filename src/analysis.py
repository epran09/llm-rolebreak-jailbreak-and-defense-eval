import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import math


df = pd.read_csv('human_validation_data.csv')
df_clean = df.dropna(subset=['human_label', 'llamaguard3_label'])
df_clean['human_label'] = df_clean['human_label'].astype(int)
df_clean['llamaguard3_label'] = df_clean['llamaguard3_label'].astype(int)

model_names = df_clean['model'].unique()
n_models = len(model_names)
cols = 3  # customize as needed!
rows = math.ceil(n_models / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))

# Flatten axes array for easy iteration
axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

for idx, model in enumerate(model_names):
    sub = df_clean[df_clean['model'] == model]
    y_true = sub['human_label']
    y_pred = sub['llamaguard3_label']
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Unsafe"])
    disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
    axes[idx].set_title(model)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

# Hide any empty subplots
for j in range(idx+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("all_models_confusion_matrices.png", dpi=200)
plt.show()
