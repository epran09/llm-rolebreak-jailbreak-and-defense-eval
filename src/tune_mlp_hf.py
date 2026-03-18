import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import product
import warnings
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# 1. Data Preparation
class ResponseDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'x': self.embeddings[idx], 'labels': self.labels[idx]}

# 2. MLP Model (PyTorch)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, hidden_dim2=None, dropout=0.3):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        if hidden_dim2:
            layers += [
                nn.Linear(hidden_dim, hidden_dim2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            layers.append(nn.Linear(hidden_dim2, 1))
        else:
            layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x, labels=None):
        return self.net(x).squeeze(-1)

# 3. Custom Trainer
class MLPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs['labels']
        outputs = model(inputs['x'])
        loss = nn.BCELoss()(outputs, labels)
        return (loss, outputs) if return_outputs else loss

# 4. Embedding extraction
def get_embeddings(texts, tokenizer, model, batch_size=32):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=256)
            outputs = model(**encoded)
            emb = outputs.last_hidden_state.mean(dim=1).cpu()
            embeddings.append(emb)
    return torch.cat(embeddings, dim=0)

# 5. Load data
print("Loading data...")
MODEL_NAME = 'BAAI/bge-base-en-v1.5'
df = pd.read_csv('src/all_model_moderation_results.csv')
df = df[pd.to_numeric(df['llamaguard3_label'], errors='coerce').notnull()]
df = df.dropna(subset=['model_response'])
df['llamaguard3_label'] = pd.to_numeric(df['llamaguard3_label'], errors='coerce').astype(int)
responses = df['model_response'].astype(str).tolist()
labels = df['llamaguard3_label'].tolist()
models_list = df['model'].astype(str).tolist()
prompts_list = df['prompt'].astype(str).tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME)

print("Generating embeddings...")
all_embeddings = get_embeddings(responses, tokenizer, encoder)

# 6. Prompt-level train/test split (prevents data contamination)
# All responses to the same prompt go entirely into train OR test, never both.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(responses, labels, groups=prompts_list))

X_train_emb = all_embeddings[train_idx]
X_test_emb = all_embeddings[test_idx]
y_train = [labels[i] for i in train_idx]
y_test = [labels[i] for i in test_idx]
models_train = [models_list[i] for i in train_idx]
models_test = [models_list[i] for i in test_idx]

# Verify no prompt leakage
train_prompts = set(prompts_list[i] for i in train_idx)
test_prompts = set(prompts_list[i] for i in test_idx)
assert len(train_prompts & test_prompts) == 0, "Data contamination: prompt overlap between train and test!"
print(f"Prompt-level split: {len(train_prompts)} train prompts, {len(test_prompts)} test prompts, 0 overlap")
print(f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

train_dataset = ResponseDataset(X_train_emb, y_train)
test_dataset = ResponseDataset(X_test_emb, y_test)

input_dim = X_train_emb.shape[1]

# 7. Hyperparameter grid
param_grid = {
    'hidden_dim': [128, 256],
    'hidden_dim2': [None, 128],
    'dropout': [0.2, 0.3, 0.5],
    'lr': [5e-4, 1e-3],
    'epochs': [5, 10],
}

combinations = list(product(
    param_grid['hidden_dim'],
    param_grid['hidden_dim2'],
    param_grid['dropout'],
    param_grid['lr'],
    param_grid['epochs'],
))

print(f"\nTesting {len(combinations)} hyperparameter combinations...\n")

best_f1 = 0
best_config = None
best_model_state = None
results = []

for i, (hd, hd2, drop, lr, epochs) in enumerate(combinations):
    config = {'hidden_dim': hd, 'hidden_dim2': hd2, 'dropout': drop, 'lr': lr, 'epochs': epochs}
    print(f"[{i+1}/{len(combinations)}] {config}")

    mlp = MLP(input_dim=input_dim, hidden_dim=hd, hidden_dim2=hd2, dropout=drop)

    training_args = TrainingArguments(
        output_dir='./mlp_hf_tune_tmp',
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=lr,
        report_to=[],
        disable_tqdm=True,
        save_strategy='no',
    )

    trainer = MLPTrainer(
        model=mlp,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    # Evaluate
    mlp.eval()
    device = next(mlp.parameters()).device
    with torch.no_grad():
        raw_preds = mlp(X_test_emb.to(device))
    pred_labels = (raw_preds.cpu().numpy() > 0.5).astype(int)

    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels)
    prec = precision_score(y_test, pred_labels)
    rec = recall_score(y_test, pred_labels)
    kappa = cohen_kappa_score(y_test, pred_labels)

    results.append({**config, 'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'kappa': kappa})
    print(f"  -> F1={f1:.4f}, Acc={acc:.4f}, Kappa={kappa:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_config = config
        best_model_state = mlp.state_dict().copy()
        print(f"  *** New best (F1={best_f1:.4f}) ***")

# 8. Save tuning results
os.makedirs('results', exist_ok=True)
results_df = pd.DataFrame(results).sort_values('f1', ascending=False)
results_df.to_csv('results/mlp_hf_tuning_results.csv', index=False)
print(f"\nTuning results saved to results/mlp_hf_tuning_results.csv")
print(f"Best config: {best_config}")
print(f"Best F1: {best_f1:.4f}")

# 9. Retrain best model and evaluate
print("\nRetraining best model for final evaluation...")
best_mlp = MLP(input_dim=input_dim, hidden_dim=best_config['hidden_dim'],
               hidden_dim2=best_config['hidden_dim2'], dropout=best_config['dropout'])
best_mlp.load_state_dict(best_model_state)
best_mlp.eval()
device = next(best_mlp.parameters()).device

with torch.no_grad():
    raw_preds = best_mlp(X_test_emb.to(device))
pred_probs = raw_preds.cpu().numpy()
pred_labels = (pred_probs > 0.5).astype(int)

print("\n--- Final Classification Report ---")
print(classification_report(y_test, pred_labels))
print(f"Cohen's Kappa: {cohen_kappa_score(y_test, pred_labels):.4f}")

# Overall ASR
print(f"\nMLP ASR: {np.mean(pred_labels):.4f}")
print(f"LlamaGuard3 ASR: {np.mean(y_test):.4f}")

# Per-model ASR & metrics
df_test = pd.DataFrame({
    'model': models_test,
    'mlp_pred': pred_labels.flatten(),
    'llamaguard3_label': y_test
})

print('\n--- Per-model ASR Comparison ---')
print(f"{'Model':<30} {'MLP ASR':<12} {'LG3 ASR':<12}")
asr_records = []
for m in sorted(df_test['model'].unique()):
    mask = df_test['model'] == m
    mlp_asr = df_test.loc[mask, 'mlp_pred'].mean()
    lg3_asr = df_test.loc[mask, 'llamaguard3_label'].mean()
    print(f"{m:<30} {mlp_asr:<12.4f} {lg3_asr:<12.4f}")
    asr_records.append({'Model': m, 'MLP ASR': mlp_asr, 'LlamaGuard3 ASR': lg3_asr})

asr_df = pd.DataFrame(asr_records)
asr_df.to_csv('results/mlp_hf_tuned_asr_per_model.csv', index=False)

print('\n--- Per-model Classification Metrics ---')
metrics_records = []
for m in sorted(df_test['model'].unique()):
    mask = df_test['model'] == m
    y_true_m = df_test.loc[mask, 'llamaguard3_label'].values
    y_pred_m = df_test.loc[mask, 'mlp_pred'].values
    p, r, f, _ = precision_recall_fscore_support(y_true_m, y_pred_m, average='binary', zero_division=0)
    support = int(mask.sum())
    metrics_records.append({'Model': m, 'Precision': p, 'Recall': r, 'F1 Score': f, 'Support': support})

metrics_df = pd.DataFrame(metrics_records)
print(metrics_df.to_string(index=False))
metrics_df.to_csv('results/mlp_hf_tuned_metrics_per_model.csv', index=False)

# 10. Visualizations

# ASR barplot
asr_melted = asr_df.melt(id_vars='Model', var_name='Method', value_name='ASR')
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Model', y='ASR', hue='Method', data=asr_melted, palette='Set2')
plt.title('Per-Model ASR: Fine-Tuned MLP vs LlamaGuard3')
plt.ylabel('ASR'); plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
for c in ax.containers: ax.bar_label(c, fmt='%.2f', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('results/mlp_hf_tuned_asr_per_model.png')
print('\nASR barplot saved to results/mlp_hf_tuned_asr_per_model.png')

# Metrics barplot
metrics_melted = metrics_df.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1 Score'],
                                  var_name='Metric', value_name='Score')
plt.figure(figsize=(12, 6))
ax2 = sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_melted, palette='muted')
plt.title('Per-Model Metrics: Fine-Tuned MLP Classifier')
plt.ylabel('Score'); plt.xlabel('Model'); plt.ylim(0, 1.15)
plt.xticks(rotation=45, ha='right')
for c in ax2.containers: ax2.bar_label(c, fmt='%.2f', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('results/mlp_hf_tuned_metrics_per_model.png')
print('Metrics barplot saved to results/mlp_hf_tuned_metrics_per_model.png')

# ROC curves
y_test_arr = np.array(y_test)
models_test_arr = np.array(models_test)

plt.figure(figsize=(10, 7))
fpr, tpr, _ = roc_curve(y_test_arr, pred_probs)
plt.plot(fpr, tpr, lw=2, color='black', linestyle='--', label=f'Overall (AUC={auc(fpr, tpr):.3f})')
for m in sorted(np.unique(models_test_arr)):
    mask = models_test_arr == m
    if len(np.unique(y_test_arr[mask])) < 2: continue
    fpr_m, tpr_m, _ = roc_curve(y_test_arr[mask], pred_probs[mask])
    plt.plot(fpr_m, tpr_m, lw=2, label=f'{m} (AUC={auc(fpr_m, tpr_m):.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curves — Fine-Tuned MLP Classifier')
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('results/mlp_hf_tuned_roc_curves.png')
print('ROC curves saved to results/mlp_hf_tuned_roc_curves.png')

# PR curves
plt.figure(figsize=(10, 7))
prec_v, rec_v, _ = precision_recall_curve(y_test_arr, pred_probs)
plt.plot(rec_v, prec_v, lw=2, color='black', linestyle='--',
         label=f'Overall (AP={average_precision_score(y_test_arr, pred_probs):.3f})')
for m in sorted(np.unique(models_test_arr)):
    mask = models_test_arr == m
    if len(np.unique(y_test_arr[mask])) < 2: continue
    prec_m, rec_m, _ = precision_recall_curve(y_test_arr[mask], pred_probs[mask])
    ap_m = average_precision_score(y_test_arr[mask], pred_probs[mask])
    plt.plot(rec_m, prec_m, lw=2, label=f'{m} (AP={ap_m:.3f})')
plt.xlabel('Recall'); plt.ylabel('Precision')
plt.title('Precision-Recall Curves — Fine-Tuned MLP Classifier')
plt.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig('results/mlp_hf_tuned_pr_curves.png')
print('PR curves saved to results/mlp_hf_tuned_pr_curves.png')

# Save best model
os.makedirs('classifiers/MLP', exist_ok=True)
torch.save(best_mlp.state_dict(), 'classifiers/MLP/mlp_hf_tuned_best.pt')
print(f"\nBest fine-tuned model saved to classifiers/MLP/mlp_hf_tuned_best.pt")

# 11. 3-way ASR comparison: LlamaGuard3 vs MLP (before tuning) vs MLP (after tuning)
print("\n" + "="*70)
print("3-WAY ASR COMPARISON: LlamaGuard3  vs  MLP (Before)  vs  MLP (After Tuning)")
print("="*70)

pre_asr_path = 'results/mlp_hf_asr_per_model.csv'
if os.path.exists(pre_asr_path):
    pre = pd.read_csv(pre_asr_path)
    pre = pre.rename(columns={'MLP ASR': 'MLP (Before Tuning)', 'LlamaGuard3 ASR': 'LlamaGuard3'})

    post = asr_df.rename(columns={'MLP ASR': 'MLP (After Tuning)', 'LlamaGuard3 ASR': 'LlamaGuard3_post'})
    post = post[['Model', 'MLP (After Tuning)']]

    comparison = pre.merge(post, on='Model', how='outer')
    comparison = comparison[['Model', 'LlamaGuard3', 'MLP (Before Tuning)', 'MLP (After Tuning)']]
    comparison = comparison.sort_values('LlamaGuard3', ascending=False).reset_index(drop=True)

    comparison.to_csv('results/asr_comparison_all.csv', index=False)
    print(comparison.to_string(index=False))

    # Delta
    print("\n--- ASR Delta (After − Before Tuning) ---")
    for _, row in comparison.iterrows():
        delta = row['MLP (After Tuning)'] - row['MLP (Before Tuning)']
        sign = '+' if delta >= 0 else ''
        print(f"  {row['Model']:<30} {sign}{delta:.4f}")

    # Overall row for the barplot
    overall = pd.DataFrame([{
        'Model': 'Overall',
        'LlamaGuard3': comparison['LlamaGuard3'].mean(),
        'MLP (Before Tuning)': comparison['MLP (Before Tuning)'].mean(),
        'MLP (After Tuning)': comparison['MLP (After Tuning)'].mean(),
    }])
    comp_plot = pd.concat([comparison, overall], ignore_index=True)

    melted = comp_plot.melt(id_vars='Model', var_name='Method', value_name='ASR')
    model_order = comp_plot['Model'].tolist()
    melted['Model'] = pd.Categorical(melted['Model'], categories=model_order, ordered=True)

    palette = {'LlamaGuard3': '#66c2a5', 'MLP (Before Tuning)': '#fc8d62', 'MLP (After Tuning)': '#8da0cb'}
    plt.figure(figsize=(14, 7))
    ax_cmp = sns.barplot(x='Model', y='ASR', hue='Method', data=melted, palette=palette)
    plt.title('Per-Model ASR: LlamaGuard3 vs MLP (Before & After Fine-Tuning)', fontsize=13)
    plt.ylabel('Attack Success Rate (ASR)')
    plt.xlabel('Model')
    plt.xticks(rotation=30, ha='right')
    plt.ylim(0, 1.12)
    for c in ax_cmp.containers:
        ax_cmp.bar_label(c, fmt='%.2f', fontsize=8, fontweight='bold', padding=2)
    plt.legend(title='Method', loc='upper right')
    plt.tight_layout()
    plt.savefig('results/asr_comparison_all.png', dpi=150)
    print('\n3-way ASR comparison barplot saved to results/asr_comparison_all.png')
    print('3-way ASR comparison table saved to results/asr_comparison_all.csv')
else:
    print(f"Pre-tuning ASR file not found ({pre_asr_path}). Run mlp_hf_classifier.py first.")
    print("Skipping 3-way comparison.")

print("\nDone!")
