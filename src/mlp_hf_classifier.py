import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report


# 1. Data Preparation
class ResponseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, model):
        self.embeddings = self.get_embeddings(texts, tokenizer, model)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    @staticmethod
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'x': self.embeddings[idx], 'labels': self.labels[idx]}

# 2. MLP Model (PyTorch)
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels=None):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x).squeeze(-1)

# 3. Load data and transformer
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
model = AutoModel.from_pretrained(MODEL_NAME)

# 4. Prompt-level train/test split (prevents data contamination)
# All responses to the same prompt go entirely into train OR test, never both.
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(responses, labels, groups=prompts_list))

X_train = [responses[i] for i in train_idx]
X_test = [responses[i] for i in test_idx]
y_train = [labels[i] for i in train_idx]
y_test = [labels[i] for i in test_idx]
models_train = [models_list[i] for i in train_idx]
models_test = [models_list[i] for i in test_idx]

# Verify no prompt leakage
train_prompts = set(prompts_list[i] for i in train_idx)
test_prompts = set(prompts_list[i] for i in test_idx)
assert len(train_prompts & test_prompts) == 0, "Data contamination: prompt overlap between train and test!"
print(f"Prompt-level split: {len(train_prompts)} train prompts, {len(test_prompts)} test prompts, 0 overlap")
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

train_dataset = ResponseDataset(X_train, y_train, tokenizer, model)
test_dataset = ResponseDataset(X_test, y_test, tokenizer, model)

# 5. Trainer setup
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

class HuggingFaceMLPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs['labels']
        outputs = model(inputs['x'])
        loss_fct = nn.BCELoss()
        loss = loss_fct(outputs, labels)
        return (loss, outputs) if return_outputs else loss

input_dim = train_dataset.embeddings.shape[1]
mlp_model = MLP(input_dim=input_dim, hidden_dim=256, dropout=0.3)

training_args = TrainingArguments(
    output_dir='./mlp_hf_results',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_dir='./mlp_hf_logs',
    report_to=[],
    disable_tqdm=False,
)

trainer = HuggingFaceMLPTrainer(
    model=mlp_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. Evaluation — run inference directly to avoid Trainer batch misalignment
mlp_model.eval()
device = next(mlp_model.parameters()).device
with torch.no_grad():
    raw_preds = mlp_model(test_dataset.embeddings.to(device))
pred_labels = (raw_preds.cpu().numpy() > 0.5).astype(int)

from sklearn.metrics import precision_score, recall_score, f1_score
print(classification_report(y_test, pred_labels))

# Calculate ASR for MLP classifier
asr_mlp = np.mean(pred_labels)
print(f"MLP ASR (Attack Success Rate): {asr_mlp:.4f}")

# Calculate LlamaGuard3 ASR for comparison (ground truth labels)
asr_llamaguard3 = np.mean(y_test)
print(f"LlamaGuard3 ASR (ground truth): {asr_llamaguard3:.4f}")

# Print precision, recall, F1 for MLP classifier
precision = precision_score(y_test, pred_labels)
recall = recall_score(y_test, pred_labels)
f1 = f1_score(y_test, pred_labels)
print(f"MLP Precision: {precision:.4f}")
print(f"MLP Recall: {recall:.4f}")
print(f"MLP F1 Score: {f1:.4f}")

# 7. Per-model ASR comparison
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_test = pd.DataFrame({
    'model': models_test,
    'mlp_pred': pred_labels.flatten(),
    'llamaguard3_label': y_test
})

print('\n--- Per-model ASR Comparison (MLP vs LlamaGuard3) ---')
print(f"{'Model':<30} {'MLP ASR':<12} {'LG3 ASR':<12}")
asr_records = []
for m in sorted(df_test['model'].unique()):
    mask = df_test['model'] == m
    mlp_asr = df_test.loc[mask, 'mlp_pred'].mean()
    lg3_asr = df_test.loc[mask, 'llamaguard3_label'].mean()
    print(f"{m:<30} {mlp_asr:<12.4f} {lg3_asr:<12.4f}")
    asr_records.append({'Model': m, 'MLP ASR': mlp_asr, 'LlamaGuard3 ASR': lg3_asr})

asr_df = pd.DataFrame(asr_records)

# 8. Visualization
os.makedirs('results', exist_ok=True)

# Grouped barplot: MLP ASR vs LlamaGuard3 ASR per model
asr_melted = asr_df.melt(id_vars='Model', var_name='Method', value_name='ASR')
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Model', y='ASR', hue='Method', data=asr_melted, palette='Set2')
plt.title('Per-Model ASR: MLP Classifier vs LlamaGuard3')
plt.ylabel('Attack Success Rate (ASR)')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('results/mlp_hf_asr_per_model.png')
print('\nPer-model ASR barplot saved to results/mlp_hf_asr_per_model.png')

# Save ASR table to CSV
asr_df.to_csv('results/mlp_hf_asr_per_model.csv', index=False)
print('Per-model ASR table saved to results/mlp_hf_asr_per_model.csv')

# 9. Per-model Precision, Recall, F1, Support visualization
from sklearn.metrics import precision_recall_fscore_support

metrics_records = []
for m in sorted(df_test['model'].unique()):
    mask = df_test['model'] == m
    y_true_m = df_test.loc[mask, 'llamaguard3_label'].values
    y_pred_m = df_test.loc[mask, 'mlp_pred'].values
    p, r, f, s = precision_recall_fscore_support(y_true_m, y_pred_m, average='binary', zero_division=0)
    support = int(mask.sum())
    metrics_records.append({'Model': m, 'Precision': p, 'Recall': r, 'F1 Score': f, 'Support': support})

metrics_df = pd.DataFrame(metrics_records)
print('\n--- Per-model Classification Metrics ---')
print(metrics_df.to_string(index=False))

# Save metrics table to CSV
metrics_df.to_csv('results/mlp_hf_metrics_per_model.csv', index=False)
print('Per-model metrics saved to results/mlp_hf_metrics_per_model.csv')

# Grouped barplot: Precision, Recall, F1 per model
metrics_melted = metrics_df.melt(id_vars='Model', value_vars=['Precision', 'Recall', 'F1 Score'],
                                  var_name='Metric', value_name='Score')
plt.figure(figsize=(12, 6))
ax2 = sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_melted, palette='muted')
plt.title('Per-Model Classification Metrics: MLP Classifier')
plt.ylabel('Score')
plt.xlabel('Model')
plt.ylim(0, 1.15)
plt.xticks(rotation=45, ha='right')
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.2f', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('results/mlp_hf_metrics_per_model.png')
print('Per-model metrics barplot saved to results/mlp_hf_metrics_per_model.png')

# Support barplot per model
plt.figure(figsize=(10, 5))
ax3 = sns.barplot(x='Model', y='Support', data=metrics_df, palette='coolwarm')
plt.title('Test Set Support (Sample Count) per Model')
plt.ylabel('Number of Samples')
plt.xlabel('Model')
plt.xticks(rotation=45, ha='right')
for container in ax3.containers:
    ax3.bar_label(container, fmt='%d', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('results/mlp_hf_support_per_model.png')
print('Support barplot saved to results/mlp_hf_support_per_model.png')

# 10. Curve-based visualizations: ROC and Precision-Recall curves
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

pred_probs = raw_preds.cpu().numpy()
y_test_arr = np.array(y_test)
models_test_arr = np.array(models_test)

# --- Overall ROC Curve ---
fpr, tpr, _ = roc_curve(y_test_arr, pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Overall ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — MLP Classifier (Overall)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('results/mlp_hf_roc_curve_overall.png')
print('Overall ROC curve saved to results/mlp_hf_roc_curve_overall.png')

# --- Overall Precision-Recall Curve ---
prec_vals, rec_vals, _ = precision_recall_curve(y_test_arr, pred_probs)
ap = average_precision_score(y_test_arr, pred_probs)
plt.figure(figsize=(8, 6))
plt.plot(rec_vals, prec_vals, color='blue', lw=2, label=f'Overall PR (AP = {ap:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — MLP Classifier (Overall)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('results/mlp_hf_pr_curve_overall.png')
print('Overall PR curve saved to results/mlp_hf_pr_curve_overall.png')

# --- Per-model ROC Curves ---
plt.figure(figsize=(10, 7))
for m in sorted(np.unique(models_test_arr)):
    mask = models_test_arr == m
    if len(np.unique(y_test_arr[mask])) < 2:
        continue
    fpr_m, tpr_m, _ = roc_curve(y_test_arr[mask], pred_probs[mask])
    roc_auc_m = auc(fpr_m, tpr_m)
    plt.plot(fpr_m, tpr_m, lw=2, label=f'{m} (AUC = {roc_auc_m:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Per-Model ROC Curves — MLP Classifier')
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('results/mlp_hf_roc_curve_per_model.png')
print('Per-model ROC curves saved to results/mlp_hf_roc_curve_per_model.png')

# --- Per-model Precision-Recall Curves ---
plt.figure(figsize=(10, 7))
for m in sorted(np.unique(models_test_arr)):
    mask = models_test_arr == m
    if len(np.unique(y_test_arr[mask])) < 2:
        continue
    prec_m, rec_m, _ = precision_recall_curve(y_test_arr[mask], pred_probs[mask])
    ap_m = average_precision_score(y_test_arr[mask], pred_probs[mask])
    plt.plot(rec_m, prec_m, lw=2, label=f'{m} (AP = {ap_m:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Per-Model Precision-Recall Curves — MLP Classifier')
plt.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig('results/mlp_hf_pr_curve_per_model.png')
print('Per-model PR curves saved to results/mlp_hf_pr_curve_per_model.png')
