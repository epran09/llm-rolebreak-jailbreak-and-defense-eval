import pandas as pd
from sklearn.metrics import cohen_kappa_score

df=pd.read_csv("human_validation_data.csv")
df_clean=df.dropna(subset=['human_label','llamaguard3_label'])

df_clean['human_label'] = df_clean['human_label'].astype(int)
df_clean['llamaguard3_label'] = df_clean['llamaguard3_label'].astype(int)

results = []
for model_name in df_clean['model'].unique():
    sub = df_clean[df_clean['model'] == model_name]
    kappa = cohen_kappa_score(sub['human_label'], sub['llamaguard3_label'])
    results.append({'model': model_name, 'cohen_kappa_score': kappa})

result_df = pd.DataFrame(results)
result_df.to_csv('cohen_kappa_scores_per_model.csv', index=False)