import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

big_df = pd.read_csv("all_model_moderation_results.csv")

domain_model_rates = big_df.groupby(['domain','model'])['llamaguard3_label'].mean().unstack() * 100

plt.figure(figsize=(12, 6))
sns.heatmap(domain_model_rates, annot=True, fmt=".1f", cmap="RdPu", linewidths=0.7)
plt.title("Jailbreak Rate (ASR %) by Domain and Model")
plt.ylabel("Domain")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("asr_domain_model_heatmap.png", dpi=300)
plt.close()
