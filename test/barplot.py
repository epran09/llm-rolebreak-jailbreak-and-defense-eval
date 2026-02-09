import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


asr_before = pd.read_csv("asr_summary.csv")  # ASR before defense
asr_after = pd.read_csv("asr_after_defense.csv")  # ASR after defense


asr_before = asr_before.sort_values(by="model")
asr_after = asr_after.sort_values(by="model")

models = asr_before['model']
asr_before_values = asr_before['ASR'] * 100  
asr_after_values = asr_after['ASR'] * 100  

bar_width = 0.35
x = np.arange(len(models))


plt.figure(figsize=(12, 6))
bars1 = plt.bar(x - bar_width / 2, asr_before_values, bar_width, label="Before Qwen3Guard", color='royalblue')
bars2 = plt.bar(x + bar_width / 2, asr_after_values, bar_width, label="After Qwen3Guard", color='seagreen')


plt.ylabel("Attack Success Rate (%)")
plt.ylim(0, 100)
plt.xticks(x, models, rotation=15)
plt.title("ASR Before and After Defense")
plt.legend()
plt.tight_layout()


for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%', 
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10)


plt.savefig("asr_comparison.png", dpi=300)
plt.close()
