import pandas as pd
import matplotlib.pyplot as plt

big_df = pd.read_csv("all_model_moderation_results.csv")
models = big_df['model'].unique()

fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs = axs.flatten()

for ax, model in zip(axs, models):
    subset = big_df[big_df['model'] == model]
    counts = subset['llamaguard3_label'].value_counts()
    labels = ["Safe", "Jailbreak"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=["mediumseagreen", "crimson"],
        startangle=90,
        wedgeprops=dict(width=0.56)
    )
    ax.set_title(model)
    plt.setp(autotexts, size=10, weight="bold", color="white")

plt.tight_layout()
plt.savefig("all_donut_charts_grid.png", dpi=300)
plt.close()
