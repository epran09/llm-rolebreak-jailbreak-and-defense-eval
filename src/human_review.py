import pandas as pd

# Load the dataset for a single model
df = pd.read_csv("llamaguard3_results_falcon11b.csv")

# Number of samples to pick
n_samples = 50

# Calculate the number of "safe" and "unsafe" samples to pick
safe_count = int(df['llamaguard3_label'].value_counts(normalize=True).get(0, 0) * n_samples)
unsafe_count = n_samples - safe_count

# Randomly pick "safe" samples
safe_samples = df[df['llamaguard3_label'] == 0].sample(n=safe_count, random_state=42)

# Randomly pick "unsafe" samples
unsafe_samples = df[df['llamaguard3_label'] == 1].sample(n=unsafe_count, random_state=42)

# Combine the selected samples
final_sample = pd.concat([safe_samples, unsafe_samples], ignore_index=True)

# Save the final sample to a CSV file
final_sample.to_csv("50_samples_falcon11b.csv", index=False)

print("50 stratified samples saved to '50_samples_llama13b.csv'")
