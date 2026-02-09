import pandas as pd

# Load the dataset
df = pd.read_csv("human_validation_data.csv")

# Convert the 'human_label' column to integers
df['human_label'] = df['human_label'].astype(int)

# Save the updated DataFrame back to a CSV file
df.to_csv("human_validation_data.csv", index=False)

print("The 'human_label' column has been converted to integers and saved to 'human_validation_data_fixed.csv'.")