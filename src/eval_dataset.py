import pandas as pd

df = pd.read_csv("model_outputs_falcon-7b-instruct_20251026_121514.csv")

# Create input string for Llama Guard-3 evaluation
df['llamaguard3_input'] = df.apply(
    lambda row: f"Prompt: {row['variants']}\nResponse: {row['model_response']}", axis=1
)

df['llamaguard3_input'].to_csv("./eval_dataset_falcon7b.csv",index=False)
