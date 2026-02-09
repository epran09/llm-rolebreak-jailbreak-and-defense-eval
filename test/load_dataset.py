import pandas as pd
import json
file_path="/Users/manishkhadka/Documents/master_thesis/data/roleplay_datasets.csv"
def load_dataset(file_path):
    df=pd.read_csv(file_path)
    
    required_columns=['prompt','domain','variants']
    for col in required_columns:
        assert col in df.columns,f"Dataset must have '{col} column"
        
    print(f"Loaded {len(df)} role-play prompts.")
    print(f"Domains: {df['domain'].unique()}")
    
    return df

dataset=load_dataset(file_path)
print(dataset.head(3))