import pandas as pd

parquet_path="/Users/manishkhadka/Downloads/train.parquet"
csv_path="training.csv"

df_parquet=pd.read_parquet(parquet_path)
df_parquet.to_csv(csv_path,index=False)
print(f"Parquet converted to CSV: {csv_path}")
