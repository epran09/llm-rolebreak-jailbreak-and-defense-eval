import pandas as pd

parquet="/Users/manishkhadka/Downloads/0000.parquet"
csv="/Users/manishkhadka/Documents/master_thesis/data/dataset.csv"

#df=pd.read_parquet(parquet)
#print("Parquet file loaded successfully")
#df.to_csv(csv,index=False)
#print("Csv file saved successfully")
df=pd.read_csv(csv)
prompts=df['prompt']
output="/Users/manishkhadka/Documents/master_thesis/data/datasets.csv"
prompts.to_csv(csv)