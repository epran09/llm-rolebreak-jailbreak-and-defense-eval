import pandas as pd
import matplotlib.pyplot as plt

dfs=[]

for name,fname in{
    "microsoft-Phi-3-mini":"qwen3guard_phi3.csv",
    "falcon-7b-instruct":"qwen3guard_falcon7b.csv",
    "Mistral-7B-Instruct":"qwen3guard_mistral7b.csv",
    "Llama-2-7b":"qwen3guard_llama2-7b.csv",
    "Falcon-11b":"qwen3guard_falcon11b.csv",
    "Llama-2-13b":"qwen3guard_llama-2-13b.csv"
}.items():
    df=pd.read_csv(fname)
    df['model']=name
    dfs.append(df)
    
big_df=pd.concat(dfs,ignore_index=True)
big_df.to_csv("all_model_defense_results.csv", index=False)
