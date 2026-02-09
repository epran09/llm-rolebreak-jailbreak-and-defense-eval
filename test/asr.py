import pandas as pd

df=pd.read_csv("all_model_defense_results.csv")

asr_summary=df.groupby("model")["final_accepted_label"].agg(
    n_attacks="count",
    n_jailbreaks="sum"
).reset_index()

asr_summary["ASR"]=asr_summary["n_jailbreaks"]/asr_summary["n_attacks"]
asr_summary["ASR"] = asr_summary["ASR"].round(3)
asr_summary.to_csv("asr_after_defense.csv",index=False)