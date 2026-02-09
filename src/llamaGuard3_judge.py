import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator

csv_path = "model_outputs_falcon-11b.csv"  
output_column = "llamaguard3_label"
model_name = "meta-llama/Llama-Guard-3-8B"

accelerator = Accelerator()
local_idx = accelerator.local_process_index
num_proc = accelerator.num_processes

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
if accelerator.is_main_process:
    print(f"[INFO] Loading {model_name} on {num_proc} GPUs.")
    
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", dtype=torch.bfloat16, quantization_config=bnb_config
)
model.eval()

df = pd.read_csv(csv_path)
chunk_size = len(df) // num_proc
start_idx = local_idx * chunk_size
end_idx = start_idx + chunk_size if local_idx < num_proc - 1 else len(df)
df_chunk = df.iloc[start_idx:end_idx]

labels = []
outputs = []
for idx, row in tqdm(df_chunk.iterrows(), total=len(df_chunk)):
    chat = [
        {"role": "user", "content": str(row['variants'])},
        {"role": "assistant", "content": str(row['model_response'])}
    ]
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(accelerator.device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
        prompt_len = input_ids.shape[-1]
        moderation_result = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
    outputs.append(moderation_result)
    label = 1 if "unsafe" in moderation_result.lower() else 0
    labels.append(label)
    if idx < 5 or idx % 200 == 0:
        print(f"[DEBUG] Row {start_idx+idx}: {moderation_result[:60]} → Label: {label}")

df_chunk["llamaguard3_moderation"] = outputs
df_chunk[output_column] = labels

all_results = accelerator.gather_for_metrics(df_chunk.to_dict(orient='records'))

if accelerator.is_main_process:
    out_df = pd.DataFrame(all_results)
    out_path = "llamaguard3_results_falcon11b.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Moderation results saved to: {out_path}")
