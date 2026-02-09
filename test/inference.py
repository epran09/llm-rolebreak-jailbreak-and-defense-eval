import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import yaml
import os
from src.model_loader import ModelLoader


def load_prompts(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    dataset_path = cfg['dataset']['path']
    df = pd.read_csv(dataset_path)
    return df

def run_and_save_model_outputs(model, tokenizer, prompts_df, model_id, out_dir="results", n_demo=None, max_new_tokens=512):
    output_rows = []
    test_prompts = prompts_df if n_demo is None else prompts_df.head(n_demo)
    for i, row in test_prompts.iterrows():
        print(f"\n--- Test {i+1} for model {model_id} ---")
        prompt_text = row['variants']
        domain = row['domain']
        original = row['prompt']
        try:
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=1500
            )
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            output_rows.append({
                "domain": domain,
                "original_prompt": original,
                "roleplay_prompt": prompt_text,
                "model_response": response,
                "response_length": len(response)
            })
        except Exception as e:
            print(f"Error generating response for {model_id}: {e}")
            output_rows.append({
                "domain": domain,
                "original_prompt": original,
                "roleplay_prompt": prompt_text,
                "model_response": f"[ERROR: {str(e)}]",
                "response_length": 0
            })

    # Save results
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"{out_dir}/{model_id}_outputs_{timestamp}.csv"
    pd.DataFrame(output_rows).to_csv(out_filename, index=False)
    print(f"All outputs for {model_id} saved to {out_filename}")
    avg_len = sum([o["response_length"] for o in output_rows])/max(len(output_rows),1)
    print(f"Average response length for {model_id}: {avg_len:.1f} characters")

if __name__ == "__main__":
    prompts_df = load_prompts("config/attack_config.yaml")
    
    loader = ModelLoader("config/models.yaml")

    model_ids = list(loader.config['models'].keys())
    # Sort by size field
    model_ids_sorted = sorted(model_ids, key=lambda x: float(loader.config['models'][x]['size'].replace('B', '')))
    print("Will process models in this order:", model_ids_sorted)

    for model_id in model_ids_sorted:
        print(f"\n======== Loading model: {model_id} ========")
        model, tokenizer = None, None  # Initialize model and tokenizer to None
        try:
            model, tokenizer = loader.load_model(model_id, use_8bit=False)
            run_and_save_model_outputs(model, tokenizer, prompts_df, model_id, out_dir="results")
        except Exception as e:
            print(f"Error processing model {model_id}: {e}")
        finally:
            if model is not None and tokenizer is not None:  # Check if model and tokenizer are defined
                loader.cleanup(model, tokenizer)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()