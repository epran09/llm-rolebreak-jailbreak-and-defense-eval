import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple
import yaml
import os
import gc


class ModelLoader:
    
    def __init__(self, config_path: str = "config/models.yaml"):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")
            
        self.cache_dir = os.path.expanduser("~/.cache/huggingface")
        
    def get_model_config(self, model_id: str) -> Dict:
        if model_id not in self.config['models']:
            raise ValueError(f"Model {model_id} not found in configuration.")
        return self.config['models'][model_id]
    
    def load_model(self, model_id: str, use_8bit: bool = True) -> Tuple:
        model_cfg = self.get_model_config(model_id)
        model_name = model_cfg['name']
        
        print(f"Loading model: {model_id}")
        print(f"Size: {model_cfg['size']}")
        print(f"Type: {model_cfg['type']}")
        
        if torch.cuda.is_available():
            print(f"GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("WARNING: No GPU detected")
            
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                raise ValueError("Tokenizer does not have a pad_token or eos_token.")
        
        print("Loading model...")
        load_kwargs = {
            'cache_dir': self.cache_dir,
            'trust_remote_code': True,
            'low_cpu_mem_usage': True
        }
        
        if torch.cuda.is_available():
            if use_8bit:
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    raise ImportError("The 'bitsandbytes' library is required for 8-bit model loading. Install it with `pip install bitsandbytes`.")
                load_kwargs['load_in_8bit'] = True
            else:
                load_kwargs['torch_dtype'] = torch.float16
            load_kwargs['device_map'] = "auto"
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model.eval()
        
        print("Model loaded successfully!\n")
        
        return model, tokenizer
    
    def get_model_group(self, group_name: str) -> list:
        if group_name not in self.config['model_groups']:
            raise ValueError(f"Model group {group_name} not found")
        return self.config['model_groups'][group_name]
    
    def cleanup(self, model, tokenizer):
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


