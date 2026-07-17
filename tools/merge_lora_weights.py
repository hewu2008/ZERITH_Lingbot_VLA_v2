import argparse
import os
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safetensors import safe_open
from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
from lingbotvla.utils.lora_utils import add_lora_to_model


def merge_lora_into_base(model):
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            lora_A = module.lora_A.weight.data
            lora_B = module.lora_B.weight.data
            alpha = module.lora_alpha
            rank = lora_A.shape[0]
            
            scaling = alpha / rank
            delta_w = (lora_B @ lora_A) * scaling
            
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data += delta_w.to(module.weight.data.dtype)
            
            module.lora_A = None
            module.lora_B = None
            module.lora_alpha = None
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model (LingbotVLA-V2 checkpoint)"
    )
    
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        required=True,
        help="Path to the LoRA weights directory"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for merged weights"
    )
    
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help="LoRA alpha"
    )
    
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.1,ffn.2",
        help="LoRA target modules"
    )
    
    parser.add_argument(
        "--lora_target_modules_support",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Supported LoRA target modules"
    )
    
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        default=False,
        help="Use bfloat16 for merged model"
    )
    
    args = parser.parse_args()
    
    print(f"Loading base model from: {args.base_model_path}")
    print(f"Loading LoRA weights from: {args.lora_weights_path}")
    print(f"Output path: {args.output_path}")
    
    training_config_path = Path(args.base_model_path).parent.parent.parent / 'lingbotvla_cli.yaml'
    if not training_config_path.exists():
        training_config_path = Path(args.base_model_path).parent.parent / 'lingbotvla_cli.yaml'
    
    import yaml
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    training_model_config = training_config['model']
    training_model_config.update(training_config['train'])
    config = LingbotVLAV2Config(**training_model_config)
    for key, value in training_model_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    config.attention_implementation = 'eager'
    config.use_cache = True
    
    print("Initializing base model...")
    model = LingbotVlaV2Policy(config, eval=True)
    
    all_safetensors = list(Path(args.base_model_path).glob("*.safetensors"))
    if len(all_safetensors) > 0:
        merged_weights = {}
        for file_path in all_safetensors:
            print(f"Loading base weights from: {file_path}")
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    merged_weights[key] = f.get_tensor(key)
        model.load_state_dict(merged_weights, strict=True)
        print(f"Loaded {len(merged_weights)} base model weights")
    else:
        print("No base weights found, using randomly initialized model")
    
    print("Injecting LoRA adapters...")
    add_lora_to_model(
        model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        pretrained_lora_path=args.lora_weights_path,
        lora_target_modules_support=args.lora_target_modules_support,
    )
    
    print("Merging LoRA weights into base model...")
    model = merge_lora_into_base(model)
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    merged_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            merged_state_dict[name] = param.data.cpu()
    
    for name, buffer in model.named_buffers():
        merged_state_dict[name] = buffer.data.cpu()
    
    if args.use_bf16:
        for key in merged_state_dict:
            merged_state_dict[key] = merged_state_dict[key].to(torch.bfloat16)
    
    safetensors_path = output_path / "model.safetensors"
    from safetensors.torch import save_file
    save_file(merged_state_dict, str(safetensors_path))
    
    lingbotvla_cli_path = output_path / "lingbotvla_cli.yaml"
    with open(lingbotvla_cli_path, 'w') as f:
        yaml.dump(training_config, f, indent=2)
    
    print(f"Merged weights saved to: {safetensors_path}")
    print(f"Training config saved to: {lingbotvla_cli_path}")
    print(f"Total parameters: {sum(p.numel() for p in merged_state_dict.values())}")


if __name__ == "__main__":
    main()