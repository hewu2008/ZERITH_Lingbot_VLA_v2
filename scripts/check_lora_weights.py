import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from safetensors import safe_open


def load_all_safetensors(checkpoint_dir):
    all_weights = {}
    safetensors_files = list(Path(checkpoint_dir).glob("**/*.safetensors"))
    
    if not safetensors_files:
        print(f"[ERROR] No .safetensors files found in {checkpoint_dir}")
        return None
    
    print(f"[INFO] Found {len(safetensors_files)} safetensors files:")
    for f in safetensors_files:
        print(f"  - {f.relative_to(checkpoint_dir)}")
    
    for file_path in safetensors_files:
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in all_weights:
                        print(f"[WARNING] Duplicate key: {key}")
                    all_weights[key] = f.get_tensor(key)
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
    
    return all_weights


def analyze_lora_weights(all_weights):
    lora_weights = {}
    base_weights = {}
    
    for key, value in all_weights.items():
        if 'lora' in key.lower():
            lora_weights[key] = value
        else:
            base_weights[key] = value
    
    print(f"\n{'='*60}")
    print(f"LoRA Weights Analysis")
    print(f"{'='*60}")
    
    if not lora_weights:
        print("[WARNING] No LoRA weights found in checkpoint!")
        return
    
    print(f"\nTotal LoRA weight keys: {len(lora_weights)}")
    print(f"Total base weight keys: {len(base_weights)}")
    
    lora_total_params = sum(p.numel() for p in lora_weights.values())
    base_total_params = sum(p.numel() for p in base_weights.values())
    total_params = lora_total_params + base_total_params
    
    print(f"\nParameter counts:")
    print(f"  - LoRA parameters: {lora_total_params:,} ({(lora_total_params/total_params*100):.4f}%)")
    print(f"  - Base model parameters: {base_total_params:,} ({(base_total_params/total_params*100):.4f}%)")
    print(f"  - Total parameters: {total_params:,}")
    
    lora_shapes = defaultdict(list)
    for key, value in lora_weights.items():
        parts = key.split('.')
        layer_name = '.'.join(parts[:-2])
        param_type = parts[-2]
        lora_shapes[(layer_name, param_type)].append((key, value.shape))
    
    print(f"\nLoRA weight distribution by layer:")
    print(f"{'Layer':<60} {'Param Type':<15} {'Shape':<25} {'Params':<15}")
    print(f"{'-'*115}")
    
    for (layer_name, param_type), items in sorted(lora_shapes.items()):
        for key, shape in items:
            params = shape.numel()
            print(f"{layer_name:<60} {param_type:<15} {str(shape):<25} {params:<15,}")
    
    print(f"\nLoRA weight statistics:")
    lora_A_weights = {}
    lora_B_weights = {}
    
    for key, value in lora_weights.items():
        if '.lora_A.weight' in key:
            lora_A_weights[key] = value
        elif '.lora_B.weight' in key:
            lora_B_weights[key] = value
    
    print(f"  - lora_A matrices: {len(lora_A_weights)}")
    print(f"  - lora_B matrices: {len(lora_B_weights)}")
    
    if lora_A_weights:
        first_A = list(lora_A_weights.values())[0]
        rank = first_A.shape[0]
        print(f"\n  LoRA rank detected: {rank}")
        
        avg_A_norm = sum(torch.norm(v).item() for v in lora_A_weights.values()) / len(lora_A_weights)
        avg_B_norm = sum(torch.norm(v).item() for v in lora_B_weights.values()) / len(lora_B_weights)
        print(f"  Average lora_A norm: {avg_A_norm:.4f}")
        print(f"  Average lora_B norm: {avg_B_norm:.4f}")
    
    non_zero_lora = sum(1 for v in lora_weights.values() if v.abs().sum() > 1e-10)
    print(f"\n  Non-zero LoRA weight tensors: {non_zero_lora}/{len(lora_weights)}")
    
    if non_zero_lora < len(lora_weights):
        print("  [WARNING] Some LoRA weights are all zeros - training may not have started!")


def main():
    parser = argparse.ArgumentParser(description="Check LoRA weights in checkpoint")
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory"
    )
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint_dir)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)
    
    print(f"[INFO] Checking checkpoint: {checkpoint_path}")
    
    all_weights = load_all_safetensors(checkpoint_path)
    if all_weights is None:
        sys.exit(1)
    
    analyze_lora_weights(all_weights)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    import torch
    main()