# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from safetensors import safe_open
from lingbotvla.utils import helper
logger = helper.create_logger(__name__)

def freeze_parameters(model: nn.Module):
    # Freeze parameters
    model.requires_grad_(False)
    model.eval()
    model.train()


def add_lora_to_align_modules(
    model: nn.Module,
    lora_rank=4,
    lora_alpha=4,
    lora_target_modules="proj_in1,proj_in2,proj_out,to_q,to_kv,to_out",
    init_lora_weights=True,
):
    """
    Add LoRA to alignment-related modules in the model.
    This includes TaskTokenDepthHead, state_proj, action_in_proj, action_out_proj, etc.
    Handles nested Linear layers in FeedForward (nn.Sequential) automatically.
    """
    if init_lora_weights == "kaiming":
        init_lora_weights = True

    align_target_modules = [m.strip() for m in lora_target_modules.split(",")]

    align_module_names = [
        "depth_align_head",
        "future_depth_align_head",
        "current_video_align_head",
        "future_video_align_head",
        "current_shared_task_proj",
        "future_shared_task_proj",
        "state_proj",
        "action_in_proj",
        "action_out_proj",
    ]

    def find_linear_modules(module, prefix=""):
        """Find all nn.Linear modules and their full paths."""
        result = {}
        for name, m in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(m, nn.Linear):
                result[full_name] = m
            elif isinstance(m, nn.Module):
                result.update(find_linear_modules(m, full_name))
        return result

    def find_modules_with_attrs(module, attr_names, prefix=""):
        """Recursively find submodules that have specific attributes."""
        result = []
        for name, m in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            has_any = any(hasattr(m, attr) for attr in attr_names)
            if has_any:
                result.append((full_name, m))
            if isinstance(m, nn.Module):
                result.extend(find_modules_with_attrs(m, attr_names, full_name))
        return result

    align_submodules = find_modules_with_attrs(model, align_module_names)

    if not align_submodules:
        logger.info_rank0("[LoRA Align] No alignment modules found in model.")
        return []

    logger.info_rank0(f"[LoRA Align] Found {len(align_submodules)} module(s) containing alignment attributes:")
    for name, _ in align_submodules:
        logger.info_rank0(f"  - {name}")

    align_module_param_prefixes = []
    for parent_name, parent_module in align_submodules:
        for module_name in align_module_names:
            if not hasattr(parent_module, module_name):
                continue
            module = getattr(parent_module, module_name)
            if module is None:
                continue

            if not isinstance(module, nn.Module):
                continue

            linear_modules = find_linear_modules(module)
            if not linear_modules:
                continue

            matched_names = [name for name in linear_modules.keys() 
                            if any(t in name for t in align_target_modules)]
            unmatched_names = [name for name in linear_modules.keys() 
                             if not any(t in name for t in align_target_modules)]

            logger.info_rank0(f"[LoRA Align] '{parent_name}.{module_name}': {len(linear_modules)} Linear layers")
            logger.info_rank0(f"  Target matches: {len(matched_names)}, unmatched: {len(unmatched_names)}")
            if matched_names:
                logger.info_rank0(f"  Matched: {matched_names}")
            if unmatched_names:
                logger.info_rank0(f"  Unmatched (will use 'Linear' target): {unmatched_names}")

            all_target_modules = list(align_target_modules)
            if unmatched_names:
                all_target_modules.append("Linear")

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights=init_lora_weights,
                target_modules=all_target_modules,
            )

            inject_adapter_in_model(lora_config, module)

            lora_count = sum(1 for _, p in module.named_parameters() if "lora" in _)
            logger.info_rank0(f"  Added {lora_count} LoRA parameters to '{parent_name}.{module_name}'")

            for name, param in module.named_parameters():
                if "lora" in name:
                    param.data = param.data.to(dtype=torch.float32)

            param_prefix = f"model.{parent_name}.{module_name}"
            align_module_param_prefixes.append(param_prefix)

    embed_keywords = [
        "depth_align_embs",
        "future_depth_align_embs",
        "current_video_align_embs",
        "future_video_align_embs",
    ]
    for name, param in model.named_parameters():
        if any(kw in name for kw in embed_keywords):
            param.requires_grad = True

    align_lora_params = []
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            if any(prefix in name for prefix in align_module_param_prefixes):
                align_lora_params.append(f"{name}: {param.numel()}")

    return align_lora_params


def add_lora_to_model(
    model: nn.Module,
    lora_rank=4,
    lora_alpha=4,
    lora_target_modules="q,k,v,o,ffn.0,ffn.2",
    init_lora_weights="kaiming",
    pretrained_lora_path=None,
    state_dict_converter=None,
    lora_target_modules_support=None,
):
    model.lora_alpha = lora_alpha
    if init_lora_weights == "kaiming":
        init_lora_weights = True

    module_name_map = {
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "o": "o_proj",
        "ffn.0": "gate_proj",
        "ffn.1": "up_proj",
        "ffn.2": "down_proj",
    }

    target_modules = []
    for module in lora_target_modules.split(","):
        module = module.strip()
        target_modules.append(module_name_map.get(module, module))

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=target_modules,
    )

    for lora_target_module in lora_config.target_modules:
        if lora_target_module not in lora_target_modules_support:
            raise ValueError(f"lora_target_module {lora_target_module} not in lora_target_modules_support")

    model = inject_adapter_in_model(lora_config, model)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)

    for name, param in model.named_parameters():
        if "lora" in name:
            param.data = param.data.to(dtype=torch.float32)

    # Lora pretrained lora weights
    if pretrained_lora_path is not None:
        state_dict = load_state_dict(pretrained_lora_path)
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        all_keys = [i for i, _ in model.named_parameters()]
        num_updated_keys = len(all_keys) - len(missing_keys)
        num_unexpected_keys = len(unexpected_keys)
        print(
            f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected."
        )


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def extract_lora_state_dict(state_dict: dict) -> dict:
    lora_state_dict = {}
    for key, value in state_dict.items():
        if "lora_A" in key or "lora_B" in key:
            lora_state_dict[key] = value
    return lora_state_dict


def save_lora_weights_only(
    model: nn.Module,
    output_path: str,
    save_dtype: torch.dtype = torch.float32,
):
    import os
    os.makedirs(output_path, exist_ok=True)
    
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state_dict[name] = param.data.detach().cpu().to(save_dtype)
    
    from safetensors.torch import save_file
    save_file(lora_state_dict, os.path.join(output_path, "lora_weights.safetensors"))
    
    print(f"LoRA weights saved to {output_path}")
    print(f"Total LoRA parameters: {sum(p.numel() for p in lora_state_dict.values())}")
    print(f"Number of LoRA keys: {len(lora_state_dict)}")
