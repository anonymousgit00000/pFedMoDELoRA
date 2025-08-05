from peft import set_peft_model_state_dict
import torch
import os
from torch.nn.functional import normalize

def clean_state_dict(state_dict):
    """
    Remove redundant prefixes in state_dict keys
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("base.base_model.model."):
            new_key = key.replace("base.base_model.model.", "")
        elif key.startswith("base_model.model."):
            new_key = key.replace("base_model.model.", "")
        elif key.startswith("base."):
            new_key = key.replace("base.", "")
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def FedAvg(model, selected_clients_set, local_dataset_len_dict):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32), p=1, dim=0)

    print("Weights:", weights_array)
    for k, client_id in enumerate(selected_clients_set):
        print(f"✅ k: {k}")
        print(f"✅ client_id: {client_id}")

        if k == 0:
            single_output_dir = "./models/PACS/xxxx.bin"
        if k == 1:
            single_output_dir = "./models/PACS/xxxx.bin"

        try:
            single_weights = torch.load(single_output_dir, map_location='cpu', weights_only=True)
            print(f"✅ Successfully loaded weights from: {single_output_dir}")

            filtered_weights = {}
            for key, value in single_weights.items():
                if ('lora_A' in key or 'lora_B' in key) and len(value.shape) == 2:
                    filtered_weights[key] = value
                else:
                    if not ('lora_A' in key or 'lora_B' in key):
                        reason = "not a LoRA weight (missing 'lora_A' or 'lora_B')"
                    elif len(value.shape) != 2:
                        reason = f"invalid shape (expected 2D, got {len(value.shape)}D)"
                    else:
                        reason = "unknown issue"
                    print(f"❌ Invalid weight: {key}, shape: {value.shape}, reason: {reason}")

            if len(filtered_weights) == 0:
                print(f"⚠️ No valid LoRA weights found in {single_output_dir}, skipping.")
                continue

            single_weights = filtered_weights

        except Exception as e:
            print(f"❌ Failed to load weights from {single_output_dir}: {e}")
            continue

        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in single_weights.keys()}
        else:
            for key in single_weights.keys():
                weighted_single_weights[key] += single_weights[key] * (weights_array[k])

    torch.save(weighted_single_weights, os.path.join("./models/PACS/xxxx.bin"))
    print("✅ Saved merged weights successfully")

    return model
