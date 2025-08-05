import os
import torch
import timm
import warnings
from peft import get_peft_model, LoraConfig, set_peft_model_state_dict
from lora_stacking import FedAvg  

warnings.simplefilter("ignore")
torch.manual_seed(0)

def load_base_model_and_lora():
    print("Loading base model...")
    base_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    print("Base model loaded.")

    # LoRA 配置
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        lora_dropout=0.1,
        bias="none"
    )

    print("Applying LoRA to the base model...")
    base_model = get_peft_model(base_model, lora_config)
    print("LoRA applied.")

    lora_weights_path = "xxxx.bin"
    if os.path.exists(lora_weights_path):
        print(f"Loading LoRA weights from {lora_weights_path}...")
        lora_weights = torch.load(lora_weights_path, map_location='cpu')
        try:
            set_peft_model_state_dict(base_model, lora_weights, "default")
            print("✅ LoRA weights loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading LoRA weights: {e}")
    else:
        print("⚠️ No LoRA weights found. Proceeding with the base model.")

    return base_model

selected_clients_set = ["client1", "client2"]
local_dataset_len_dict = {
    "client1": 2000,
    "client2": 8000,
}

weight_paths = {
    "client1": "./models/PACS/xxxx.bin",
    "client2": "./models/PACS/xxxx.bin",
}

model = load_base_model_and_lora()

model = FedAvg(
    model=model,
    selected_clients_set=selected_clients_set,
    local_dataset_len_dict=local_dataset_len_dict,
    weight_paths=weight_paths
)
