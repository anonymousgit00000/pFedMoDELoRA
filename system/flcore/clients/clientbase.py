import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from peft import get_peft_model_state_dict, set_peft_model_state_dict, get_peft_model, LoraConfig, TaskType
import copy
from flcore.trainmodel.models import BaseHeadSplit
import torch.nn as nn
import timm
from peft import PeftModel
import os
import torch
from peft import set_peft_model_state_dict, PeftModel, LoraConfig, get_peft_model
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict


def apply_lora_with_dynamic_scan(model, r=4, alpha=8, dropout=0.1, bias="none"):

    
    target_modules = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(k in name for k in ["attn.qkv"]):

                target_modules.append(name)

    module_dict = dict(model.named_modules())

    for name in target_modules:
        module = module_dict.get(name)
        if isinstance(module, nn.Conv2d):
            raise RuntimeError(f"❌ Conv2d 层被错误注入：{name}")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        lora_dropout=dropout,
        bias=bias
    )

    model = get_peft_model(model, lora_config)

    return model

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_item('model')
        self.model = self.load_result_model()

        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                preds = torch.argmax(output, dim=1) 

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                # 统计 per-class
                for label, pred in zip(y, preds):
                    per_class_total[label.item()] += 1
                    if label == pred:
                        per_class_correct[label.item()] += 1

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        self.model.cpu()
        # self.save_model(self.model, 'model')
        # self.save_item(self.model, 'model')
        self.model.to(self.device)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        print("\n📊 Per-class Accuracy:")
        for c in range(self.num_classes):
            correct = per_class_correct[c]
            total = per_class_total[c]
            acc = 100.0 * correct / total if total > 0 else 0.0
            print(f"  Class {c}: {acc:.2f}% ({correct}/{total})")

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model = self.load_result_model()

        # self.model = self.load_item('model')
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        self.model.cpu()
        # self.save_model(self.model, 'model')
        # self.save_item(self.model, 'model')
        self.model.to(self.device)

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y

    def load_result_model(self, lora_path=None, head_path=None):
        # Get device
        device = get_device(self.global_model) if hasattr(self, 'global_model') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join("models", self.dataset, self.save_folder_name)

        # Set default paths
        # if lora_path is None:
        #     lora_path = os.path.join(model_path, self.algorithm + "_lora_weights.bin")
        # if head_path is None:
        #     head_path = os.path.join(model_path, self.algorithm + "_head_weights.bin")
        # lora_path = os.path.join(model_path, self.algorithm + "_lora_d01_d02_91.bin")
        lora_path = os.path.join(model_path, self.algorithm + "_lora_xxxx.bin")
        head_path = os.path.join(model_path, self.algorithm + "_head_xxxx.bin")
        head_path_1 = os.path.join(model_path, self.algorithm + "_head_xxxx.bin")
        head_path_2 = os.path.join(model_path, self.algorithm + "_head_xxxx.bin")
        alpha = 0.5

        if hasattr(self.model, 'base'):
            base_model = self.model.base
        else:
            base_model = self.model

        if hasattr(base_model, 'model'):
            backbone = copy.deepcopy(base_model.model.to("cpu")).to(device)
            lora_model = apply_lora_with_dynamic_scan(backbone, r=8, alpha=16, dropout=0.1, bias="none")
        else:
            lora_model = copy.deepcopy(base_model)

        if hasattr(self.model, 'base'):
            global_model = BaseHeadSplit(lora_model, copy.deepcopy(self.model.head.to("cpu")).to(device))
        else:
            global_model = lora_model

        global_model = global_model.to(device)
        global_model.eval()

        print(f"Model parameter hash before loading: {hash(str(list(global_model.parameters())[0].data.cpu().numpy()))}")

        if os.path.exists(lora_path):
            print(f"🔄 Loading LoRA weights: {lora_path}")
            try:
                lora_weights = torch.load(lora_path, map_location=device)
                if isinstance(global_model, BaseHeadSplit):
                    target_model = global_model.base
                    print("✅ Detected BaseHeadSplit model. Loading LoRA into the base model.")
                else:
                    target_model = global_model

                if not hasattr(target_model, 'peft_config'):
                    raise AttributeError("❌ Failed to load LoRA weights: target model has no attribute 'peft_config'")

                print("🔍 LoRA weights (before loading):")
                for name, param in target_model.named_parameters():
                    if "lora_" in name:
                        print(f"{name}: {param.data.cpu().numpy().flatten()[:5]} ...")
                        break

                set_peft_model_state_dict(target_model, lora_weights, adapter_name="default")

                print("🔍 LoRA weights (after loading):")
                for name, param in target_model.named_parameters():
                    if "lora_" in name:
                        print(f"{name}: {param.data.cpu().numpy().flatten()[:5]} ...")
                        break

                print("✅ LoRA weights loaded successfully")
            except Exception as e:
                raise RuntimeError(f"❌ Failed to load LoRA weights: {e}")
        else:
            raise FileNotFoundError(f"⚠️ LoRA weight file does not exist: {lora_path}")

        # # Load a single Head weight
        # if os.path.exists(head_path):
        #     print(f"🔄 Loading Head weights: {head_path}")
        #     try:
        #         head_weights = torch.load(head_path, map_location=device)
        #         if hasattr(global_model, 'head'):
        #             global_model.head.load_state_dict(head_weights, strict=False)
        #             print("✅ Head weights loaded successfully")
        #         else:
        #             print("⚠️ Head module not found, skipping load")
        #     except Exception as e:
        #         print(f"❌ Failed to load Head weights: {e}")
        # else:
        #     print(f"⚠️ Head weight file does not exist: {head_path}")

        # Weighted merging of two Head weights
        if os.path.exists(head_path_1) and os.path.exists(head_path_2):
            print(f"🔄 Merging and loading Head weights:\n  - {head_path_1}\n  - {head_path_2}\n  α = {alpha}")
            try:
                head_weights_1 = torch.load(head_path_1, map_location=device)
                head_weights_2 = torch.load(head_path_2, map_location=device)
                merged_weights = {}

                for key in head_weights_1.keys():
                    if key in head_weights_2:
                        merged_weights[key] = alpha * head_weights_1[key] + (1 - alpha) * head_weights_2[key]
                    else:
                        print(f"⚠️ Key {key} not found in second head, using first head only.")
                        merged_weights[key] = head_weights_1[key]

                if hasattr(global_model, 'head'):
                    global_model.head.load_state_dict(merged_weights, strict=False)
                    print("✅ Weighted Head weights loaded successfully")
                else:
                    print("⚠️ Head module not found, skipping load")

            except Exception as e:
                print(f"❌ Failed to load Head weights: {e}")
        else:
            print(f"⚠️ Head weight file(s) missing:\n - {head_path_1}\n - {head_path_2}")

        # Update global model
        self.global_model = global_model
        print(f"✅ Global model updated: {id(self.global_model)}")

        # Print model parameter hash after loading
        print(f"Model parameter hash after loading: {hash(str(list(self.global_model.parameters())[0].data.cpu().numpy()))}")

        # # Test the new model
        # print("🔍 Testing the loaded model...")
        # self.evaluate()
        # print("✅ Test completed")

        return self.global_model

    
    def load_result_model_avg(self):

        import torch
        import os
        import copy
        from peft import set_peft_model_state_dict

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join("models", self.dataset, self.save_folder_name)

        # 固定路径
        lora_path_1 = os.path.join(model_path, self.algorithm + "_lora_d03_01_best.bin")
        lora_path_2 = os.path.join(model_path, self.algorithm + "_lora_d03_02_best.bin")
        head_path_1 = os.path.join(model_path, self.algorithm + "_head_d03_01_best.bin")
        head_path_2 = os.path.join(model_path, self.algorithm + "_head_d03_02_best.bin")

        # 加载 LoRA 和 Head 权重
        lora_1 = torch.load(lora_path_1, map_location=device)
        lora_2 = torch.load(lora_path_2, map_location=device)
        head_1 = torch.load(head_path_1, map_location=device)
        head_2 = torch.load(head_path_2, map_location=device)

        # 参数平均
        merged_lora = {k: 0.5 * lora_1[k] + 0.5 * lora_2[k] for k in lora_1}
        merged_head = {k: 0.5 * head_1[k] + 0.5 * head_2[k] for k in head_1}

        # 初始化结构
        if hasattr(self.model, 'base'):
            base_model = self.model.base
        else:
            base_model = self.model

        backbone = copy.deepcopy(base_model.model.to("cpu")).to(device)
        lora_model = apply_lora_with_dynamic_scan(backbone, r=8, alpha=16, dropout=0.1, bias="none")

        if hasattr(self.model, 'base'):
            global_model = BaseHeadSplit(lora_model, copy.deepcopy(self.model.head.to("cpu")).to(device))
        else:
            global_model = lora_model

        global_model = global_model.to(device)
        global_model.eval()

        target_model = global_model.base if hasattr(global_model, 'base') else global_model
        set_peft_model_state_dict(target_model, merged_lora, adapter_name="default")

        if hasattr(global_model, 'head'):
            global_model.head.load_state_dict(merged_head, strict=False)

        self.global_model = global_model
        return self.global_model


    def save_item(self, item, item_name, item_path=None):
        if item_path is None:
            item_path = self.save_folder_name
        # 在前面加上 models/PACS
        item_path = os.path.join("models", "PACS", item_path)
        
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, f"domain1_82_{item_name}_r2.pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))
