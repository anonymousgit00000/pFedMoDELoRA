import os, random, pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs("tsne_visualization", exist_ok=True)

domain_paths = {
    'art_painting': './PACS/7class_4domain_70percent/train/0.pkl',
    'cartoon':      './PACS/7class_4domain_70percent/train/1.pkl',
    'photo':        './PACS/7class_4domain_70percent/train/2.pkl',
    'sketch':       './PACS/7class_4domain_70percent/train/3.pkl',
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,)*3, std=(0.5,)*3)
])

def add_gaussian_noise(features, q=0.2, s=0.05):
    noise = torch.randn_like(features) * s
    return features + q * noise

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_array = self.data[idx]
        if isinstance(img_array, torch.Tensor):
            img_array = img_array.numpy()
        if img_array.shape[0] in [1, 3]:
            img_array = np.transpose(img_array, (1, 2, 0))
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        img = self.transform(img)
        return img, self.labels[idx]

# 模型组件
class FeatureProjector(nn.Module):
    def __init__(self, input_dim=1024, output_dim=768):  # ✅ 改为 input_dim=1024
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
    
class PrototypeClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, z):
        z = F.normalize(z, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        return -torch.cdist(z, prototypes)

class ViTPrototypeNet(nn.Module):
    def __init__(self, vit_model, num_classes):
        super().__init__()
        self.vit = vit_model
        self.projector = FeatureProjector(input_dim=1024, output_dim=768)  # ✅ 改这里
        self.classifier = PrototypeClassifier(num_classes, 768)

    def forward(self, x, add_noise=True, q=0.2, s=0.05):
        h = self.vit(x)
        if add_noise:
            h = add_gaussian_noise(h, q=q, s=s)
        z = self.projector(h)
        logits = self.classifier(z)
        return logits, z

def compute_mean_prototypes(model, loader, num_classes):
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = model.projector(model.vit(imgs))
            all_features.append(feats)
            all_labels.append(labels)
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    mean_proto = torch.zeros(num_classes, all_features.size(1)).to(device)
    for c in range(num_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            mean_proto[c] = all_features[mask].mean(dim=0)
    return mean_proto

def visualize_tsne_convnext(epoch, model, test_loader, num_classes, device, show=False):
    model.eval()
    features, labels, preds = [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="t-SNE feature extraction", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits, z = model(imgs, add_noise=False)
            pred = logits.argmax(dim=1)
            features.append(z.cpu())
            labels.append(lbls.cpu())
            preds.append(pred.cpu())
            correct += (pred == lbls).sum().item()
            total += lbls.size(0)

    acc = correct / total
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    domain_names = ['art_painting', 'cartoon', 'photo', 'sketch']  # ✅ 统一 domain 命名

    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(num_classes):
        idxs = (labels == c)
        ax.scatter(features_2d[idxs, 0], features_2d[idxs, 1], s=10, alpha=0.5, label=domain_names[c])

    ax.set_title(f"(d) Features after Training (ConvNeXt) | Acc: {acc:.2%}", fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-40, 40)
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(f"tsne_visualization/convnext_epoch_{epoch+1}.pdf")

    if show:
        fig.savefig("tsne_visualization/fig4.pdf")
        plt.show()

    plt.close(fig)


# 构造数据
samples_per_class_train = 50
samples_per_class_test = 25
num_classes_per_domain = 7
num_domains = len(domain_paths)

train_data, train_labels, test_data, test_labels = [], [], [], []
label_map = {domain: i for i, domain in enumerate(domain_paths.keys())}

for domain, path in domain_paths.items():
    with open(path, 'rb') as f:
        data = pickle.load(f)
    imgs, labels = list(data['x']), list(data['y'])

    class_to_imgs = defaultdict(list)
    for img, cls in zip(imgs, labels):
        class_to_imgs[cls].append(img)

    domain_id = label_map[domain]
    
    # for cls_id in range(num_classes_per_domain):
    #     if len(class_to_imgs[cls_id]) < samples_per_class_train + samples_per_class_test:
    #         continue
    #     img_list = class_to_imgs[cls_id]
    #     random.shuffle(img_list)
    #     train_imgs = img_list[:samples_per_class_train]
    #     test_imgs = img_list[samples_per_class_train:samples_per_class_train + samples_per_class_test]
    #     train_data += train_imgs
    #     train_labels += [domain_id] * len(train_imgs)
    #     test_data += test_imgs
    #     test_labels += [domain_id] * len(test_imgs)

    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    for cls_id in range(num_classes_per_domain):
        if len(class_to_imgs[cls_id]) < samples_per_class_train + samples_per_class_test:
            continue

        img_list = class_to_imgs[cls_id]
        random.shuffle(img_list)

        # ✅ 提取 ConvNeXt 特征
        convnext.eval()
        all_feats = []
        with torch.no_grad():
            for img_array in img_list:
                if isinstance(img_array, torch.Tensor):
                    img_array = img_array.numpy()
                if img_array.shape[0] in [1, 3]:
                    img_array = np.transpose(img_array, (1, 2, 0))
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                img_tensor = transform(img).unsqueeze(0).to(device)
                feat = convnext(img_tensor).squeeze(0).cpu().numpy()
                all_feats.append(feat)
        all_feats = np.stack(all_feats, axis=0)

        # ✅ k-means 聚类
        kmeans = KMeans(n_clusters=samples_per_class_train, random_state=42, n_init='auto')
        kmeans.fit(all_feats)
        closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, all_feats)

        # ✅ 聚类代表样本作为训练样本
        train_imgs = [img_list[idx] for idx in closest_indices]

        # ✅ 其余作为测试样本
        test_pool = [img_list[i] for i in range(len(img_list)) if i not in closest_indices]
        test_imgs = test_pool[:samples_per_class_test]

        train_data += train_imgs
        train_labels += [domain_id] * len(train_imgs)
        test_data += test_imgs
        test_labels += [domain_id] * len(test_imgs)


train_loader = DataLoader(CustomDataset(train_data, train_labels, transform), batch_size=64, shuffle=True)
test_loader = DataLoader(CustomDataset(test_data, test_labels, transform), batch_size=64, shuffle=False)

# 加载冻结的 ViT 主干
convnext = timm.create_model('convnext_base', pretrained=True, num_classes=0).to(device)
for name, param in convnext.named_parameters():
    param.requires_grad = False

model = ViTPrototypeNet(convnext, num_classes=num_domains).to(device)

# ✅ 用 mean prototype 初始化 learnable prototype
mean_proto = compute_mean_prototypes(model, train_loader, num_classes=num_domains)
with torch.no_grad():
    model.classifier.prototypes.copy_(mean_proto)
print("[INFO] Prototypes initialized from mean features.")

# 优化器
optimizer = optim.Adam(
    [p for n, p in model.named_parameters() if 'vit' not in n],
    lr=1e-4,
    weight_decay=1e-4
)

# optimizer = optim.Adam([
#     {'params': model.projector.parameters(), 'lr': 1e-4},
#     {'params': model.classifier.parameters(), 'lr': 1e-5},  # prototype lr 更大
# ], weight_decay=1e-4)

# 训练函数
def train_one_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _ = model(imgs, add_noise=True)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    print(f"Train Loss: {total_loss/total:.4f} | Train Acc: {correct/total*100:.2f}%")

# 评估函数
def evaluate():
    model.eval()
    correct, total = 0, 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs, add_noise=False)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            for t, p in zip(labels.cpu(), preds.cpu()):
                per_class_total[t.item()] += 1
                if t.item() == p.item():
                    per_class_correct[t.item()] += 1
    print(f"Test Accuracy: {correct/total*100:.2f}%")
    for d in sorted(per_class_total.keys()):
        acc = per_class_correct[d] / per_class_total[d]
        print(f"  Domain {d} Acc: {acc*100:.2f}%")

# 训练循环
for epoch in range(15):
    print(f"\nEpoch {epoch+1}")
    train_one_epoch()
    evaluate()
    visualize_tsne_convnext(
    epoch,
    model,
    test_loader,
    num_classes=num_domains,
    device=device,
    show=(epoch == 14)  # ✅ 最后一轮显示（Python 中 14 表示第 15 轮）
)



