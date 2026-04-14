import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# ==========================================
# 1. 核心路徑與參數
# ==========================================
BASE_DIR = './split'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

BATCH_SIZE = 64
NUM_EPOCHS = 25         # 加入 Scheduler 後建議多跑幾輪
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 進階影像增強 (加入 RandomErasing)
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)) # 模擬部分影像缺失，提升魯棒性
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. 三大指標繪圖函數
# ==========================================
def plot_metrics(history):
    epochs = range(1, len(history['t_loss']) + 1)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['t_loss'], 'b-o', label='Train'); plt.plot(epochs, history['v_loss'], 'r-s', label='Val')
    plt.title('Loss Trajectory'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['t_acc'], 'b-o', label='Train'); plt.plot(epochs, history['v_acc'], 'r-s', label='Val')
    plt.title('Accuracy Growth'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['v_f1'], 'g-^', label='Val Macro F1')
    plt.title('Macro F1-Score'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('training_metrics_pro.png', dpi=300); plt.show()

# ==========================================
# 4. 主訓練邏輯
# ==========================================
def main():
    print(f"啟動設備: {DEVICE} | 正在計算類別權重...")

    train_set = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_set = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    # --- 關鍵修正：計算類別權重 (Weight Balancing) ---
    y_train = train_set.targets
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(y_train), 
                                         y=y_train)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"類別權重設定完成: {weights_tensor}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 建立 ConvNeXt-Tiny
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 4)
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    
    # --- 關鍵修正：Loss 加入權重，並引入 Scheduler ---
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': [], 'v_f1': []}
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        t_loss, t_corr = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item() * imgs.size(0)
            t_corr += out.max(1)[1].eq(lbls).sum().item()
            pbar.set_postfix(acc=f"{100.*t_corr/len(train_set):.2f}%")

        # 驗證
        model.eval()
        v_loss, v_corr, y_true, y_pred = 0, 0, [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                out = model(imgs)
                v_loss += criterion(out, lbls).item() * imgs.size(0)
                preds = out.max(1)[1]
                v_corr += preds.eq(lbls).sum().item()
                y_true.extend(lbls.cpu().numpy()); y_pred.extend(preds.cpu().numpy())

        # 更新學習率
        scheduler.step()

        current_v_acc = 100. * v_corr / len(val_set)
        current_v_f1 = f1_score(y_true, y_pred, average='macro')

        history['t_loss'].append(t_loss / len(train_set)); history['v_loss'].append(v_loss / len(val_set))
        history['t_acc'].append(100. * t_corr / len(train_set)); history['v_acc'].append(current_v_acc)
        history['v_f1'].append(current_v_f1)

        print(f" >> Val Acc: {current_v_acc:.2f}% | F1: {current_v_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if current_v_acc > best_acc:
            best_acc = current_v_acc
            torch.save(model.state_dict(), 'best.pth')
            print(f"!! 儲存最佳權重: {best_acc:.2f}%")

    plot_metrics(history)
    print("\n最後一輪分類報告：")
    print(classification_report(y_true, y_pred, target_names=train_set.classes, digits=4))

if __name__ == '__main__':
    main()