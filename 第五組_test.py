import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 核心參數
# ==========================================
TEST_DIR = './split/test'
MODEL_PATH = 'best.pth'  
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_test():
    # 檢查資料夾是否存在
    if not os.path.exists(TEST_DIR):
        print(f"錯誤：找不到測試資料夾 {TEST_DIR}")
        return

    test_set = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    # 建立架構並讀取權重
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 4)
    
    if not os.path.exists(MODEL_PATH):
        print(f"錯誤：找不到權重檔案 {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE).eval()

    # --- 加入 Loss 計算 ---
    criterion = nn.CrossEntropyLoss()
    total_test_loss = 0.0
    all_preds, all_labels = [], []

    print(f"正在對 {len(test_set)} 張測試影像進行評估...")

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            out = model(imgs)
            
            # 計算 Loss
            loss = criterion(out, lbls)
            total_test_loss += loss.item() * imgs.size(0)
            
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    # --- 計算最終指標 ---
    avg_test_loss = total_test_loss / len(test_set)
    test_acc = accuracy_score(all_labels, all_preds) * 100

    print("\n" + "="*40)
    print("      獨立測試集最終評估報告")
    print("="*40)
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Acc : {test_acc:.2f}%")
    print("-" * 40)
    print(classification_report(all_labels, all_preds, target_names=test_set.classes, digits=4))

    # ==========================================
    # 混淆矩陣繪製與儲存
    # ==========================================
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_set.classes, 
                yticklabels=test_set.classes,
                annot_kws={"size": 14}) 
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Test Confusion Matrix (Acc: {test_acc:.2f}%)', fontsize=15, pad=20)
    
    save_path = 'test_confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 混淆矩陣已儲存至：{save_path}")
    
    plt.show()

if __name__ == '__main__':
    run_test()