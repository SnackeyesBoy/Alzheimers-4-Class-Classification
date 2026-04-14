import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# 1. 參數配置 (裕翔請確認路徑)
# ==========================================
CONFIG = {
    "model_path": "best.pth",
    "output_dir": "grad_cam_comparison",
    "image_size": 224,
    "class_names": ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented'],
    "test_samples": [
        {"path": "./split/test/MildDemented/mild_9.jpg", "label": 0},
        {"path": "./split/test/ModerateDemented/moderate_3.jpg", "label": 1},
        {"path": "./split/test/NonDemented/non_5.jpg", "label": 2},
        {"path": "./split/test/VeryMildDemented/verymild_27.jpg", "label": 3}
    ]
}

# ==========================================
# 2. 模型載入函數
# ==========================================
def get_trained_model(model_path, device):
    print(f"正在初始化 ConvNeXt-Tiny 並載入權重...")
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    
    # 定義 Grad-CAM 的目標層
    target_layers = [model.features[7][-1]]
    return model, target_layers

# ==========================================
# 3. 核心處理邏輯
# ==========================================
def run_comparison_analysis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    model, target_layers = get_trained_model(CONFIG["model_path"], device)
    
    preprocess = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"開始生成 Grad-CAM 對比圖...")

    for sample in CONFIG["test_samples"]:
        img_path = sample["path"]
        target_id = sample["label"]
        class_label = CONFIG["class_names"][target_id]
        
        if not os.path.exists(img_path):
            print(f"找不到路徑: {img_path}，跳過。")
            continue

        # A. 讀取與預處理
        rgb_pil = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(rgb_pil).unsqueeze(0).to(device)
        orig_img = np.array(rgb_pil.resize((CONFIG["image_size"], CONFIG["image_size"])), dtype=np.float32) / 255.0

        # B. 執行 Grad-CAM
        with GradCAM(model=model, target_layers=target_layers) as cam:
            targets = [ClassifierOutputTarget(target_id)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            visualization = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)

        # C. 繪製三位一體比對圖 (解決字體遮擋版)
        fig = plt.figure(figsize=(18, 7)) # 加寬加高
        fig.suptitle(f"Model Diagnostic Explainability - {class_label}", fontsize=20, fontweight='bold', y=0.95)
        
        # 1. 原始圖
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(orig_img)
        ax1.set_title("Original MRI Scan", fontsize=14, pad=15)
        plt.axis('off')
        
        # 2. 純熱力圖
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(grayscale_cam, cmap='jet')
        ax2.set_title("Grad-CAM Activation", fontsize=14, pad=15)
        plt.axis('off')
        
        # 3. 疊加圖
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(visualization)
        ax3.set_title("Overlay (Pathological Focus)", fontsize=14, pad=15)
        plt.axis('off')
        
        # 存檔微調
        plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # 為頂部大標題留空間
        
        save_name = f"Compare_{class_label}.png"
        save_path = os.path.join(CONFIG["output_dir"], save_name)
        
        # 使用 bbox_inches='tight' 確保所有文字邊界都被包含
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"成功生成並儲存: {save_path}")
