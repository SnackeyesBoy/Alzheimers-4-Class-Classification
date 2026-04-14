# Alzheimers-4-Class-Classification
### Alzheimer's 4-Class Classification AI model training
### 所使用公開資料集 →  [Kaggle Dataset](https://www.kaggle.com/datasets/preetpalsingh25/alzheimers-dataset-4-class-of-images)

== Introduction ==
本研究旨在開發一套自動化分類系統，針對Kaggle上公開資料集，Alzheimers 4-Class Classification Dataset腦部 MRI 影像進行阿茲海默症四種病理階段的精確辨識 。研究路徑從基礎的卷積神經網路（AlzheimerCNN）出發，透過分析訓練過程中的不穩定性，逐步導入殘差結構（ResNet）、倒置殘差與編碼技術（EfficientNet）、多尺度特徵提取（GoogLeNet）以及最新的卷積架構（ConvNeXt） 。結合遷移學習與多樣化的資料增強策略，本研究成功解決了醫療影像樣本不均與過擬合的問題，將測試準確率從初期的88%大幅提升至99%以上 。本研究證實了遷移學習與先進卷積架構在小型醫療資料集上的巨大價值 。透過 ImageNet預訓練權重的特徵提取能力，搭配餘弦退火等策略解決訓練不穩問題，最終建立了一個具備高穩定性與高準確度的阿茲海默症輔助診斷系統 。 

本研究成功開發並評估了一套基於深度卷積神經網路的阿茲海默症MRI 影像分類系統。透過對比 AlzheimerCNN2、 ResNet、 GoogLeNet、 EfficientNet 及 ConvNeXt等多種經典與現代架構，實驗結果顯示，現代化純卷積網路ConvNeX在結合遷移學習與特定優化策略後，展現出最卓越的性能。該模型在獨立測試集中取得了 99.53% 的準確率與 99.72%的 Macro F1-Score，顯著優於其他對比模型，證實了其在自動化輔助診斷中的高度潛力。
