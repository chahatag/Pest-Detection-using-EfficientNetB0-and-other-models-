# Pest Detection using EfficientNetB0 and Other Models

This repository presents a deep learning-based **pest classification system** using various CNN architectures. Among them, **EfficientNetB0** achieved the **highest accuracy of 95.96%**. The system classifies 12 types of pests from images and includes a **Streamlit web app** for real-time predictions.

---

## Dataset

Dataset used:  
[Agricultural Pests Image Dataset - Kaggle](https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset)

- **Total Classes:** 12  
  `ants`, `bees`, `beetle`, `catterpillar`, `earthworms`, `earwig`, `grasshopper`, `moth`, `slug`, `snail`, `wasp`, `weevil`

- **Image Count:**
  - Training: 5,394  
  - Validation: 1,331  
  - Test: 1,337

##  Models Trained & Performance

| Model           | Accuracy     | Remarks                            |
|----------------|--------------|-------------------------------------|
| **Custom CNN**   | 41.14%       | Built from scratch                  |
| **MobileNetV2**  | 88.26%       | Transfer learning                   |
| **ResNet50**     | 94.91%       | Transfer + fine-tuning              |
| **EfficientNetB0** | **95.96%** | Best model - selected for deployment |

All models were trained in the same Jupyter notebook:  
`Pest_Detect_modeltrain.ipynb`

The final model (EfficientNetB0) is extracted in a clean notebook for deployment:  
`EfficientNetB0_Classifier.ipynb`

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pest-detection-efficientnet.git
cd pest-detection-efficientnet

