# 🔬 Melanoma Detection using Two-Stage Fine-Tuned ResNet50

A deep learning research project for automated melanoma detection from dermoscopic images using transfer learning on the HAM10000 dataset.

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 88.34% |
| Sensitivity (Melanoma Recall) | 87.56% |
| Specificity (Benign Recall) | 89.13% |
| AUC-ROC | 0.9559 |

## 🏗️ Architecture

- **Base Model**: ResNet50 pretrained on ImageNet
- **Approach**: Two-Stage Fine-Tuning
  - Stage 1: Train classification head (base frozen, 15 epochs)
  - Stage 2: Fine-tune top layers (30 epochs with early stopping)
- **Dataset**: HAM10000 — 19,128 balanced images (9,564 malignant + 9,564 benign)
- **Input Size**: 224×224 RGB

## 🚀 Run the Detection App

```bash
pip install streamlit tensorflow pillow numpy
streamlit run melanoma_detector_app.py
```

Upload any dermoscopy image and get instant malignant/benign prediction with confidence scores.

## 🏋️ Train the Model

```bash
pip install tensorflow scikit-learn pandas matplotlib seaborn
python melanoma_research_updated.py
```

Place `ham10000_metadata.csv` and ISIC images folder in the same directory before running.

## 📁 Files

| File | Description |
|------|-------------|
| `melanoma_research_updated.py` | Full training pipeline |
| `melanoma_detector_app.py` | Streamlit detection app |
| `confusion_matrix.png` | Model evaluation results |

## 🔧 Requirements

```
tensorflow>=2.10
streamlit
pillow
numpy
scikit-learn
pandas
matplotlib
seaborn
```

## ⚠️ Disclaimer

This tool is for **research purposes only** and is not intended for clinical diagnosis. Always consult a qualified dermatologist for medical advice.

## 👤 Author

**Aryan Harshadbhai Bhagat**  
MS Computer Science — Florida Atlantic University  
[LinkedIn](https://linkedin.com/in/aryan-bhagat-57474b280) | [GitHub](https://github.com/Aryanbhagat23)