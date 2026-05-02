# 🩺 NIH Chest X-Ray Multi-Label Classification using EfficientNet-B0

## 📌 Project Overview

This project focuses on building a **deep learning model for multi-label disease classification** using chest X-ray images from the NIH dataset.

Unlike traditional classification tasks, a single X-ray image can contain **multiple diseases simultaneously**, making this a **multi-label classification problem**.

The model leverages **EfficientNet-B0 with transfer learning** to predict 14 thoracic diseases.

---

## 📊 Dataset

* **Source:** NIH Chest X-ray Dataset
* **Total Images:** 112,120
* **Patients:** 30,805
* **Classes:** 14 diseases + *No Finding*

---

## 🎯 Problem Statement

Given a chest X-ray image, predict the presence of multiple diseases such as:

* Atelectasis
* Cardiomegaly
* Effusion
* Pneumonia
* Pneumothorax
* and more...

---

## ⚙️ Approach

### 1. Exploratory Data Analysis (EDA)

* Analyzed label distribution
* Identified **severe class imbalance**
* Visualized disease co-occurrence and patient statistics

### 2. Handling Class Imbalance

* Downsampled "No Finding" class
* Used **balanced subset (~70k images)** 
* Applied **weighted loss function (pos_weight)**

### 3. Data Preprocessing

* Image resizing (224x224)
* Data augmentation:

  * Random rotation
  * Horizontal flipping
  * Color jitter
  * Random erasing

### 4. Model Architecture

* **EfficientNet-B0 (pretrained on ImageNet)**
* Fine-tuned last layers
* Added custom classification head with dropout

### 5. Training Strategy

* Loss: `BCEWithLogitsLoss` (weighted)
* Optimizer: `AdamW`
* Scheduler: Cosine Annealing
* Mixed precision training (for speed)

### 6. Evaluation Metrics

* AUC-ROC (per class + mean)
* Precision-Recall curves
* F1 Score
* Confusion Matrix

---

## 📈 Results

| Metric                    | Value                     |
| ------------------------- | ------------------------- |
| **Mean AUC-ROC**          | **0.7545**                |
| **Macro F1 Score**        | 0.2442                    |
| **Best Performing Class** | Edema (AUC: 0.859)        |
| **Weakest Class**         | Infiltration (AUC: 0.660) |

---

## 🧠 Key Features

* Multi-label classification pipeline
* Patient-level data split (prevents leakage)
* Class imbalance handling using weighted loss
* Transfer learning with EfficientNet
* Grad-CAM visualization for interpretability

---

## ⚠️ Challenges Faced

### 1. Severe Class Imbalance

* Some classes had extremely low samples (e.g., Hernia)
* Imbalance ratio was ~87x between largest and smallest class 
  ✅ Solution:
* Downsampling majority class
* Weighted loss function

---

### 2. Large Dataset Size

* Training on full 112k images was computationally expensive
  ✅ Solution:
* Smart sampling (~70k subset)
* Used Kaggle GPU (T4/P100)

---

### 3. Multi-Label Complexity

* Images can contain multiple diseases simultaneously
  ✅ Solution:
* Used sigmoid activation + BCE loss instead of softmax

---

### 4. Data Leakage Risk

* Same patient images appearing in train & test
  ✅ Solution:
* **Patient-level splitting**

---

### 5. Model Overfitting

* Deep models easily overfit medical data
  ✅ Solution:
* Dropout layers
* Data augmentation
* Early stopping

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn

---

## 📂 Project Structure

```
.
├── notebook.ipynb
├── README.md
├── results_summary.csv
├── images/
└── best_model.pth
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook
```

---

## 🔍 Future Improvements

* Use larger models (EfficientNet-B4 / ViT)
* Hyperparameter tuning
* Better threshold optimization
* Use full dataset with distributed training

---

## 🙌 Acknowledgements

* NIH Chest X-ray Dataset
* PyTorch Team
* Kaggle for GPU support

---

## 📬 Contact

If you found this project useful, feel free to connect!
