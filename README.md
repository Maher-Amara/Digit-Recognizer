# Kaggle Digit Recognizer 🔢

![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-blue?style=for-the-badge&logo=python&logoColor=white)

This repository contains my participation in the [Kaggle Digit Recognizer Competition](https://www.kaggle.com/competitions/digit-recognizer). The goal of this competition is to correctly identify digits from a dataset of tens of thousands of handwritten images.

## 📊 Dataset

The data files `train.csv` and `test.csv` contain gray-scale images of hand-drawn digits, from zero through nine.

- **Image Size:** 28x28 pixels (784 pixels in total)
- **Training Set:** 42,000 images with labels
- **Test Set:** 28,000 images (labels to be predicted)

Data can be downloaded directly from the [Kaggle Data Page](https://www.kaggle.com/competitions/digit-recognizer/data).

## 🏆 Results

With this work, we managed to get a score of **0.99546**, securing a **Top 100** position on the Kaggle leaderboard!

## 💡 Approach & Methodology

Our goal is to build a robust architecture that generalizes well, strictly adhering to standard machine learning practices and aiming for a >99% validation accuracy without data leakage.

### 1. Strict Training & Evaluation

We train **only** on the provided 42,000 images in the training set and use a designated validation split for evaluation.

### 2. Model Architecture

We use a deep 3-block CNN architecture (32➔64➔128 filters) known to achieve >99.4% accuracy. *Batch Normalization* and heavy *Dropout* (0.25/0.50) are critical in this architecture to stabilize training and prevent overfitting on the limited dataset.

### 3. Careful Data Augmentation

To improve model robustness, we synthetically expand our training data. We carefully fine-tune these parameters to ensure digits are not accidentally cropped or structurally altered (e.g., flipping a `6` into a `9`):

- **Rotation Range:** Maximum 10 degrees.
- **Zoom Range:** 10% (0.1).
- **Width/Height Shift:** 10% (0.1).
- **Flips:** Strictly disabled.

### 4. Training Optimization

- **Optimizer:** `Adam`.
- **Callbacks:** We employ `ReduceLROnPlateau` (factor=0.5, patience=3) to dynamically adjust the learning rate when validation accuracy plateaus, and `EarlyStopping` to restore the best weights.

## 🔗 References & Resources

- [Competition Homepage](https://www.kaggle.com/competitions/digit-recognizer)
- [Competition Data](https://www.kaggle.com/competitions/digit-recognizer/data)
- [Discussion: Achieving 100% Accuracy (Exploit Context)](https://www.kaggle.com/competitions/digit-recognizer/discussion/61480)
- [Discussion: 99.47% using PyTorch from Scratch (No Cheating)](https://www.kaggle.com/competitions/digit-recognizer/discussion/728928)
- [Discussion: CNN Architecture & Augmentation Recommendations](https://www.kaggle.com/competitions/digit-recognizer/discussion/699592)
