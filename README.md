# SMOTE-OSBNR: A Hybrid Sampling Method for Imbalanced Classification

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SMOTE-OSBNR is a **hybrid sampling technique** designed to address **class imbalance** in machine learning, combining:
- **SMOTE (Synthetic Minority Oversampling Technique)** for generating synthetic samples of minority classes.
- **OSBNR (One-Side Behavioral Noise Reduction)** for removing noisy or overlapping majority samples.

This method significantly improves classification performance on highly imbalanced datasets, particularly for **credit card fraud detection**.

---

## 📌 Features
- Handles extreme class imbalance effectively.
- Combines oversampling and noise reduction in a two-step process.
- Improves metrics like **G-mean**, **AUC-PR**, **AUC-ROC**, **Recall**, and **F1-score**.
- Validated on multiple datasets and classifiers (GBT, XGB, RF).

---

## 📂 Repository Structure

```plaintext
SMOTE-OSBNR/
├── README.md                # Project documentation
├── LICENSE                  # License file
├── requirements.txt         # Dependencies
├── smote_osbnr.py           # Core implementation
├── experiments/             # Jupyter notebooks for experiments
│   └── fraud_detection.ipynb
├── data/                    # (Optional: instructions for downloading datasets)
│   └── README.md
└── results/                 # Performance plots and metrics


