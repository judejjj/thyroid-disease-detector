Thyroid Disease Detection Using Machine Learning
This is a mini-project aimed at detecting thyroid dysfunction (hypothyroidism, hyperthyroidism, or normal) using machine learning models trained on clinical lab data.

## 🔍 Project Overview
Predicts thyroid status based on patient lab parameters (TSH, T3, FTI, etc.).

Aims for a reliable classification between Negative, Hyperthyroid, Primary Hypothyroid, and Compensated Hypothyroid.

The entire workflow is built on tabular data, with no image processing involved.

The project is structured into two phases: (1) Data Preparation and (2) Model Training.

## ⚙️ Tech Stack
Language: Python

Notebooks: Jupyter Notebook

Core Libraries: Pandas, NumPy

Machine Learning: scikit-learn, imblearn

Serialization: pickle or joblib

## ✅ Key Methodologies
Data Combination: Merged allhyper.data and allhypo.data to create a single, comprehensive dataset.

Data Cleaning: Handled inconsistencies and missing values. Numerical features were imputed using the median for robustness against outliers.

Feature Encoding: Converted all categorical and binary features (e.g., 'sex', 'on thyroxine') into a fully numerical format suitable for machine learning.

Imbalance Handling: The project addresses severe class imbalance in the dataset, a critical step for building a fair and accurate model.

## 🤖 Algorithms & Techniques
The following algorithms and techniques are used in this project for classification and data balancing:

Decision Tree

Random Forest

SMOTE (Synthetic Minority Over-sampling Technique)

## 📊 Dataset
Source: UCI Machine Learning Repository

Raw Files: allhyper.data & allhypo.data

Final Cleaned File: thyroid_final_cleaned.csv

Samples: ~2,800 after cleaning and deduplication.

## 📝 License
This project is for academic use only.