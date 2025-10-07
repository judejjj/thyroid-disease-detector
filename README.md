# 🧠 Thyroid Disease Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/) [![License](https://img.shields.io/badge/License-Academic-lightgrey)](#)

Mini-project to detect **thyroid dysfunction** — **hypothyroidism**, **hyperthyroidism**, or **normal** — from clinical lab test data using **machine learning**. Designed for early and accurate diagnosis.

---

## 🔍 Project Overview

- Combined and cleaned raw **UCI Thyroid Disease files** (`allhypo.data`, `allhyper.data`).  
- Handled missing values with **median imputation** and encoded categorical/binary features (`sex`, treatment flags, etc.).  
- Created a **3-class target**: Normal / Hypothyroid / Hyperthyroid.  
- Dropped irrelevant columns (measurement flags, referral source, original diagnosis).  
- Final cleaned dataset saved as `thyroid_final_3class_data.csv`.  
- Trained and evaluated **weighted 19-feature XGBoost model** for multi-class thyroid classification.

---

## ⚙️ Tech Stack

**Python** | **Pandas**, **NumPy** | **scikit-learn**, **XGBoost** | **Jupyter Notebook** | **Git & GitHub**

---

## ✅ Workflow

### 1️⃣ Data Preparation

- Load raw `.data` files and standardize column names.  
- Clean numeric and categorical columns.  
- Encode binary/TF flags and `sex` column.  
- Impute missing numeric values with median.  
- Generate **3-class target**:  
  - Normal → 0  
  - Hypothyroid → 1  
  - Hyperthyroid → 2  
- Save final cleaned dataset → `thyroid_final_3class_data.csv`.

---

### 2️⃣ Model Training

- Split data into training and test sets.  
- Train **XGBoost** using **weighted 19 features**.  
- Evaluate model performance:  

| Metric    | Normal | Hypothyroid | Hyperthyroid |
|----------|--------|-------------|--------------|
| Precision | 0.97  | 0.98        | 0.98         |
| Recall    | 0.98  | 0.82        | 0.99         |
| F1-score  | 0.97  | 0.89        | 0.99         |

- **Training Accuracy:** 99%  
- **Test Accuracy:** 97%  
- Save trained model → `thyroid_model_weighted_19features.joblib`

---

### 3️⃣ Model Testing (Sample Patients)

Tested the **weighted 19-feature XGBoost model** with **3 new sample patients**:

| Sample | Prediction       | Notes                                           |
|--------|-----------------|------------------------------------------------|
| 1      | Hyperthyroid ✅   | Correct prediction                             |
| 2      | Normal ✅         | Correct prediction (mild hypothyroid handled as expected) |
| 3      | Normal ✅         | Correct prediction                             |

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository  
- **Samples:** Combined hypothyroid + hyperthyroid datasets  
- **Features:** 22 clinical features (numeric/binary) + target  
- **Target Classes:**  
  - 0 → Normal  
  - 1 → Hypothyroid  
  - 2 → Hyperthyroid  

---

## 📝 License

Academic use only.
