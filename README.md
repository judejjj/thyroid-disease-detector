# 🧠 Thyroid Disease Detection Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/) [![License](https://img.shields.io/badge/License-Academic-lightgrey)](#)

Mini-project to detect **thyroid dysfunction** — **hypothyroidism**, **hyperthyroidism**, or **normal** — from clinical lab test data using **machine learning**. Designed for early and accurate diagnosis.

---

## 🚀 Quick Start

### Step 1: Install Python
Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/)

**Important:** Check ✅ "Add Python to PATH" during installation!

### Step 2: Download & Extract
1. Download this repository (click Code → Download ZIP)
2. Extract the ZIP file to any folder

### Step 3: Initial Setup (Run Once)
**Double-click `SETUP.bat`** 
- This installs all dependencies (Flask, XGBoost, scikit-learn, etc.)
- Takes 5-10 minutes on first run
- Wait for "Setup Complete!" message

### Step 4: Run the Application
**Double-click `RUN.bat`**
- Server starts automatically
- Browser opens at `http://localhost:5000`
- Enter patient data and get predictions!
- Press Ctrl+C to stop the server when done

---

## 📦 Alternative: Standalone EXE

A standalone EXE (~412 MB) is available for users who prefer not to install Python:

📥 **[Download from Google Drive](YOUR_DRIVE_LINK_HERE)**

⚠️ **Note:** This is a large file. For the lightweight option (~10 MB), use the BAT files above instead!

---

## 🔍 Project Overview

- Combined and cleaned raw **UCI Thyroid Disease files** (`allhypo.data`, `allhyper.data`)
- Handled missing values with **median imputation** and encoded categorical/binary features
- Trained **weighted 19-feature XGBoost model** for multi-class thyroid classification

---

## ⚙️ Tech Stack

**Python 3.8+** | **Flask** | **XGBoost** | **scikit-learn** | **Pandas**, **NumPy** | **Jupyter Notebook**

---

## ✅ Model Performance

**Accuracy:** 97% Test Accuracy

| Metric    | Normal | Hypothyroid | Hyperthyroid |
|----------|--------|-------------|--------------|
| Precision | 0.97  | 0.98        | 0.98         |
| Recall    | 0.98  | 0.82        | 0.99         |
| F1-score  | 0.97  | 0.89        | 0.99         |

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository  
- **Features:** 19 clinical features (numeric/binary)
- **Target Classes:** Normal (0), Hypothyroid (1), Hyperthyroid (2)

---

## 📝 License

Academic use only.

