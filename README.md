Thyroid Disease Detection Using Machine Learning
This is a mini-project aimed at detecting thyroid dysfunction (hypothyroidism, hyperthyroidism, or normal) using machine learning models trained on clinical lab data.

## 🔍 Project Overview
This project develops a robust model to accurately classify thyroid conditions from clinical data. The primary goal is to provide a reliable tool for early and accurate diagnosis based on lab parameters.

Key Objectives:

Process and combine raw data files from the UCI Thyroid Disease dataset.

Handle significant data quality issues, including missing values and severe class imbalance.

Train and evaluate an interpretable, high-performance classification model.

Structure the project into a reproducible two-notebook pipeline (data prep and model training).

## ⚙️ Tech Stack
Language: Python

Core Libraries: Pandas, NumPy

Machine Learning: scikit-learn, imblearn

Development: Jupyter Notebook

Source Control: Git & GitHub

This stack was chosen as it represents the industry standard for data science, providing powerful, easy-to-use tools for every step of the machine learning workflow.

## ✅ Project Workflow & Methodology
The project is split into two distinct phases, each in its own notebook:

01_Data_Preparation.ipynb: This notebook handles all data cleaning. It takes the raw .data files and produces two clean outputs: thyroid_final_cleaned.csv and target_label_encoder.pkl.

02_Model_Training.ipynb: This notebook focuses on modeling. It loads the clean files, splits the data, applies SMOTE to the training set, trains the model, and evaluates its performance.

Key Processing Steps:

Data Combination: Merged allhyper.data and allhypo.data to create a single, comprehensive dataset.

Imputation: Filled missing numerical values (e.g., in TSH, FTI) using the median for robustness against outliers.

Encoding: Converted all categorical and binary features into a fully numerical format suitable for modeling.

## 🤖 Algorithms & Techniques
The following algorithms and techniques are used in this project for classification and data balancing:

Decision Tree

Random Forest

SMOTE (Synthetic Minority Over-sampling Technique)

## 📊 Dataset
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/102/thyroid+disease)

Raw Files: allhyper.data & allhypo.data

Final Cleaned File: thyroid_final_cleaned.csv

Samples: ~2,800 after cleaning and deduplication.

## 📝 License
This project is for academic use only.
