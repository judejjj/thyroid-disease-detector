Thyroid Disease Detection Using Machine Learning

Mini-project to detect thyroid dysfunction (hypothyroidism, hyperthyroidism, or normal) from clinical lab data using machine learning. Designed for early and accurate diagnosis.

## 🔍 Project Overview

Combined and cleaned UCI Thyroid Disease raw files (allhyper.data, allhypo.data).

Handled missing values (median imputation) and encoded categorical features.

Augmented and balanced dataset to 3,768 samples using SMOTE and CTGAN, ensuring all classes are well-represented.

Trained and evaluated models (Decision Tree, Random Forest, XGBoost) for high accuracy.

Final tuned model saved as thyroid_model_clinical.pkl.

## ⚙️ Tech Stack

Python | Pandas, NumPy | scikit-learn, imblearn | Jupyter Notebook | Git & GitHub

## ✅ Workflow

Data Preparation (01_Data_Preparation.ipynb)

Merged raw files, cleaned, imputed, encoded features.

Balanced dataset (~3,768 samples) saved as thyroid_clinical_balanced.csv.

Model Training (02_Model_Training.ipynb)

Split data, applied SMOTE, trained and tuned models.

Final performance:

Training Accuracy: 99.97%

Test Accuracy: 97.48%

Classification metrics (precision / recall / f1-score):

Class 0: 0.97 / 0.98 / 0.97

Class 1: 0.98 / 0.82 / 0.89

Class 2: 0.98 / 0.99 / 0.99

Class 3: 0.95 / 0.98 / 0.96

## 🤖 Key Techniques

Decision Tree, Random Forest, XGBoost

SMOTE & CTGAN for data augmentation

Hyperparameter tuning, model saving, reproducible pipelines

## 📊 Dataset

Source: UCI Machine Learning Repository

Cleaned + balanced samples: 3,768

Features: 22 clinical + target (multi-class labels)

📝 License

Academic use only.
