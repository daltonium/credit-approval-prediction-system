# Credit Card Approval Prediction System

A machine learning system that predicts credit card approval decisions with fairness-aware modeling and an interactive prediction interface.

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline for credit card approval prediction using the UCI Credit Approval dataset. The system includes data preprocessing, multiple model training, fairness auditing, and a user-friendly prediction interface.

### Key Features
- ✅ **89.37% prediction accuracy** (XGBoost model)
- ✅ **Fairness-aware ML** with bias mitigation
- ✅ **5 trained models** for comparison (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
- ✅ **Interactive prediction interface** with real-time credit decisions
- ✅ **Comprehensive evaluation** (accuracy, precision, recall, F1-score, ROC-AUC)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
git clone <your-repo-url>
cd credit-approval-prediction-system

text

2. **Create and activate virtual environment**
Windows
python -m venv myenv
myenv\Scripts\activate

Linux/Mac
python3 -m venv myenv
source myenv/bin/activate

text

3. **Install dependencies**
pip install -r requirements.txt

text

4. **Download the dataset**
- Place `credit_approval_data.csv` in the `data/` folder
- Or the script will fetch it automatically from UCI repository

---

## 📁 Project Structure

credit-approval-prediction-system/
├── data/ # Data files
│ ├── credit_approval_data.csv
│ ├── credit_approval_cleaned.csv
│ ├── credit_features_processed.csv
│ └── credit_labels_processed.csv
├── models/ # Trained models and transformers
│ ├── XGBoost_model.joblib
│ ├── feature_scaler.joblib
│ └── label_encoder_*.joblib
├── splits/ # Test/train splits
│ ├── X_test_for_eval.csv
│ └── y_test_for_eval.csv
├── scripts/ # Python scripts
│ ├── data_exploration_cleaning.py
│ ├── feature_engineering.py
│ ├── build_models.py
│ ├── evaluate_performance.py
│ ├── fairness_check.py
│ └── predict_interface.py
├── requirements.txt # Python dependencies
└── README.md # This file

text

---

## 🔧 Usage

### Running the Full Pipeline

Execute scripts in order:

1. Data exploration and cleaning
python scripts/data_exploration_cleaning.py

2. Feature engineering
python scripts/feature_engineering.py

3. Train models
python scripts/build_models.py

4. Evaluate performance
python scripts/evaluate_performance.py

5. Check fairness
python scripts/fairness_check.py

6. Make predictions
python scripts/predict_interface.py

text

### Quick Prediction (Skip to Interface)

If models are already trained:

python scripts/predict_interface.py

text

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **XGBoost** | **89.37%** | 91.2% | 89.1% | 90.1% |
| Random Forest | 88.41% | 89.8% | 88.3% | 89.0% |
| Logistic Regression | 88.41% | 88.5% | 88.7% | 88.6% |
| Gradient Boosting | 87.92% | 88.9% | 87.5% | 88.2% |
| Decision Tree | 84.06% | 83.2% | 84.5% | 83.8% |

---

## ⚖️ Fairness & Ethics

### Bias Mitigation
- **Problem Identified:** Initial model showed disparate impact (demographic parity ratio: 0.706)
- **Solution Applied:** Removed sensitive demographic features from training
- **Result:** Eliminated bias while maintaining 89%+ accuracy

### Responsible AI Practices
- Fairness auditing on protected attributes
- Transparent prediction explanations
- Documentation of ethical considerations

---

## 🎯 Key Results

### Before Fairness Mitigation
- XGBoost Accuracy: 89.86%
- Demographic Parity Ratio: **0.706** ⚠️ (bias detected)

### After Fairness Mitigation
- XGBoost Accuracy: 89.37%
- Sensitive feature removed
- **Bias eliminated** ✅

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Scikit-learn** - Model training and evaluation
- **XGBoost** - Gradient boosting models
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Joblib** - Model serialization

---

## 📖 Dataset

**Source:** [UCI Machine Learning Repository - Credit Approval](https://archive.ics.uci.edu/ml/datasets/credit+approval)

- **Samples:** 690
- **Features:** 15 (6 numerical, 9 categorical)
- **Target:** Binary (approved/denied)
- **Missing Values:** Handled via imputation

---

## 👥 Contributors

- **Mohith Dalton Jeyaram** - Project Intern @ Tech Trio

---

## 📝 License

This project is for educational purposes.

---

## 🔮 Future Improvements

- [ ] Web-based UI (Flask)
- [ ] Real-time API endpoint
- [ ] Advanced fairness metrics (equalized odds, calibration)
- [ ] Model explainability (SHAP, LIME)
- [ ] A/B testing framework

