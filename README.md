# Credit Card Fraud Detection using Data Mining

This repository contains a comprehensive data mining project focused on detecting fraudulent credit card transactions. The project utilizes a large-scale dataset to explore data imbalances, perform feature engineering, and evaluate machine learning models.

## 🚀 Project Overview
The goal of this assignment is to build a predictive model that can accurately distinguish between legitimate and fraudulent transactions. Given the extreme rarity of fraud (less than 1% of the data), the project emphasizes data preprocessing and performance metrics beyond simple accuracy.

## 📊 Dataset Summary
- **Training Set:** ~1.3 million transactions
- **Testing Set:** ~555,000 transactions
- **Target Variable:** `is_fraud` (0 for legitimate, 1 for fraud)
- **Fraud Rate:** ~0.58% (Highly Imbalanced)
- **Key Features:** Transaction amount, category, merchant, distance (lat/long), and customer demographics.

## 🛠️ Tech Stack
- **Language:** Python 3
- **Libraries:** - `pandas` & `numpy` (Data manipulation)
  - `matplotlib` & `seaborn` (Data visualization)
  - `scikit-learn` (Machine learning and preprocessing)

## 📈 Methodology

### 1. Exploratory Data Analysis (EDA)
We conducted detailed analysis to identify patterns in fraudulent behavior:
- **Class Balance:** Visualized the disparity between classes.
- **Categorical Trends:** Identified high-risk categories (e.g., online shopping, grocery).
- **Temporal Analysis:** Analyzed fraud occurrences by the hour of the day.

### 2. Preprocessing & Feature Engineering
- **Encoding:** Categorical variables were transformed using `LabelEncoder`.
- **Scaling:** Numeric features like `amt` and `city_pop` were normalized using `StandardScaler`.
- **Weighting:** Handled class imbalance by calculating weights for the minority class.

### 3. Machine Learning Models
We implemented and compared two main algorithms:
- **Logistic Regression:** Used as a baseline classifier.
- **Random Forest:** Employed to capture complex, non-linear relationships in the transaction data.

## 🏆 Results & Evaluation
The models were evaluated using:
- **Confusion Matrices:** To visualize Type I and Type II errors.
- **ROC-AUC Curves:** To measure the trade-off between sensitivity and specificity.
- **Summary Metrics:** Precision, Recall, and F1-Score (saved in `model_results_summary.csv`).

## 📁 Repository Structure
```text
├── Data_Minining_Fraud_Detection.ipynb   # Main Jupyter Notebook
├── model_results_summary.csv            # Final performance metrics
├── figures/                             # Generated plots and charts
│   ├── class_balance.png
│   ├── fraud_rate_by_category.png
│   ├── fraud_rate_by_hour.png
│   └── model_comparison.png
└── README.md                            # Project documentation
