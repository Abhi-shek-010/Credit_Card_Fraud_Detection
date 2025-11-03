# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A machine learning project that detects fraudulent credit card transactions using a Random Forest classifier. The model achieves 96.1% precision and 75.5% recall on fraud detection, with an ROC-AUC of 0.957.

## Project Overview

This project implements a complete ML pipeline for credit card fraud detection, including:

- Data cleaning and preprocessing
- Feature scaling
- Model training with class imbalance handling
- Model evaluation and artifact saving
- Prediction script for new data
- Basic EDA notebook

## Results

On the test set (20% of data, stratified split):

```
              precision    recall  f1-score   support
           0     0.9996    0.9999    0.9998     56864
           1     0.9610    0.7551    0.8457        98

    accuracy                         0.9995     56962
   macro avg     0.9803    0.8775    0.9227     56962
weighted avg     0.9995    0.9995    0.9995     56962
```

- ROC-AUC: 0.957
- Fraud (class 1) precision: 0.961
- Fraud (class 1) recall: 0.755

## Project Structure

```
├── data/
│   └── creditcard.csv         # Dataset (not included in repo)
├── models/                    # Saved model artifacts
│   ├── model.pkl             # Trained RandomForest
│   ├── scaler.pkl           # Fitted StandardScaler
│   ├── imputer.pkl          # Fitted SimpleImputer
│   └── feature_names.pkl    # Feature column order
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis
├── src/
│   ├── train.py            # Training pipeline
│   └── predict.py          # Prediction script
└── requirements.txt        # Python dependencies
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Abhi-shek-010/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection
```

2. Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

### Training

Train the model (creates artifacts in `models/`):
```powershell
python src\train.py
```

### Making Predictions

Predict on new data:
```powershell
python src\predict.py --input data\creditcard.csv --output predictions.csv
```

Options:
- `--input`: Input CSV file path
- `--output`: Output CSV file path
- `--nrows`: Optional, process only N rows (e.g., `--nrows 1000`)

### Exploratory Analysis

Open the EDA notebook:
```powershell
jupyter notebook notebooks\eda.ipynb
```

## Dataset

The dataset contains credit card transactions made in September 2013 by European cardholders. Features V1-V28 are PCA-transformed numerical values, while 'Time' and 'Amount' are the original features.

- Total transactions: 284,807
- Features: 30 (28 PCA, Time, Amount)
- Target: Class (0: normal, 1: fraud)
- Imbalance: 0.172% fraud cases

## Model Details

- Algorithm: Random Forest Classifier
- Trees: 200
- Class imbalance handling: class_weight='balanced'
- Feature preprocessing:
  - Missing values: Median imputation
  - Scaling: StandardScaler

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

