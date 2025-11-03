import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def preprocess(df: pd.DataFrame):
    # Copy to avoid changing original
    X = df.copy()

    # Drop any identifier-like columns if present (none required, but keep Time/Amount)
    # Impute missing numeric values with median
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

    # Label column
    if 'Class' in X.columns:
        y = X['Class'].astype(int).values
        X = X.drop(columns=['Class'])
    else:
        raise ValueError('Expected a "Class" column as label')

    # There are no categorical columns in this dataset; if there were, encode them here.

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save feature names in original column order (useful for prediction)
    feature_names = X.columns.tolist()

    return X_scaled, y, imputer, scaler, feature_names


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Use a RandomForest with class_weight to help with imbalance
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print('Classification report:')
    print(classification_report(y_test, y_pred, digits=4))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('ROC AUC:', roc_auc_score(y_test, y_proba))

    return clf


def save_artifacts(model, imputer, scaler, feature_names, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, 'model.pkl'))
    joblib.dump(imputer, os.path.join(out_dir, 'imputer.pkl'))
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(out_dir, 'feature_names.pkl'))


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / 'data' / 'creditcard.csv'

    print('Loading data from', data_path)
    df = load_data(data_path)
    print('Dataset shape:', df.shape)

    X, y, imputer, scaler, feature_names = preprocess(df)
    print('After preprocessing: X shape =', X.shape, 'y shape =', y.shape)

    model = train_and_evaluate(X, y)

    models_dir = root / 'models'
    save_artifacts(model, imputer, scaler, feature_names, models_dir)
    print('Saved model and preprocessing artifacts to', models_dir)


if __name__ == '__main__':
    main()
