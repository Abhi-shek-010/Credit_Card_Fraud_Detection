import argparse
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


def load_artifacts(models_dir: Path):
    model = joblib.load(models_dir / 'model.pkl')
    imputer = joblib.load(models_dir / 'imputer.pkl')
    scaler = joblib.load(models_dir / 'scaler.pkl')
    feature_names = joblib.load(models_dir / 'feature_names.pkl')
    return model, imputer, scaler, feature_names


def predict_df(input_df: pd.DataFrame, model, imputer, scaler, feature_names):
    # Align input with the feature names the imputer was fitted on (if available)
    imputer_features = getattr(imputer, 'feature_names_in_', None)

    if imputer_features is not None:
        temp = input_df.copy()
        # Add any missing columns expected by the imputer (fill with NaN)
        for c in imputer_features:
            if c not in temp.columns:
                temp[c] = np.nan
        # Reorder to imputer's expected order
        temp = temp[imputer_features]
        # Transform (returns numpy array)
        X_imputed_arr = imputer.transform(temp)
        X_imputed = pd.DataFrame(X_imputed_arr, columns=imputer_features, index=temp.index)
        # Select columns that the model expects (feature_names)
        X_selected = X_imputed[feature_names]
    else:
        # If imputer has no feature metadata, assume input_df contains the model features
        missing = [c for c in feature_names if c not in input_df.columns]
        if missing:
            raise ValueError(f'Missing expected feature columns: {missing}')
        X_selected = input_df[feature_names].copy()
        # Impute
        X_imputed_arr = imputer.transform(X_selected)
        X_selected = pd.DataFrame(X_imputed_arr, columns=feature_names, index=X_selected.index)

    # Scale and predict
    X_scaled = scaler.transform(X_selected)

    preds = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else None

    out = input_df.copy()
    out['prediction'] = preds
    if proba is not None:
        out['fraud_probability'] = proba
    return out


def main():
    parser = argparse.ArgumentParser(description='Load trained model artifacts and predict on CSV rows')
    parser.add_argument('--input', '-i', required=True, help='Input CSV file to predict on')
    parser.add_argument('--output', '-o', required=True, help='Output CSV file to write predictions')
    parser.add_argument('--nrows', type=int, default=None, help='If set, read only the first N rows from input')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    models_dir = repo_root / 'models'

    if not models_dir.exists():
        raise FileNotFoundError(f'Models directory not found: {models_dir}')

    print('Loading artifacts from', models_dir)
    model, imputer, scaler, feature_names = load_artifacts(models_dir)

    print('Reading input CSV:', args.input)
    df = pd.read_csv(args.input, nrows=args.nrows)
    print('Input shape:', df.shape)

    # If 'Class' exists in input, keep it but don't use it for prediction (we select feature_names)
    result = predict_df(df, model, imputer, scaler, feature_names)

    out_path = Path(args.output)
    result.to_csv(out_path, index=False)
    print(f'Wrote predictions to {out_path} (n={len(result)})')
    print('Prediction counts:')
    print(result['prediction'].value_counts())


if __name__ == '__main__':
    main()
