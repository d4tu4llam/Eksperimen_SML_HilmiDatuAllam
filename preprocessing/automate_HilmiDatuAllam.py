from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import numpy as np

def preprocess_data(data_path, target_column, output_dir):
    # Menentukan fitur numerik dan kategoris
    data = pd.read_csv(data_path,sep=",")
    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_features = ['gender', 'hypertension', 'heart_disease']
    column_names = data.columns
    # Mendapatkan nama kolom tanpa kolom target
    column_names = data.columns.drop(target_column)

    # Pastikan target_column tidak ada di numeric_features atau categorical_features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    df = data.copy()
    # 1. Hapus duplikat
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"[INFO] Menghapus {duplicates:,} baris duplikat...")
        df = df.drop_duplicates().reset_index(drop=True)

    # 2. Drop kolom yang tidak dipakai (smoking_history tidak berkorelasi dengan diabetes)
    if 'smoking_history' in df.columns:
        df = df.drop('smoking_history', axis=1)

    # 3. Bersihkan gender: buang 'Other' menyederhanakan model, dan pada dataset ini other hanya 0.018% atau 18 baris.
    if 'gender' in df.columns:
        other_count = (df['gender'] == 'Other').sum()
        if other_count > 0:
            df = df[df['gender'] != 'Other'].reset_index(drop=True)

    if numeric_features:
        z_scores = np.abs(
            (df[numeric_features] - df[numeric_features].mean()) / df[numeric_features].std()
        )
        before_outlier = len(df)
        df = df[(z_scores < 3.5).all(axis=1)].reset_index(drop=True)
        outlier_removed = before_outlier - len(df)
        print(f"[INFO] Outlier dibuang: {outlier_removed:,} baris")

    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk fitur kategoris
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Memisahkan target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # Buat DataFrame
    train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_df['diabetes'] = y_train.values

    test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_df['diabetes'] = y_test.values

    # Buat folder
    os.makedirs(output_dir, exist_ok=True)

    # Simpan
    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path = os.path.join(output_dir, "test_processed.csv")
    prep_path = os.path.join(output_dir, "preprocessor.joblib")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    dump(preprocessor, prep_path)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='preprocessing/diabetes_prediction_dataset_preprocessing')
    args = parser.parse_args()

    preprocess_data(
    data_path=args.input,
    target_column='diabetes',
    output_dir=args.output_dir
)