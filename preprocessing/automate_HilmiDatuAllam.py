from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import numpy as np
import os

def preprocess_data(data_path, target_column, output_dir):
    """
    Fungsi utama: Preprocessing otomatis dataset diabetes 

    Tahapan preprocessing : 
    1. Load file CSV
    2. Hapus duplikat
    3. Hapus kolom smoking_history (tidak memiliki korelasi dengan target default : diabetes)
    4. Hapus baris dengan gender = 'Other' (cuma 18 baris)
    5. Deteksi dan Hapus outlier pake Z-score > 3.5
    6. Pisah fitur numerik & kategorikal
    7. Scaling + OneHotEncoding pake Pipeline
    8. Split train/test (70:30)
    9. Simpan hasil ke folder:
         - train_processed.csv
         - test_processed.csv
         - preprocessor.joblib (untuk inference nanti)
    10. Buat Folder dan Simpan hasil

    """
    # 1. Load data dari file CSV
    data = pd.read_csv(data_path,sep=",")
    print("Data berhasil di load")

    # Menentukan fitur numerik dan kategorik
    numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_features = ['gender', 'hypertension', 'heart_disease']
    
    # Pastikan target_column tidak ada di numeric_features atau categorical_features
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    df = data.copy()

    # 2. Hapus duplikat
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"[INFO] Menghapus {duplicates:,} baris duplikat...")
        df = df.drop_duplicates().reset_index(drop=True)

    # 3. Drop kolom yang tidak dipakai (smoking_history tidak berkorelasi dengan diabetes)
    if 'smoking_history' in df.columns:
        df = df.drop('smoking_history', axis=1)

    # 4. Bersihkan gender: buang 'Other' menyederhanakan model, dan pada dataset ini other hanya 0.018% atau 18 baris.
    if 'gender' in df.columns:
        other_count = (df['gender'] == 'Other').sum()
        if other_count > 0:
            df = df[df['gender'] != 'Other'].reset_index(drop=True)

    # 5. Hapus outlier pake Z-score > 3.5
    if numeric_features:
        z_scores = np.abs(
            (df[numeric_features] - df[numeric_features].mean()) / df[numeric_features].std()
        )
        before_outlier = len(df)
        df = df[(z_scores < 3.5).all(axis=1)].reset_index(drop=True)
        outlier_removed = before_outlier - len(df)
        print(f"[INFO] Outlier dibuang: {outlier_removed:,} baris")
    
    # 6. Pipeline preprocessing
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
            ( 'cat',categorical_transformer, categorical_features)
        ],
        verbose_feature_names_out=False,
        remainder='passthrough'
    )

    # 7. Split data train & test
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 8. Transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    # 9. Gabung lagi jadi DataFrame + tambah kolom target
    train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_df['diabetes'] = y_train.values

    test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_df['diabetes'] = y_test.values

    # 10. Buat folder & simpan hasil
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train_processed.csv")
    test_path = os.path.join(output_dir, "test_processed.csv")
    prep_path = os.path.join(output_dir, "preprocessor.joblib")
    print(f"\n Preprocessing selesai")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    dump(preprocessor, prep_path)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
    help = 'Path ke file csv')
    parser.add_argument('--target', type=str, required=True,
    help = 'Nama kolom target default : diabetes')
    
    parser.add_argument('--output-dir', type=str, default='preprocessing/diabetes_prediction_dataset_preprocessing',
    help = 'Path output file')
    args = parser.parse_args()

    preprocess_data(
    data_path=args.input,
    target_column=args.target,
    output_dir=args.output_dir
)