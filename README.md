# Eksperimen_SML_HilmiDatuAllam

Deskripsi Proyek:
Repository ini berisikan proyek Eksperimen Machine Learning.
proyek ini berfokus pada otomatisasi proses preprocessing data menggunakan Python dan integrasi workflow otomatis dengan Github Actions

Dataset yang digunakan adalah Diabetes Dataset, yang berisikan informasi mengenai informasi pasien diabetes

Struktur Folder:
```bash
📦Eksperimen_SML_HilmiDatuAllam
 ┣ 📂.github
 ┃ ┗ 📂workflows
 ┃ ┃ ┗ 📜main.yml
 ┣ 📂preprocessing
 ┃ ┣ 📂diabetes_prediction_dataset_preprocessing
 ┃ ┃ ┣ 📜preprocessor.joblib
 ┃ ┃ ┣ 📜test_processed.csv
 ┃ ┃ ┗ 📜train_processed.csv
 ┃ ┣ 📜automate_HilmiDatuAllam.py
 ┃ ┗ 📜Eksperimen_HilmiDatuAllam.ipynb
 ┣ 📜.gitattributes
 ┣ 📜diabetes_prediction_dataset_raw.csv
 ┗ 📜requirements.txt
```
Tahapan Preprocessing:
Scrips pada fioe automate_Satriana.py akan secara otomatis menjalankan tahapan-tahapan:
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

Automatisasi Workflow (Github Actions)
untuk file workflow terletak di
.github/workflows/main.yml

Alur workflow:
1. Checkout repository : mengambil seluruh isi repository
2. Set up Python 3.12.7 : menggunakan python versi 3.12.7
3. Install dependencies : upgrade pip python dan menginstal semua library pada requerements.txt
4. Menjalankan Script : automate_HilmiDatuAllam.py
5. Tampilkan isi folder hasil automate
6. Upload hasil preprocessing ke github sebagai artifact
7. Commit and Push otomatis hasil dataset baru ke repository
