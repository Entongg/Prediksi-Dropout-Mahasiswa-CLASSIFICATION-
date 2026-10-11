import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# LOAD DATASET
# Menggunakan sep=';' karena dari pengecekan awal, dataset ini dipisahkan oleh titik koma
df = pd.read_csv('data.csv', sep=';')

print("=== INFORMASI DATASET AWAL ===")
print(f"Jumlah baris dan kolom: {df.shape}")
print(df.info())

# EKSPLORASI DATA (EDA) - VISUALISASI TARGET AWAL
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Target', palette='viridis')
plt.title('Distribusi Status Mahasiswa (Sebelum Preprocessing)')
plt.xlabel('Status Akhir (Target)')
plt.ylabel('Jumlah Mahasiswa')
plt.show()

# PREPROCESSING
print("\n=== MULAI PREPROCESSING DATA ===")

# Filtering Kelas Target
# Menghapus kelas 'Enrolled'
df_clean = df[df['Target'] != 'Enrolled'].copy()
print(f"Jumlah baris setelah menghapus 'Enrolled': {df_clean.shape[0]}")

# Mengecek Missing Values dan Duplikasi
missing_values = df_clean.isnull().sum().sum()
duplicates = df_clean.duplicated().sum()
print(f"Total Missing Values: {missing_values}")
print(f"Total Data Duplikat: {duplicates}")

# Opsional: Jika ada duplikat, hapus dengan df_clean.drop_duplicates(inplace=True)

# Encoding Target Variabel
# Mengubah 'Dropout' menjadi 1 (Kelas yang ingin dideteksi) dan 'Graduate' menjadi 0
df_clean['Target'] = df_clean['Target'].map({'Dropout': 1, 'Graduate': 0})
print("\nDistribusi Target Setelah Encoding (1 = Dropout, 0 = Graduate):")
print(df_clean['Target'].value_counts())

# PEMBAGIAN DATA (SPLIT X & y)
X = df_clean.drop('Target', axis=1)
y = df_clean['Target']

# FEATURE SCALING (Opsional untuk Decision Tree, Wajib untuk KNN/LogReg)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# TRAIN-TEST SPLIT
# Membagi data menjadi 80% data latih (train) dan 20% data uji (test) dengan stratify 
# agar proporsi Dropout/Graduate seimbang di kedua set.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# VISUALISASI SETELAH PREPROCESSING (Untuk disisipkan di Laporan Bab 2.3)
plt.figure(figsize=(8, 5))
# Menambahkan x pada hue dan mengatur legend=False
sns.countplot(data=df_clean, x='Target', hue='Target', palette='magma', legend=False)
plt.title('Distribusi Status Mahasiswa (Setelah Preprocessing & Encoding)')
plt.xlabel('Status Akhir (1 = Dropout, 0 = Graduate)')
plt.ylabel('Jumlah Mahasiswa')
plt.show()

# MENYIMPAN DATASET HASIL PREPROCESSING
df_clean.to_csv('data_preprocessed.csv', sep=';', index=False)
print("\nDataset hasil preprocessing berhasil disimpan sebagai 'data_preprocessed.csv'")

print("\n=== HASIL PEMBAGIAN DATA ===")
print(f"Dimensi Data Latih (X_train): {X_train.shape}")
print(f"Dimensi Data Uji (X_test): {X_test.shape}")

