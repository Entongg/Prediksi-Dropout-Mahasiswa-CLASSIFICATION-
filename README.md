# PREDIKSI DROPOUT MAHASISWA (CLASSIFICATION)

## 📌 Deskripsi Proyek

Proyek ini merupakan implementasi data mining menggunakan Python untuk melakukan analisis data, preprocessing, pemodelan machine learning, dan evaluasi model. Proyek ini bertujuan untuk mengolah dataset serta menghasilkan visualisasi yang membantu memahami hasil analisis dan kinerja model.

## 📂 Struktur Folder

```text
Project Data Mining/
│
├── Dataset/
│   ├── data.csv
│   └── data_preprocessed.csv
│
├── Visualisasi/
│   ├── confusion_matrix_comparison.png
│   ├── decision_tree_plot.png
│   ├── feature_importance.png
│   ├── Status_Mahasiswa (after).png
│   └── Status_Mahasiswa(after).png
│
├── evaluasi.py
├── modeling.py
├── processing.py
└── README.md
```

## 📊 Dataset

Folder `Dataset` berisi file data yang digunakan dalam proses analisis.

- **`data.csv`**: Dataset awal yang digunakan sebagai sumber data.
- **`data_preprocessed.csv`**: Dataset yang telah melalui tahap preprocessing dan siap digunakan untuk pemodelan.

## 🐍 File Program

### 1. `processing.py`

Digunakan untuk melakukan preprocessing data, seperti persiapan data sebelum pemodelan. Tahapan detail menyesuaikan implementasi yang terdapat dalam kode program.

### 2. `modeling.py`

Digunakan untuk membangun dan melatih model machine learning berdasarkan dataset yang telah diproses.

### 3. `evaluasi.py`

Digunakan untuk mengevaluasi hasil pemodelan guna mengetahui kinerja model berdasarkan metrik evaluasi yang diterapkan.

## 📈 Hasil Visualisasi

Folder `Visualisasi` menyimpan gambar hasil analisis dan pemodelan.

| File                              | Keterangan                                           |
| --------------------------------- | ---------------------------------------------------- |
| `confusion_matrix_comparison.png` | Perbandingan confusion matrix model                  |
| `decision_tree_plot.png`          | Visualisasi struktur pohon keputusan                 |
| `feature_importance.png`          | Visualisasi tingkat kepentingan atribut              |
| `Status_Mahasiswa (before).png`   | Visualisasi status mahasiswa sebelum pengolahan data |
| `Status_Mahasiswa(after).png`     | Visualisasi status mahasiswa setelah pengolahan data |

## ⚙️ Teknologi yang Digunakan

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 🚀 Cara Menjalankan Proyek

1. Pastikan Python telah terinstal di komputer.
2. Buka folder proyek melalui Visual Studio Code.
3. Instal library yang dibutuhkan:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

4. Jalankan preprocessing data:

   ```bash
   python processing.py
   ```

5. Jalankan proses pemodelan:

   ```bash
   python modeling.py
   ```

6. Jalankan evaluasi model:

   ```bash
   python evaluasi.py
   ```

**Catatan:** Urutan eksekusi di atas merupakan alur yang disarankan berdasarkan nama file. Pastikan setiap script menggunakan nama file, lokasi dataset, dan dependensi yang sesuai.
