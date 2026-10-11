import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

print("=== MEMUAT DATA PREPROCESSING ===")
# Memuat data yang sudah bersih dari tahap sebelumnya
df = pd.read_csv('data_preprocessed.csv', sep=';')

# Memisahkan Fitur (X) dan Target (y)
X = df.drop('Target', axis=1)
y = df['Target']

print("=== PEMBAGIAN DATA (TRAIN-TEST SPLIT) ===")
# Melakukan scaling (penting untuk Logistic Regression)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Membagi data: 80% Latih (Train), 20% Uji (Test) dengan stratify
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Jumlah Data Latih: {X_train.shape[0]} baris")
print(f"Jumlah Data Uji: {X_test.shape[0]} baris")

print("\n=== INISIALISASI & PELATIHAN MODEL ===")
# Model Utama: Decision Tree (Menggunakan kriteria Entropy menyerupai C4.5)
# max_depth=4 digunakan agar pohon tidak terlalu kompleks dan mudah ditarik aturannya (insight)
dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)

# Model Pembanding 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Model Pembanding 2: Logistic Regression
lr_model = LogisticRegression(random_state=42)

# Melatih (Training) ketiga model menggunakan data latih
print("Sedang melatih Decision Tree...")
dt_model.fit(X_train, y_train)

print("Sedang melatih Random Forest...")
rf_model.fit(X_train, y_train)

print("Sedang melatih Logistic Regression...")
lr_model.fit(X_train, y_train)
print("Semua model berhasil dilatih!")

print("\n=== VISUALISASI POHON KEPUTUSAN ===")
# Membuat dan menyimpan visualisasi Decision Tree
plt.figure(figsize=(35, 15))
plot_tree(dt_model, 
          feature_names=X.columns, 
          class_names=['Graduate (0)', 'Dropout (1)'], 
          filled=True, 
          rounded=True, 
          fontsize=10,
          proportion=False)

plt.title("Visualisasi Decision Tree (Early Warning System Dropout)", fontsize=24)
plt.tight_layout()
# Menyimpan gambar
plt.savefig('decision_tree_plot.png', dpi=400, bbox_inches='tight')
plt.show()

print("Visualisasi Pohon Keputusan berhasil disimpan sebagai 'decision_tree_plot.png'")