import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# LOAD DATA & TRAINING ULANG
print("Memuat data dan melatih ulang model...")
df = pd.read_csv('data_preprocessed.csv', sep=';')
X = df.drop('Target', axis=1)
y = df['Target']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
lr_model = LogisticRegression(random_state=42)

dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

print("\n=== EVALUASI KINERJA MODEL ===")

def evaluate_model(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred) # Kelas positif = 1 (Dropout)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n--- {model_name} ---")
    print(f"Accuracy  : {acc:.4f} ({(acc*100):.2f}%)")
    print(f"Precision : {prec:.4f} ({(prec*100):.2f}%)")
    print(f"Recall    : {rec:.4f} ({(rec*100):.2f}%)")
    print(f"F1-Score  : {f1:.4f} ({(f1*100):.2f}%)")
    
    return acc, prec, rec, f1

dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print("Menghitung metrik performa (Fokus pada kelas positif: Dropout = 1)")
dt_metrics = evaluate_model("Decision Tree (CART/C4.5)", y_test, dt_pred)
rf_metrics = evaluate_model("Random Forest", y_test, rf_pred)
lr_metrics = evaluate_model("Logistic Regression", y_test, lr_pred)

print("\n=== VISUALISASI CONFUSION MATRIX ===")

def plot_confusion_matrix(y_true, y_pred, model_name, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Graduate (0)', 'Dropout (1)'], 
                yticklabels=['Graduate (0)', 'Dropout (1)'])
    ax.set_title(f'Confusion Matrix: {model_name}')
    ax.set_xlabel('Prediksi Model')
    ax.set_ylabel('Status Sebenarnya (Aktual)')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_confusion_matrix(y_test, dt_pred, "Decision Tree", axes[0])
plot_confusion_matrix(y_test, rf_pred, "Random Forest", axes[1])
plot_confusion_matrix(y_test, lr_pred, "Logistic Regression", axes[2])

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
print("Visualisasi Confusion Matrix berhasil disimpan sebagai 'confusion_matrix_comparison.png'")
plt.show()

print("\n=== FEATURE IMPORTANCE (Untuk Insight Bisnis) ===")
feature_importances = pd.Series(dt_model.feature_importances_, index=X.columns)
top_features = feature_importances.sort_values(ascending=False).head(5)

print("\n5 Atribut Paling Berpengaruh (Berdasarkan Decision Tree):")
print(top_features)

plt.figure(figsize=(10, 6))
top_features.plot(kind='barh', color='teal')
plt.title('5 Atribut Paling Berpengaruh Terhadap Risiko Dropout (Decision Tree)')
plt.xlabel('Tingkat Kepentingan (Gini Importance)')
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nVisualisasi Feature Importance berhasil disimpan sebagai 'feature_importance.png'")
plt.show()