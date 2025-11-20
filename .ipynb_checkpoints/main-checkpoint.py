import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

file_path = 'hcvdata.csv'
data = pd.read_csv(file_path)

print(f"Dataset berhasil dimuat: {data.shape[0]} baris, {data.shape[1]} kolom")
print(data.head())

plt.figure(figsize=(8, 5))
sns.countplot(x='Category', data=data, palette='viridis')
plt.title('Distribusi Kategori Pasien HCV')
plt.xlabel('Kategori')
plt.ylabel('Jumlah Pasien')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

X = data.drop('Category', axis=1)
y = data['Category']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_cols = X.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

def evaluate_model(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(f"Akurasi: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("Random Forest", y_test, y_pred_rf)

importances = rf_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\n=== 10 Fitur Terpenting (Random Forest) ===")
print(feat_imp.head(10))

feat_imp.head(10).plot(kind='barh', title='Top 10 Fitur Terpenting')
plt.show()

models = {
    'Logistic Regression': accuracy_score(y_test, y_pred_log),
    'Random Forest': accuracy_score(y_test, y_pred_rf)
}

plt.figure(figsize=(6, 4))
pd.Series(models).plot(kind='bar', color=['orange', 'green'])
plt.title('Perbandingan Akurasi Model')
plt.ylabel('Akurasi')
plt.ylim(0, 1)
plt.show()

cv_log = cross_val_score(log_model, X_scaled, y_encoded, cv=5)
cv_rf = cross_val_score(rf_model, X_scaled, y_encoded, cv=5)

print("\n=== Cross Validation (5-Fold) ===")
print(f"Logistic Regression CV Mean: {cv_log.mean():.4f}")
print(f"Random Forest CV Mean: {cv_rf.mean():.4f}")

best_model = rf_model if cv_rf.mean() > cv_log.mean() else log_model
joblib.dump(best_model, 'best_hcv_model.pkl')
print(f"\nModel terbaik disimpan sebagai 'best_hcv_model.pkl' ({type(best_model).__name__})")