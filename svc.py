#Author: Kiran Bharat Gaikwad
#Created on: 09-09-25

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# === STEP 1: Load dataset ===
df = pd.read_csv("D:/IOB/Projects/ocularDiseases_ML/Sep_2025/1_expressionProfiles/FECD/expressionProfile_fecd_adjustedCount_090225_transpose.txt", sep="\t")
X_full = df.drop(['classification', 'hgnc_symbol'], axis=1)
y_full = df['classification']
feature_names = X_full.columns
X_log = np.log2(X_full + 1)

print(f"\nTotal dataset size: {X_log.shape[0]} samples")

# === STEP 2: Train/Val/Test Split ===
X_temp, X_test, y_temp, y_test = train_test_split(X_log, y_full, test_size=0.2, stratify=y_full, random_state=42)
#X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)  
# Now: 60% train, 20% val, 20% test

print(f"Train set size: {X_temp.shape[0]} samples")
print(f"Validation set size: {X_test.shape[0]} samples")

# === STEP 3: Cross-validation on TRAIN only ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
selected_feature_sets = []

print("\n=== Cross-validation Performance (SVM) ===")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_temp, y_temp), start=1):
    X_train_raw, X_val_raw = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
    y_train_cv, y_val_cv = y_temp.iloc[train_idx], y_temp.iloc[val_idx]

    print(f"\nFold {fold}: Train = {len(train_idx)} samples, Validation = {len(val_idx)} samples")

    # Feature selection
    selector = SelectFromModel(SVC(kernel='linear', probability=True, C=6, random_state=42), threshold='median')
    selector.fit(X_train_raw, y_train_cv)
    mask = selector.get_support()
    fold_selected_features = feature_names[mask]
    selected_feature_sets.append(set(fold_selected_features))

    X_train_sel = selector.transform(X_train_raw)
    X_val_sel = selector.transform(X_val_raw)

    # Scale + SMOTE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train_cv)

    # Train model
    model = SVC(kernel='poly', probability=True, C=6, random_state=42)
    model.fit(X_train_res, y_train_res)

    # Predict
    y_val_pred = model.predict(X_val_scaled)
    print(classification_report(y_val_cv, y_val_pred))

# === STEP 4: Stable feature intersection ===
stable_features = list(set.intersection(*selected_feature_sets))
with open('svmSelected_features.txt', 'w') as o:
    for feat in stable_features:
        o.write(feat + '\n')

# === STEP 5: Final model training (train+val) ===
X_final = X_temp[stable_features]
scaler = StandardScaler()
X_final_scaled = scaler.fit_transform(X_final)
X_final_res, y_final_res = SMOTE(random_state=42).fit_resample(X_final_scaled, y_temp)
final_model = SVC(kernel='poly', probability=True, C=6, random_state=42)
final_model.fit(X_final_res, y_final_res)


# === STEP 6: Test Evaluation ===
print(f"\nHeld-out Test Set: {X_test.shape[0]} samples")
print("Class distribution (test set):")
print(y_test.value_counts())

X_test_sel = X_test[stable_features]
X_test_scaled = scaler.transform(X_test_sel)

y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = final_model.predict(X_test_scaled)

print("\n=== Test Set Performance (SVM) ===")
print(classification_report(y_test, y_test_pred))
print("Test ROC AUC:", roc_auc_score(y_test, y_test_prob))

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# === Save plots to PDF ===
with PdfPages("SVM_rocAuc.pdf") as pdf1, PdfPages("SVM_cm.pdf") as pdf2:

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Test ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    pdf1.savefig(bbox_inches='tight')   # Save current figure into PDF
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', "FECD"], yticklabels=["Control", "FECD"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap - Test Set')
    plt.tight_layout()
    pdf2.savefig(bbox_inches='tight')   # Save current figure into PDF
    plt.close()
