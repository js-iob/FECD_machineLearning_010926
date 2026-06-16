#Author: Kiran Bharat Gaikwad
#Created on: 09-09-2025

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, confusion_matrix
#from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Ensure text remains editable in Illustrator
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# === STEP 1: Load dataset ===
df = pd.read_csv("D:/IOB/Projects/ocularDiseases_ML/Sep_2025/1_expressionProfiles/FECD/expressionProfile_fecd_adjustedCount_090225_transpose.txt", sep="\t")
X_full = df.drop(['classification', 'hgnc_symbol'], axis=1)
y_full = df['classification']
feature_names = X_full.columns
X_log = np.log2(X_full + 1)

print(f"\nTotal dataset size: {X_log.shape[0]} samples")

# === STEP 2: Train/Test Split ===
X_temp, X_test, y_temp, y_test = train_test_split(X_log, y_full, test_size=0.2, stratify=y_full, random_state=42)
 
# Now: 60% train, 20% val, 20% test

print(f"Train set size: {X_temp.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# === STEP 3: Cross-validation on TRAIN only ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
selected_feature_sets = []
cv_auc = []
cv_acc = []
cv_f1 = []
cv_sens = []
cv_spec = []
print("\n=== Cross-validation Performance (Logistic Regression) ===")
for fold, (train_idx, val_idx) in enumerate(cv.split(X_temp, y_temp), start=1):
    X_train_raw, X_val_raw = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
    y_train_cv, y_val_cv = y_temp.iloc[train_idx], y_temp.iloc[val_idx]

    print(f"\nFold {fold}: Train = {len(train_idx)} samples, Validation = {len(val_idx)} samples")
    scaler_fs = StandardScaler()
    X_train_scaled_fs = scaler_fs.fit_transform(X_train_raw)
    X_val_scaled_fs = scaler_fs.transform(X_val_raw)

    
    # Feature selection
    selector = SelectFromModel(LogisticRegression(penalty='l1', C=10, solver='liblinear', random_state=42, class_weight='balanced'), prefit=False)
    selector.fit(X_train_scaled_fs, y_train_cv)
    mask = selector.get_support()
    fold_selected_features = X_train_raw.columns[mask]
    selected_feature_sets.append(set(fold_selected_features))

    X_train_sel = selector.transform(X_train_scaled_fs)
    X_val_sel = selector.transform(X_val_scaled_fs)
    '''
    # Scale + SMOTE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_val_scaled = scaler.transform(X_val_sel)
    #X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train_cv)
    '''
    # Train model
    model = LogisticRegression(penalty='l1', C=10, solver='liblinear', random_state=42, class_weight='balanced')
    model.fit(X_train_sel, y_train_cv)

    # Predict
    y_val_pred = model.predict(X_val_sel)
    y_val_prob = model.predict_proba(X_val_sel)[:,1]

    auc_fold = roc_auc_score(y_val_cv, y_val_prob)
    acc_fold = accuracy_score(y_val_cv, y_val_pred)
    f1_fold = f1_score(y_val_cv, y_val_pred)

    cm_fold = confusion_matrix(y_val_cv, y_val_pred)

    tn, fp, fn, tp = cm_fold.ravel()

    sens_fold = tp / (tp + fn)
    spec_fold = tn / (tn + fp)

    cv_auc.append(auc_fold)
    cv_acc.append(acc_fold)
    cv_f1.append(f1_fold)
    cv_sens.append(sens_fold)
    cv_spec.append(spec_fold)
    
print("\n==============================")
print("5-FOLD CV PERFORMANCE")
print("==============================")

print(
    f"AUC: {np.mean(cv_auc):.3f} ± {np.std(cv_auc):.3f}"
)

print(
    f"Accuracy: {np.mean(cv_acc):.3f} ± {np.std(cv_acc):.3f}"
)

print(
    f"F1-score: {np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}"
)

print(
    f"Sensitivity: {np.mean(cv_sens):.3f} ± {np.std(cv_sens):.3f}"
)

print(
    f"Specificity: {np.mean(cv_spec):.3f} ± {np.std(cv_spec):.3f}"
)


# === STEP 4: Stable feature intersection ===
from collections import Counter

counter = Counter()

for s in selected_feature_sets:
    counter.update(s)
stable_features = [
    gene for gene, count in counter.items()
    if count >= 4
]
with open('logregSelected_features.txt', 'w') as o:
    for feat in stable_features:
        o.write(feat + '\n')

# === STEP 5: Final model training (train+val) ===
X_final = X_temp[stable_features]
scaler = StandardScaler()
X_final_scaled = scaler.fit_transform(X_final)
#X_final_res, y_final_res = SMOTE(random_state=42).fit_resample(X_final_scaled, y_temp)
final_model = LogisticRegression(solver='liblinear', penalty='l1', C=10, random_state=42, class_weight='balanced')
final_model.fit(X_final_scaled, y_temp)

# === STEP 6: Test Evaluation ===
print(f"\nHeld-out Test Set: {X_test.shape[0]} samples")
print("Class distribution (test set):")
print(y_test.value_counts())

X_test_sel = X_test[stable_features]
X_test_scaled = scaler.transform(X_test_sel)

y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = final_model.predict(X_test_scaled)

print("\n=== Test Set Performance (Logistic Regression) ===")
print(classification_report(y_test, y_test_pred))
print("Test ROC AUC:", roc_auc_score(y_test, y_test_prob))

rng = np.random.RandomState(42)

n_bootstraps = 2000

boot_auc = []

y_test_np = np.array(y_test)
boot_auc = []
boot_acc = []
boot_sens = []
boot_spec = []
boot_prec = []
boot_f1 = []
for i in range(n_bootstraps):

    idx = rng.randint(
        0,
        len(y_test_np),
        len(y_test_np)
    )

    y_boot_true = y_test_np[idx]
    y_boot_pred = y_test_pred[idx]
    y_boot_prob = y_test_prob[idx]

    # AUC requires both classes
    if len(np.unique(y_boot_true)) < 2:
        continue

    boot_auc.append(
        roc_auc_score(
            y_boot_true,
            y_boot_prob
        )
    )

    boot_acc.append(
        accuracy_score(
            y_boot_true,
            y_boot_pred
        )
    )

    boot_prec.append(
        precision_score(
            y_boot_true,
            y_boot_pred,
            zero_division=0
        )
    )

    boot_f1.append(
        f1_score(
            y_boot_true,
            y_boot_pred,
            zero_division=0
        )
    )

    cm_boot = confusion_matrix(
        y_boot_true,
        y_boot_pred,
        labels=[0,1]
    )

    tn, fp, fn, tp = cm_boot.ravel()

    sens = (
        tp/(tp+fn)
        if (tp+fn) > 0
        else np.nan
    )

    spec = (
        tn/(tn+fp)
        if (tn+fp) > 0
        else np.nan
    )

    boot_sens.append(sens)
    boot_spec.append(spec)
def get_ci(metric_values):

    return (
        np.nanpercentile(metric_values, 2.5),
        np.nanpercentile(metric_values, 97.5)
    )
auc_ci = get_ci(boot_auc)
acc_ci = get_ci(boot_acc)
prec_ci = get_ci(boot_prec)
f1_ci = get_ci(boot_f1)
sens_ci = get_ci(boot_sens)
spec_ci = get_ci(boot_spec)

print("\n===== TEST SET PERFORMANCE =====")

print(
    f"AUC = {roc_auc_score(y_test, y_test_prob):.3f} "
    f"(95% CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f})"
)

print(
    f"Accuracy = {accuracy_score(y_test, y_test_pred):.3f} "
    f"(95% CI {acc_ci[0]:.3f}-{acc_ci[1]:.3f})"
)

print(
    f"Precision = {precision_score(y_test, y_test_pred):.3f} "
    f"(95% CI {prec_ci[0]:.3f}-{prec_ci[1]:.3f})"
)

print(
    f"F1-score = {f1_score(y_test, y_test_pred):.3f} "
    f"(95% CI {f1_ci[0]:.3f}-{f1_ci[1]:.3f})"
)

cm = confusion_matrix(y_test, y_test_pred)

tn, fp, fn, tp = cm.ravel()

sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)

print(
    f"Sensitivity = {sensitivity:.3f} "
    f"(95% CI {sens_ci[0]:.3f}-{sens_ci[1]:.3f})"
)

print(
    f"Specificity = {specificity:.3f} "
    f"(95% CI {spec_ci[0]:.3f}-{spec_ci[1]:.3f})"
)

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# === Save plots to PDF ===
with PdfPages("LogReg_rocAuc.pdf") as pdf1, PdfPages("LogReg_cm.pdf") as pdf2:

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
    pdf1.savefig(bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Control', "Disease"], yticklabels=["Control", "Disease"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix Heatmap - Test Set')
    plt.tight_layout()
    pdf2.savefig(bbox_inches='tight')
    plt.close()
