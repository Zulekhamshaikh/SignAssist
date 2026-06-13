# train_old_SAFE.py — FIXES GROUP SHUFFLE SPLIT REPORTING

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from StaticNorm import HandNormalizer132

OLD_DATA = 'master_static_gesture_data1.csv'
PIPELINE_FILE = 'asl_STATIC_OLD_SAFE_v3.joblib' # New version filename
ENCODER_FILE = 'le_OLD_SAFE.joblib'

print("Loading OLD 84K data (SAFE MODE)...")
df = pd.read_csv(OLD_DATA)
X = df.drop(columns=['label', 'signer']).values.astype(np.float32)
y = df['label'].values
signers = df['signer'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, ENCODER_FILE)

pipeline = ImbPipeline([
    ('norm', HandNormalizer132()),
    # TWEAK: Smaller k_neighbors for tighter synthetic clusters
    ('smote', SMOTE(random_state=42, k_neighbors=3)), 
    ('clf', RandomForestClassifier(
        n_estimators=500, # TWEAK: Reduced trees
        max_depth=50,     # TWEAK: Increased depth for finer boundaries
        class_weight='balanced_subsample',
        max_features='log2', # TWEAK: Changed feature selection strategy
        n_jobs=-1,
        random_state=42
    ))
])

print("\nSAFE CV - GroupShuffleSplit (20% test, COLLECTING ALL FOLDS FOR REPORT)...")
gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cv_scores = []
y_true_report, y_pred_report = [], [] # Initialize as lists to collect all results

for fold, (train_idx, test_idx) in enumerate(gss.split(X, y_enc, groups=signers)):
    if len(test_idx) == 0:
        print(f"  Fold {fold+1}: SKIPPED (empty test)")
        continue
        
    pipeline.fit(X[train_idx], y_enc[train_idx])
    acc = pipeline.score(X[test_idx], y_enc[test_idx])
    cv_scores.append(acc)
    print(f"  Fold {fold+1}: {acc:.3f}")
    
    # --- REPORT FIX: Collect predictions from ALL folds ---
    preds = pipeline.predict(X[test_idx])
    y_true_report.extend(y[test_idx])
    y_pred_report.extend(le.inverse_transform(preds))

# Convert lists to numpy arrays for the report
y_true_report = np.array(y_true_report)
y_pred_report = np.array(y_pred_report)

if len(cv_scores) == 0:
    raise ValueError("ALL FOLDS EMPTY — CHECK DATA")

print(f"\nSAFE CV MEAN: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

if len(y_true_report) > 0:
    print("\n=== CLASSIFICATION REPORT (ALL FOLDS) ===")
    print(classification_report(y_true_report, y_pred_report, digits=3))

    cm = confusion_matrix(y_true_report, y_pred_report, labels=le.classes_)
    # Handle the RuntimeWarning when a class has no actual samples in the set
    with np.errstate(invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(15, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("SAFE OLD DATA - ALL FOLDS REPORT")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("Static_gest_confusion_v4_all_folds.png", dpi=300) 
    plt.show()

print("\nLOSO...")
logo = LeaveOneGroupOut()
scores = []
for train_idx, test_idx in logo.split(X, y_enc, groups=signers):
    if len(test_idx) == 0:
        continue
    pipeline.fit(X[train_idx], y_enc[train_idx])
    acc = pipeline.score(X[test_idx], y_enc[test_idx])
    print(f"  {np.unique(signers[test_idx])[0]}: {acc:.3f}")
    scores.append(acc)
print(f"LOSO: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

pipeline.fit(X, y_enc)
joblib.dump(pipeline, PIPELINE_FILE)
print(f"\nSAVED → {PIPELINE_FILE}")
print("GSS REPORTING FIXED. RUNNING NEW TEST.")