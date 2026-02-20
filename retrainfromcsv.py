"""
Retrain model directly from YOUR actual CSV files
No dependency on extract_features.csv from another folder
"""
import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FOLDER = 'dataset'

# Map folder names to labels
CLASS_MAP = {
    'healthy': 0,
    'voltage_unbalance': 1,
    'broken_rotor_bar': 2,
    'stator_fault': 3
}
CLASS_NAMES = ['Healthy', 'Voltage Unbalance', 'Rotor Fault', 'Stator Fault']


def extract_19_features(filepath):
    """Extract same 19 features as extract_features.m"""
    df = pd.read_csv(filepath)
    
    # Drop time column if present
    if 't' in df.columns:
        df = df.drop(columns=['t'])
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
    
    ia = df['ia'].values.astype(float)
    ib = df['ib'].values.astype(float)
    ic = df['ic'].values.astype(float)
    Te = df['Te'].values.astype(float)
    wm = df['wm'].values.astype(float)
    
    # 1. Phase A current features
    ia_rms = np.sqrt(np.mean(ia**2))
    ia_std = np.std(ia)
    
    # Kurtosis (match MATLAB's kurtosis which subtracts 3)
    n = len(ia)
    ia_mean = np.mean(ia)
    m2 = np.mean((ia - ia_mean)**2)
    m4 = np.mean((ia - ia_mean)**4)
    ia_kurt = (m4 / (m2**2 + 1e-12)) - 3  # excess kurtosis
    
    ia_crest = np.max(np.abs(ia)) / (ia_rms + 1e-9)
    
    # FFT
    Y = np.abs(np.fft.fft(ia))
    Yh = Y[:len(Y)//2]
    ia_domFreq = np.argmax(Yh)
    
    # 2. Cross-phase features
    ib_rms = np.sqrt(np.mean(ib**2))
    ic_rms = np.sqrt(np.mean(ic**2))
    current_unbalance = np.std([ib_rms, ic_rms])
    
    # 3. Torque features
    Te_mean = np.mean(Te)
    Te_std = np.std(Te)
    Te_p2p = np.max(Te) - np.min(Te)
    
    # 4. Speed features
    wm_mean = np.mean(wm)
    wm_std = np.std(wm)
    wm_p2p = np.max(wm) - np.min(wm)
    
    # 5. Global features
    energy = np.sum(ia**2)
    fft_energy = np.sum(Y**2)
    torque_ripple = Te_std
    speed_ripple = wm_std
    thd_like = np.sum(Yh[4:]**2) / (np.sum(Yh**2) + 1e-9)
    
    return [ia_rms, ia_std, ia_kurt, ia_crest, ia_domFreq,
            ib_rms, ic_rms, current_unbalance,
            Te_mean, Te_std, Te_p2p,
            wm_mean, wm_std, wm_p2p,
            energy, fft_energy,
            torque_ripple, speed_ripple, thd_like]


FEATURE_NAMES = [
    'ia_rms', 'ia_std', 'ia_kurt', 'ia_crest', 'ia_domFreq',
    'ib_rms', 'ic_rms', 'current_unbalance',
    'Te_mean', 'Te_std', 'Te_p2p',
    'wm_mean', 'wm_std', 'wm_p2p',
    'energy', 'fft_energy',
    'torque_ripple', 'speed_ripple', 'thd_like'
]

# ============================================================
# STEP 1: Process all CSVs
# ============================================================
print("=" * 60)
print("EXTRACTING FEATURES FROM YOUR ACTUAL CSV FILES")
print("=" * 60)

all_features = []
all_labels = []
errors = []

for folder_name, label in CLASS_MAP.items():
    folder_path = os.path.join(DATA_FOLDER, folder_name)
    if not os.path.exists(folder_path):
        print(f"  WARNING: Folder not found: {folder_path}")
        continue
    
    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
    print(f"\n  {folder_name}: {len(csv_files)} files")
    
    for f in csv_files:
        try:
            feats = extract_19_features(f)
            all_features.append(feats)
            all_labels.append(label)
        except Exception as e:
            errors.append((f, str(e)))
            print(f"    ERROR in {os.path.basename(f)}: {e}")

X = np.array(all_features)
y = np.array(all_labels)

print(f"\n  Total samples: {len(y)}")
print(f"  Errors: {len(errors)}")

# Show what the features actually look like
print("\n" + "=" * 60)
print("FEATURE STATISTICS PER CLASS (from YOUR data)")
print("=" * 60)
df_check = pd.DataFrame(X, columns=FEATURE_NAMES)
df_check['label'] = y
print(df_check.groupby('label')[['ia_rms', 'ib_rms', 'ic_rms', 
                                   'current_unbalance', 'wm_mean', 'wm_std']].mean())

# Save the new extract_features.csv (overwrites old one)
df_check.to_csv('extract_features.csv', index=False)
print("\n  Saved: extract_features.csv (from YOUR actual CSVs)")

# ============================================================
# STEP 2: Train
# ============================================================
print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

print(f"\n  ACCURACY: {acc:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ============================================================
# STEP 3: Save
# ============================================================
joblib.dump(model, 'ml_model.pkl')
joblib.dump(FEATURE_NAMES, 'features.pkl')
print(f"  Saved: ml_model.pkl (trained on YOUR data)")
print(f"  Saved: features.pkl ({len(FEATURE_NAMES)} features)")

# ============================================================
# STEP 4: Confusion Matrix
# ============================================================
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix — Accuracy: {acc:.2f}%')
plt.tight_layout()
plt.savefig('ml_results.png', dpi=150)
plt.show()
print("  Saved: ml_results.png")

# ============================================================
# STEP 5: Feature Importance
# ============================================================
importances = model.feature_importances_
idx = np.argsort(importances)[::-1]
print("\n  Top 10 Features:")
for i in range(min(10, len(idx))):
    print(f"    {i+1}. {FEATURE_NAMES[idx[i]]:25s} {importances[idx[i]]:.4f}")

print("\n" + "=" * 60)
print("DONE — Now run: streamlit run dashboard.py")
print("=" * 60)