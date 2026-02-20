import pandas as pd
import numpy as np

# Read stator CSV the CORRECT way
df = pd.read_csv('dataset/stator_fault/stator_007.csv')
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst 3 rows:")
print(df.head(3))

# Drop time column
if 't' in df.columns:
    df = df.drop(columns=['t'])

ia = df['ia'].values
ib = df['ib'].values
ic = df['ic'].values

ia_rms = np.sqrt(np.mean(ia**2))
ib_rms = np.sqrt(np.mean(ib**2))
ic_rms = np.sqrt(np.mean(ic**2))
unbal = np.std([ib_rms, ic_rms])

print(f"\nia_rms = {ia_rms:.2f}  (expected ~342 for stator)")
print(f"current_unbalance = {unbal:.4f}  (expected ~0.007 for stator)")