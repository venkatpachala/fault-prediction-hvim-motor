import pandas as pd
df = pd.read_csv('extract_features.csv')
print(df.groupby('label')[['ia_rms','current_unbalance','wm_mean','ia_std']].mean())

import pandas as pd
df = pd.read_csv(r'C:\Users\venka_5gwzxwk\OneDrive\Desktop\Majorproject-B\dataset\stator_fault\stator_005.csv', header=None, nrows=5)
print(df)
print("Shape:", df.shape)