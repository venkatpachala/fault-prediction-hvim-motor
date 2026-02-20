import pandas as pd
df = pd.read_csv('dataset/healthy/healthy_001.csv')
print(f"Time start: {df['t'].iloc[0]}")
print(f"Time end:   {df['t'].iloc[-1]}")
print(f"Time step:  {df['t'].iloc[1] - df['t'].iloc[0]}")
print(f"Samples:    {len(df)}")
print(f"Duration:   {df['t'].iloc[-1] - df['t'].iloc[0]:.4f} seconds")
print(f"Sampling rate: {1/(df['t'].iloc[1] - df['t'].iloc[0]):.0f} Hz")