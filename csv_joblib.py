import pandas as pd
import numpy as np
import json
import joblib
import os

csv_filename = "processed_embeddings.csv"
joblib_filename = "vector_database.joblib"

# 1. Verification
if not os.path.exists(csv_filename):
    print(f"Error: Could not find '{csv_filename}' in your directory.")
    print("Please check your sidebar file name matches exactly.")
    exit()

print(f"Reading '{csv_filename}'...")
df = pd.read_csv(csv_filename)

print("Converting embedding strings back into numeric numpy arrays...")
# Pandas reads CSV lists as standard text strings (e.g. "[0.12, -0.4]"). 
# This parsing step transforms them back into mathematical objects.
def parse_embedding(val):
    if isinstance(val, str):
        try:
            return np.array(json.loads(val))
        except Exception:
            return val
    return val

df['embedding'] = df['embedding'].apply(parse_embedding)

print(f"Saving optimized binary database to '{joblib_filename}'...")
# compress=3 keeps the file footprint compact on your disk
joblib.dump(df, joblib_filename, compress=3)

print("\n" + "="*50)
print("SUCCESS: Your vector database has been generated!")
print("You can now run your main read_chunks.py script, and it will load instantly.")
print("="*50)