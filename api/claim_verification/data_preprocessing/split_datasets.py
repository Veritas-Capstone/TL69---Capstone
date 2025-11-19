# Split averitec and fever into 80/20, random(seed set) by implementation for training, validation and testing

import pandas as pd
from sklearn.model_selection import train_test_split

# Input Paths
df1_path = "../data/unprocessed/averitec.csv"
df2_path = "../data/unprocessed/fever_train_claims.csv"
# Output Paths
df1_out_path = "../data/processed/averitec_80.csv"
df1_out_path_2 = "../data/processed/averitec_20.csv"
df2_out_path = "../data/processed/fever_train_claims_80.csv"
df2_out_path_2 = "../data/processed/fever_train_claims_20.csv"


# --- Load your two CSV files ---
df1 = pd.read_csv(df1_path)
df2 = pd.read_csv(df2_path)

# --- Split each dataset 80/20 ---
train1, test1 = train_test_split(df1, test_size=0.2, random_state=42, shuffle=True)
train2, test2 = train_test_split(df2, test_size=0.2, random_state=42, shuffle=True)

# --- Save to new CSV files ---
train1.to_csv(df1_out_path, index=False)
test1.to_csv(df1_out_path_2, index=False)

train2.to_csv(df2_out_path, index=False)
test2.to_csv(df2_out_path_2, index=False)

print("Done! Files saved:")
print(f"Saved to \n{df1_out_path}\n{df1_out_path_2}\n{df2_out_path}\n{df2_out_path_2}")
