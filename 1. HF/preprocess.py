import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import pickle

# --- 1. Load feature metadata ---
with open('/mnt/user-data/uploads/nslkdd_features.json', 'r') as f:
    features_meta = json.load(f)

# Column names from metadata (41 features + label + difficulty)
column_names = [feat['name'] for feat in features_meta]

# --- 2. Load datasets ---
train_df = pd.read_csv('/mnt/user-data/uploads/KDDTrain_.txt', header=None, names=column_names)
test_df = pd.read_csv('/mnt/user-data/uploads/KDDTest_.txt', header=None, names=column_names)

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

# --- 3. Binary labels: normal=0, attack=1 ---
train_df['binary_label'] = (train_df['label'] != 'normal').astype(int)
test_df['binary_label'] = (test_df['label'] != 'normal').astype(int)

print(f"\nTrain label distribution:\n{train_df['binary_label'].value_counts()}")
print(f"\nTest label distribution:\n{test_df['binary_label'].value_counts()}")

# Save original labels for later analysis if needed
train_original_labels = train_df['label'].copy()
test_original_labels = test_df['label'].copy()

# Drop 'label' and 'difficulty' columns
train_df = train_df.drop(columns=['label', 'difficulty'])
test_df = test_df.drop(columns=['label', 'difficulty'])

# --- 4. One-hot encoding with combined feature information ---
categorical_cols = ['protocol_type', 'service', 'flag']

# Combine train and test to get consistent one-hot encoding
combined = pd.concat([train_df[categorical_cols], test_df[categorical_cols]], axis=0)
combined_onehot = pd.get_dummies(combined, columns=categorical_cols)

# Split back
train_onehot = combined_onehot.iloc[:len(train_df)].reset_index(drop=True)
test_onehot = combined_onehot.iloc[len(train_df):].reset_index(drop=True)

# Replace categorical columns with one-hot encoded versions
train_numeric = train_df.drop(columns=categorical_cols).reset_index(drop=True)
test_numeric = test_df.drop(columns=categorical_cols).reset_index(drop=True)

# Separate binary_label before merging
y_train = train_numeric['binary_label'].values
y_test = test_numeric['binary_label'].values
train_numeric = train_numeric.drop(columns=['binary_label'])
test_numeric = test_numeric.drop(columns=['binary_label'])

# Combine numeric features with one-hot encoded features
X_train_df = pd.concat([train_numeric, train_onehot], axis=1)
X_test_df = pd.concat([test_numeric, test_onehot], axis=1)

print(f"\nFeature columns after one-hot encoding: {X_train_df.shape[1]}")
print(f"X_train shape: {X_train_df.shape}")
print(f"X_test shape:  {X_test_df.shape}")

# --- 5. Standardization: fit on train, transform both ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df.values.astype(np.float64))
X_test = scaler.transform(X_test_df.values.astype(np.float64))

print(f"\nAfter standardization:")
print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
print(f"X_test shape:  {X_test.shape}, dtype: {X_test.dtype}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
print(f"X_train mean (should be ~0): {X_train.mean():.6f}")
print(f"X_train std  (should be ~1): {X_train.std():.6f}")

# --- 6. Save preprocessed data ---
np.save('/home/claude/X_train.npy', X_train)
np.save('/home/claude/X_test.npy', X_test)
np.save('/home/claude/y_train.npy', y_train)
np.save('/home/claude/y_test.npy', y_test)

# Save scaler and feature names for later tasks
with open('/home/claude/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('/home/claude/feature_names.pkl', 'wb') as f:
    pickle.dump(list(X_train_df.columns), f)

# Save original labels
np.save('/home/claude/train_original_labels.npy', train_original_labels.values)
np.save('/home/claude/test_original_labels.npy', test_original_labels.values)

print("\nPreprocessing complete! Files saved.")
print(f"Feature names (first 10): {list(X_train_df.columns)[:10]}")
print(f"Feature names (last 10):  {list(X_train_df.columns)[-10:]}")
