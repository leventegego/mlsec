import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from IDS import IntrusionDetector

# --- 1. Load preprocessed data ---
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
train_original_labels = np.load('train_original_labels.npy', allow_pickle=True)
test_original_labels = np.load('test_original_labels.npy', allow_pickle=True)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# --- 2. Create DataLoader ---
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# --- 3. Initialize model, loss, optimizer ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IntrusionDetector(input_dim=X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 4. Training loop ---
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), 'ids_model.pth')
print("\nModel saved to ids_model.pth")

# --- 5. Evaluation ---
model.eval()
with torch.no_grad():
    X_test_dev = X_test_t.to(device)
    y_pred_prob = model(X_test_dev).cpu().numpy()
    y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

y_true = y_test

def print_metrics(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")

# 5a. Full test set
print_metrics("Full Test Set", y_true, y_pred)

# 5b. Identify zero-day vs non-zero-day attacks
# Zero-day attacks: attack types in test that are NOT in train
train_attack_types = set(train_original_labels[train_original_labels != 'normal'])
test_attack_types = set(test_original_labels[test_original_labels != 'normal'])
zero_day_types = test_attack_types - train_attack_types

print(f"\nTrain attack types: {sorted(train_attack_types)}")
print(f"Test attack types:  {sorted(test_attack_types)}")
print(f"Zero-day types:     {sorted(zero_day_types)}")

# Zero-day mask: test samples whose original label is a zero-day attack type
is_zero_day = np.isin(test_original_labels, list(zero_day_types))
# Non-zero-day attacks: attacks that appeared in training
is_non_zero_day_attack = (y_test == 1) & (~is_zero_day)

# 5c. Zero-day attacks evaluation
if is_zero_day.sum() > 0:
    print_metrics("Zero-day Attacks", y_true[is_zero_day], y_pred[is_zero_day])
    print(f"  (Number of zero-day samples: {is_zero_day.sum()})")
else:
    print("\nNo zero-day attacks found in test set.")

# 5d. Non-zero-day attacks evaluation
if is_non_zero_day_attack.sum() > 0:
    non_zd_mask = (~is_zero_day)  # all non-zero-day samples (normal + known attacks)
    # But the task asks for accuracy on non-zero-day attacks specifically
    print_metrics("Non-zero-day Attacks (known attack types only)", 
                  y_true[is_non_zero_day_attack], y_pred[is_non_zero_day_attack])
    print(f"  (Number of non-zero-day attack samples: {is_non_zero_day_attack.sum()})")
