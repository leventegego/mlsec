import sys
sys.stdout.reconfigure(encoding="utf-8")
import functools
print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = 'cpu'
print(f"Device: {device}")

# ── 0. ADAT BETÖLTÉS ─────────────────────────────────────────────────────────
with open('../HW1/nslkdd_features.json', 'r') as f:
    features_info = json.load(f)
col_names = [f['name'] for f in features_info]

train_df = pd.read_csv('../HW1/NSL-KDD/KDDTrain+.txt', header=None, names=col_names)
test_df  = pd.read_csv('../HW1/NSL-KDD/KDDTest+.txt',  header=None, names=col_names)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

train_df['binary_label'] = (train_df['label'] != 'normal').astype(int)
test_df['binary_label']  = (test_df['label']  != 'normal').astype(int)
print("Train label dist:", dict(train_df['binary_label'].value_counts()))
print("Test  label dist:", dict(test_df['binary_label'].value_counts()))

# One-hot encoding
categorical_cols = ['protocol_type', 'service', 'flag']
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
combined_enc = pd.get_dummies(combined, columns=categorical_cols)
train_enc = combined_enc.iloc[:len(train_df)].copy()
test_enc  = combined_enc.iloc[len(train_df):].copy().reset_index(drop=True)

drop_cols   = ['label', 'difficulty', 'binary_label']
feature_cols = [c for c in train_enc.columns if c not in drop_cols]
X_train_full = train_enc[feature_cols].values.astype(np.float32)
y_train_full = train_enc['binary_label'].values.astype(np.float32)
X_test_all   = test_enc[feature_cols].values.astype(np.float32)
y_test_all   = test_enc['binary_label'].values.astype(np.float32)
print(f"Feature dim: {X_train_full.shape[1]}")

# StandardScaler + 80/20 split
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full).astype(np.float32)
X_test_all   = scaler.transform(X_test_all).astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full
)
feat_min = X_train.min(axis=0)
feat_max = X_train.max(axis=0)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test_all.shape}")

# ── MODELL ────────────────────────────────────────────────────────────────────
class IntrusionDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x)

def train_ids(X_tr, y_tr, epochs=10, batch_size=128, lr=0.001, seed=None):
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    X_t = torch.FloatTensor(X_tr)
    y_t = torch.FloatTensor(y_tr).unsqueeze(1)
    loader  = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    model   = IntrusionDetector(X_tr.shape[1])
    crit    = nn.BCEWithLogitsLoss()
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad(); loss = crit(model(bx), by); loss.backward(); opt.step()
    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return (torch.sigmoid(model(torch.FloatTensor(X))) > 0.5).float().numpy().flatten()

def get_accuracy(model, X, y): return accuracy_score(y, predict(model, X))

# ── CLEAN MODEL + 5 SURROGATES ───────────────────────────────────────────────
print("\nClean modell tanítása...")
clean_model = train_ids(X_train, y_train, seed=SEED)
clean_val_acc = get_accuracy(clean_model, X_val, y_val)
print(f"Clean val acc: {clean_val_acc:.4f}")

print("\n5 surrogate modell tanítása a tesztadaton...")
surrogates = []
for i in range(5):
    s = train_ids(X_test_all, y_test_all, seed=i * 1000 + 42)
    s_acc = get_accuracy(s, X_test_all, y_test_all)
    print(f"  Surrogate {i+1}: test acc = {s_acc:.4f}")
    surrogates.append(s)

# ── 1. FELADAT: UNTARGETED POISONING ─────────────────────────────────────────
print("\n" + "="*70)
print("1. FELADAT: Untargeted Poisoning")
print("="*70)

p_values = [30, 50, 70]

# Q1: Random label flipping
print("\nQ1: Random label flipping")
q1_results = {p: [] for p in p_values}
for p in p_values:
    for trial in range(5):
        seed_t = trial * 111 + p
        np.random.seed(seed_t)
        n_poison = int(len(y_train) * p / 100)
        idx = np.random.choice(len(y_train), size=n_poison, replace=False)
        Xp = np.concatenate([X_train, X_train[idx]])
        yp = np.concatenate([y_train, 1.0 - y_train[idx]])
        m  = train_ids(Xp, yp, seed=seed_t)
        va = get_accuracy(m, X_val, y_val)
        q1_results[p].append(va)
    print(f"  p={p}%: mean={np.mean(q1_results[p]):.4f} ± {np.std(q1_results[p]):.4f}  "
          f"values={[round(v,4) for v in q1_results[p]]}")

# Q2: Loss-based label flipping
print("\nQ2: Loss-based label flipping")
crit_none = nn.BCEWithLogitsLoss(reduction='none')
X_t_tr = torch.FloatTensor(X_train)
y_inv  = torch.FloatTensor(1.0 - y_train).unsqueeze(1)

all_losses = []
for sm in surrogates:
    sm.eval()
    with torch.no_grad():
        l = crit_none(sm(X_t_tr), y_inv).numpy().flatten()
        all_losses.append(l)
avg_inv_loss = np.mean(all_losses, axis=0)
print(f"  Avg inv-loss stats: mean={avg_inv_loss.mean():.4f}, "
      f"min={avg_inv_loss.min():.4f}, max={avg_inv_loss.max():.4f}")

q2_results = {p: [] for p in p_values}
for p in p_values:
    n_poison = int(len(y_train) * p / 100)
    idx = np.argsort(avg_inv_loss)[-n_poison:]
    Xp = np.concatenate([X_train, X_train[idx]])
    yp = np.concatenate([y_train, 1.0 - y_train[idx]])
    for trial in range(5):
        seed_t = trial * 222 + p
        m  = train_ids(Xp, yp, seed=seed_t)
        va = get_accuracy(m, X_val, y_val)
        q2_results[p].append(va)
    print(f"  p={p}%: mean={np.mean(q2_results[p]):.4f} ± {np.std(q2_results[p]):.4f}  "
          f"values={[round(v,4) for v in q2_results[p]]}")

# Box plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
colors = ['#4C72B0', '#55A868', '#C44E52']
for ax, data, title in [
    (axes[0], q1_results, 'Q1: Véletlenszerű (Random) Label Flipping'),
    (axes[1], q2_results, 'Q2: Loss-alapú Label Flipping'),
]:
    bp = ax.boxplot([data[p] for p in p_values],
                    labels=[f'p={p}%' for p in p_values], patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(clean_val_acc, color='red', ls='--', lw=2,
               label=f'Clean: {clean_val_acc:.4f}')
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Validációs Accuracy')
    ax.set_xlabel('Poison százalék (p)')
    ax.legend(loc='lower left'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('boxplot_task1.png', dpi=150, bbox_inches='tight')
print("\nBoxplot mentve: boxplot_task1.png")

# ── 2. FELADAT: TARGETED POISONING (WiB) ─────────────────────────────────────
print("\n" + "="*70)
print("2. FELADAT: Targeted Poisoning (Witches' Brew)")
print("="*70)

# Target minták kiválasztása
print("\nTarget minták kiválasztása...")
clean_model.eval()
clean_preds = predict(clean_model, X_test_all)
attack_correct_mask = (y_test_all == 1) & (clean_preds == 1)
candidate_idx = np.where(attack_correct_mask)[0]
print(f"Helyesen klasszifikált attack minták a tesztben: {len(candidate_idx)}")

with torch.no_grad():
    X_cand_t = torch.FloatTensor(X_test_all[candidate_idx])
    y_adv_cand = torch.zeros(len(candidate_idx), 1)
    out_cand = clean_model(X_cand_t)
    loss_normal = nn.BCEWithLogitsLoss(reduction='none')(out_cand, y_adv_cand)
    loss_normal = loss_normal.numpy().flatten()

top10 = np.argsort(loss_normal)[:10]
target_indices = candidate_idx[top10]
print(f"10 target index: {target_indices}")
print(f"Loss (y=0): {np.round(loss_normal[top10], 4)}")

X_targets = X_test_all[target_indices]
y_adv     = 0.0

# Gradiens segédfüggvény
def get_param_grad(model, x, y_label, create_graph=False):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)
    x_in = x.unsqueeze(0) if x.dim() == 1 else x
    if not create_graph:
        x_in = x_in.detach()
    x_in = x_in.to(next(model.parameters()).device)
    y_in = torch.tensor([[float(y_label)]], dtype=torch.float32, device=x_in.device)
    out  = model(x_in)
    loss = nn.BCEWithLogitsLoss()(out, y_in)
    params = [p for p in model.parameters() if p.requires_grad]
    grads  = torch.autograd.grad(loss, params, create_graph=create_graph)
    return torch.cat([g.flatten() for g in grads])

# Base minták kiválasztása (max 15, sorted desc)
def select_base_samples(target_x, y_adv_t, X_test, y_test, surrogates, p_count):
    cand_mask = (y_test == y_adv_t)
    cand_idx  = np.where(cand_mask)[0]
    print(f"    Kandidátus base minták: {len(cand_idx)}")

    target_grads = []
    x_t = torch.FloatTensor(target_x)
    for model in surrogates:
        g = get_param_grad(model, x_t, y_adv_t, create_graph=False)
        target_grads.append(g.detach())
        for param in model.parameters():
            param.requires_grad_(False)

    avg_sims = np.zeros(len(cand_idx))
    for bi in range(len(cand_idx)):
        ci = cand_idx[bi]
        x_c = torch.FloatTensor(X_test[ci])
        sims = []
        for j, model in enumerate(surrogates):
            g_c = get_param_grad(model, x_c, float(y_test[ci]), create_graph=False)
            for param in model.parameters():
                param.requires_grad_(False)
            sims.append(F.cosine_similarity(
                target_grads[j].unsqueeze(0), g_c.unsqueeze(0)
            ).item())
        avg_sims[bi] = np.mean(sims)
        if (bi + 1) % 1000 == 0:
            print(f"      Progress: {bi+1}/{len(cand_idx)}")

    top_p_idx = np.argsort(avg_sims)[-p_count:][::-1]
    selected  = cand_idx[top_p_idx]
    sims_sel  = avg_sims[np.argsort(avg_sims)[-p_count:][::-1]]
    print(f"    Top-{p_count} cos sim range: [{sims_sel[-1]:.4f}, {sims_sel[0]:.4f}]")
    return selected

# WiB PGD attack
def witches_brew_attack(target_x, y_adv_t, base_indices, X_test, y_test,
                        surrogates, feat_min_np, feat_max_np, max_iter=1000):
    p    = len(base_indices)
    fmin = torch.FloatTensor(feat_min_np)
    fmax = torch.FloatTensor(feat_max_np)

    x_target = torch.FloatTensor(target_x)
    target_grads = []
    for model in surrogates:
        g = get_param_grad(model, x_target, y_adv_t, create_graph=False)
        target_grads.append(g.detach())
        for param in model.parameters():
            param.requires_grad_(False)

    base_tensors = [torch.FloatTensor(X_test[base_indices[i]]).detach() for i in range(p)]
    base_labels  = [float(y_test[base_indices[i]]) for i in range(p)]
    deltas       = [torch.zeros_like(base_tensors[i], requires_grad=True) for i in range(p)]

    optimizer = torch.optim.SGD(deltas, lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=75, eps=1e-06
    )

    best_loss   = float('inf')
    best_deltas = [d.data.clone() for d in deltas]

    for it in range(max_iter):
        optimizer.zero_grad()
        total_loss = torch.zeros(1)[0]

        for j, model in enumerate(surrogates):
            model.eval()
            for param in model.parameters():
                param.requires_grad_(True)

            for i in range(p):
                poison_x = torch.clamp(base_tensors[i] + deltas[i], fmin, fmax)
                g_poison = get_param_grad(model, poison_x, base_labels[i], create_graph=True)
                cos_sim  = F.cosine_similarity(
                    target_grads[j].unsqueeze(0), g_poison.unsqueeze(0)
                )
                total_loss = total_loss + (1.0 - cos_sim.squeeze())

        total_loss = total_loss / 5.0
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())

        for model in surrogates:
            model.zero_grad()
            for param in model.parameters():
                param.requires_grad_(False)

        with torch.no_grad():
            for i in range(p):
                clamped      = torch.clamp(base_tensors[i] + deltas[i], fmin, fmax)
                deltas[i].data = clamped - base_tensors[i]

        loss_val = total_loss.item()
        if loss_val < best_loss:
            best_loss   = loss_val
            best_deltas = [d.data.clone() for d in deltas]

        if it % 200 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"      Iter {it:4d}: loss={loss_val:.6f}, lr={lr_now:.6f}")

    print(f"      Best loss: {best_loss:.6f}")
    poison_X = [torch.clamp(base_tensors[i] + best_deltas[i], fmin, fmax).numpy()
                for i in range(p)]
    return np.array(poison_X, dtype=np.float32), np.array(base_labels, dtype=np.float32)

# Precompute base candidates once per target (max p=15)
print("\nBase minták előszámítása minden target-re (max p=15)...")
all_base_idx = {}
for t_pos, t_idx in enumerate(target_indices):
    print(f"\n  Target {t_pos+1}/10 (test index={t_idx})")
    all_base_idx[t_idx] = select_base_samples(
        X_test_all[t_idx], y_adv, X_test_all, y_test_all, surrogates, p_count=15
    )

targeted_results = {}
for p_base in [5, 15]:
    print(f"\n{'='*70}")
    print(f"TARGETED POISONING: p = {p_base} base minta target-enként")
    print(f"{'='*70}")

    successes = 0
    for t_pos, t_idx in enumerate(target_indices):
        print(f"\n  --- Target {t_pos+1}/10 (test index={t_idx}) ---")
        base_idx = all_base_idx[t_idx][:p_base]

        print(f"    WiB PGD (1000 iter, p={p_base})...")
        poison_X, poison_y = witches_brew_attack(
            X_test_all[t_idx], y_adv, base_idx,
            X_test_all, y_test_all, surrogates,
            feat_min, feat_max, max_iter=1000
        )

        Xp = np.concatenate([X_train, poison_X])
        yp = np.concatenate([y_train, poison_y])
        retrained = train_ids(Xp, yp, seed=int(t_idx) + 9999)
        pred = predict(retrained, X_test_all[t_idx:t_idx+1])

        is_success = (pred[0] == y_adv)
        if is_success:
            successes += 1
        print(f"    Pred: {int(pred[0])}, Target: {int(y_adv)} "
              f"-> {'SIKERES' if is_success else 'SIKERTELEN'}")

    rate = successes / 10
    targeted_results[p_base] = {'successes': successes, 'rate': rate}
    print(f"\n  >> Támadás sikeressége p={p_base}: {successes}/10 = {rate:.0%}")

# ── ÖSSZEFOGLALÓ ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("ÖSSZEFOGLALÓ")
print("="*70)
print(f"\nClean modell validációs accuracy: {clean_val_acc:.4f}")
print("\n--- 1. feladat: Untargeted Poisoning ---")
print(f"{'Módszer':<25} {'p=30%':>10} {'p=50%':>10} {'p=70%':>10}")
print("-"*55)
print(f"{'Q1: Random':<25} "
      f"{np.mean(q1_results[30]):>10.4f} "
      f"{np.mean(q1_results[50]):>10.4f} "
      f"{np.mean(q1_results[70]):>10.4f}")
print(f"{'Q2: Loss-based':<25} "
      f"{np.mean(q2_results[30]):>10.4f} "
      f"{np.mean(q2_results[50]):>10.4f} "
      f"{np.mean(q2_results[70]):>10.4f}")
print("\n--- 2. feladat: Targeted Poisoning (WiB) ---")
for p_b, res in targeted_results.items():
    print(f"  p={p_b}: {res['successes']}/10 sikeresen félreklasszifikálva ({res['rate']:.0%})")
