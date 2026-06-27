# -*- coding: utf-8 -*-
"""
FAST version:
Complex scheme (Same D, Diff genus negatives 포함) +
form-disjoint split +
normalization ablation (raw / log1p / zscore) +
P in {400,500,600} +
seed 5회 반복 -> mean±std

속도 개선 핵심:
1) theta_series(P) 대신 NumPy로 직접 r_Q(n) 계산 (n=0..MAX_P-1)
2) ID/OOD 전체 theta를 MAX_P까지 한 번만 계산하고 디스크 캐시 저장/로드
3) pair->feature 생성 완전 벡터화
4) Transformer는 작은 모델 + early stopping(옵션)

실행:
    sage -python complex_fast_5seeds.py

출력:
    results_per_seed.csv
    results_summary_mean_std.csv
"""

from sage.all import *
from sage.quadratic_forms.binary_qf import BinaryQF_reduced_representatives

import gc
import math
import os
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 0) 설정
# ============================================================
SEEDS = [42, 43, 44, 45, 46]
P_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 300, 400, 500, 600]
MAX_P = max(P_LIST)

# |D| ranges
ID_MIN_ABS_DISC = 4
ID_MAX_ABS_DISC = 4000
OOD_MIN_ABS_DISC = 4001
OOD_MAX_ABS_DISC = 8000

# form-disjoint split ratio
ID_FORM_TEST_SIZE = 0.30

# Complex scheme pair counts
ID_TRAIN_SAME_D_SAME_G = 10000
ID_TRAIN_SAME_D_DIFF_G = 5000
ID_TRAIN_DIFF_D_DIFF_G = 5000

ID_TEST_SAME_D_SAME_G = 5000
ID_TEST_SAME_D_DIFF_G = 2500
ID_TEST_DIFF_D_DIFF_G = 2500

OOD_SAME_D_SAME_G = 5000
OOD_SAME_D_DIFF_G = 2500
OOD_DIFF_D_DIFF_G = 2500

NORMALIZATION_MODES = ["raw", "log1p", "zscore"]

# Model options
RUN_TRANSFORMER = True  # CPU에서 너무 오래 걸리면 False로 먼저 DT/RF/MLP만

# Small Transformer for speed (same idea as the FAST code)
TF_D_MODEL = 48
TF_NHEAD = 4
TF_NUM_LAYERS = 1
TF_FF_DIM = 96
TF_LR = 1e-3
TF_BATCH = 128
TF_MAX_EPOCHS_BY_P = {400: 15, 500: 18, 600: 20}
TF_PATIENCE = 3
TF_MIN_EPOCHS = 6

# sklearn MLP
MLP_HIDDEN = (512, 256)
MLP_MAX_ITER = 250

# True = fit zscore using all train forms; False = fit only forms appearing in train pairs
# Original Section 8 default is False.
ZSCORE_FIT_ON_ALL_TRAIN_FORMS = False

# Disk cache
CACHE_DIR = "fast_cache_theta"
USE_DISK_CACHE = True

# theta 계산 병렬화(선택)
# 0 또는 1이면 단일 프로세스. 4~(코어수-1) 추천.
THETA_N_JOBS = 0
THETA_CHUNK = 500

# ============================================================
# 1) 재현성
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2) Catalog 생성 (a,b,c + genus_id(int) + disc_map)
# ============================================================
def binaryqf_to_quadratic_form(bqf):
    a, b, c = list(bqf)
    return QuadraticForm(ZZ, 2, [a, b, c])


def genus_id_string(Q) -> str:
    """Conway–Sloane genus symbol based identifier."""
    try:
        symbols = Q.CS_genus_symbol_list()
        return " | ".join(str(s) for s in symbols)
    except Exception:
        from sage.quadratic_forms.genera.genus import Genus
        G = Genus(Q.Gram_matrix())
        return f"{G.signature_pair()} | {G.discriminant_form()}"


def build_catalog(min_abs_disc: int, max_abs_disc: int, primitive_only: bool = True):
    """
    Returns:
      a_arr, b_arr, c_arr: (n_forms,)
      absD_arr: (n_forms,)
      genus_int_arr: (n_forms,)
      genus_str_list: genus_int -> genus_str
      disc_map: absD -> genus_int -> [form_idx, ...]
    """
    a_list, b_list, c_list = [], [], []
    absD_list = []
    genus_int_list = []
    genus_to_int: Dict[str, int] = {}
    genus_str_list: List[str] = []
    disc_map = defaultdict(lambda: defaultdict(list))

    for D_abs in range(min_abs_disc, max_abs_disc + 1):
        D = -D_abs
        try:
            reps = BinaryQF_reduced_representatives(D, primitive_only=primitive_only)
        except Exception:
            continue

        for bqf in reps:
            if not bqf.is_positive_definite():
                continue
            if primitive_only and (not bqf.is_primitive()):
                continue
            if not bqf.is_reduced():
                continue

            a, b, c = list(bqf)
            Q = binaryqf_to_quadratic_form(bqf)
            gstr = genus_id_string(Q)

            if gstr not in genus_to_int:
                gid = len(genus_str_list)
                genus_to_int[gstr] = gid
                genus_str_list.append(gstr)
            else:
                gid = genus_to_int[gstr]

            idx = len(a_list)
            a_list.append(int(a))
            b_list.append(int(b))
            c_list.append(int(c))
            absD_list.append(int(D_abs))
            genus_int_list.append(int(gid))
            disc_map[int(D_abs)][int(gid)].append(idx)

    a_arr = np.array(a_list, dtype=np.int32)
    b_arr = np.array(b_list, dtype=np.int32)
    c_arr = np.array(c_list, dtype=np.int32)
    absD_arr = np.array(absD_list, dtype=np.int32)
    genus_int_arr = np.array(genus_int_list, dtype=np.int32)
    return a_arr, b_arr, c_arr, absD_arr, genus_int_arr, genus_str_list, disc_map


def save_catalog_npz(
    path: str,
    a_arr,
    b_arr,
    c_arr,
    absD_arr,
    genus_int_arr,
    genus_str_list: List[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        a=a_arr,
        b=b_arr,
        c=c_arr,
        absD=absD_arr,
        genus_int=genus_int_arr,
        genus_str=np.array(genus_str_list, dtype=object),
    )


def load_catalog_npz(path: str):
    data = np.load(path, allow_pickle=True)
    a_arr = data["a"].astype(np.int32)
    b_arr = data["b"].astype(np.int32)
    c_arr = data["c"].astype(np.int32)
    absD_arr = data["absD"].astype(np.int32)
    genus_int_arr = data["genus_int"].astype(np.int32)
    genus_str_list = list(data["genus_str"])

    disc_map = defaultdict(lambda: defaultdict(list))
    for idx in range(len(a_arr)):
        d = int(absD_arr[idx])
        g = int(genus_int_arr[idx])
        disc_map[d][g].append(idx)

    return a_arr, b_arr, c_arr, absD_arr, genus_int_arr, genus_str_list, disc_map


# ============================================================
# 3) Form-disjoint split
# ============================================================
def restrict_disc_map(disc_map_full, allowed_mask: np.ndarray):
    out = defaultdict(lambda: defaultdict(list))
    for d, gdict in disc_map_full.items():
        for g, idxs in gdict.items():
            sub = [i for i in idxs if allowed_mask[i]]
            if sub:
                out[d][g] = sub
    return out


def count_groups_ge2(disc_map_subset) -> int:
    cnt = 0
    for _, gdict in disc_map_subset.items():
        for _, idxs in gdict.items():
            if len(idxs) >= 2:
                cnt += 1
    return cnt


def form_disjoint_split_id(
    n_forms: int,
    test_size: float,
    seed: int,
    disc_map_id,
    min_ge2_groups: int = 1500,
    max_tries: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns train_mask, test_mask (bool arrays)."""
    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_forms, dtype=np.int32)
    best_train_mask = None
    best_test_mask = None

    for t in range(max_tries):
        rng.shuffle(all_idx)
        n_test = int(round(n_forms * test_size))
        test_idx = all_idx[:n_test]
        train_idx = all_idx[n_test:]

        train_mask = np.zeros(n_forms, dtype=bool)
        test_mask = np.zeros(n_forms, dtype=bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        disc_map_train = restrict_disc_map(disc_map_id, train_mask)
        disc_map_test = restrict_disc_map(disc_map_id, test_mask)
        gt = count_groups_ge2(disc_map_train)
        gv = count_groups_ge2(disc_map_test)

        if gt >= min_ge2_groups and gv >= min_ge2_groups:
            print(f"[form split] success try={t+1}: ge2 groups train={gt}, test={gv}")
            return train_mask, test_mask

        best_train_mask, best_test_mask = train_mask, test_mask

    print("[form split] WARNING: using last split (may reduce available pairs).")
    return best_train_mask, best_test_mask


# ============================================================
# 4) Complex pair generation
# ============================================================
def make_complex_pairs(
    disc_map_subset,
    count_sd_sg: int,
    count_sd_dg: int,
    count_dd_dg: int,
    rng: np.random.Generator,
    max_attempts_factor: int = 800,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      same_pairs: (count_sd_sg, 2), label 0
      diff_pairs: (count_sd_dg + count_dd_dg, 2), label 1
    """
    discs = list(disc_map_subset.keys())
    if not discs:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)

    eligible_A = []  # discs with a genus containing >= 2 forms
    A_genera = {}    # disc -> eligible genera list
    eligible_B = []  # discs with >= 2 genera
    disc_to_pool = {}

    for d in discs:
        gdict = disc_map_subset[d]
        gA = [g for g, idxs in gdict.items() if len(idxs) >= 2]
        if gA:
            eligible_A.append(d)
            A_genera[d] = gA
        if len(gdict.keys()) >= 2:
            eligible_B.append(d)

        pool = []
        for idxs in gdict.values():
            pool.extend(idxs)
        disc_to_pool[d] = pool

    # --- Same D, Same G ---
    same_pairs = []
    attempts = 0
    max_attempts = max_attempts_factor * max(1, count_sd_sg)
    while len(same_pairs) < count_sd_sg and attempts < max_attempts:
        attempts += 1
        if not eligible_A:
            break
        d = int(rng.choice(eligible_A))
        g = int(rng.choice(A_genera[d]))
        idxs = disc_map_subset[d][g]
        if len(idxs) < 2:
            continue
        i, j = rng.choice(idxs, size=2, replace=False)
        i, j = int(i), int(j)
        if i > j:
            i, j = j, i
        same_pairs.append((i, j))
    if len(same_pairs) < count_sd_sg:
        print(f"[pairs] WARNING same(SD,SG): {len(same_pairs)}/{count_sd_sg}")

    # --- Same D, Diff G ---
    diff_pairs = []
    attempts = 0
    max_attempts = max_attempts_factor * max(1, count_sd_dg)
    while len(diff_pairs) < count_sd_dg and attempts < max_attempts:
        attempts += 1
        if not eligible_B:
            break
        d = int(rng.choice(eligible_B))
        gids = list(disc_map_subset[d].keys())
        if len(gids) < 2:
            continue
        g1, g2 = rng.choice(gids, size=2, replace=False)
        i = int(rng.choice(disc_map_subset[d][int(g1)]))
        j = int(rng.choice(disc_map_subset[d][int(g2)]))
        if i > j:
            i, j = j, i
        diff_pairs.append((i, j))
    if len(diff_pairs) < count_sd_dg:
        print(f"[pairs] WARNING diff(SD,DG): {len(diff_pairs)}/{count_sd_dg}")

    # --- Diff D, Diff G ---
    target_total = count_sd_dg + count_dd_dg
    attempts = 0
    max_attempts = max_attempts_factor * max(1, count_dd_dg)
    while len(diff_pairs) < target_total and attempts < max_attempts:
        attempts += 1
        if len(discs) < 2:
            break
        d1, d2 = rng.choice(discs, size=2, replace=False)
        i = int(rng.choice(disc_to_pool[int(d1)]))
        j = int(rng.choice(disc_to_pool[int(d2)]))
        if i > j:
            i, j = j, i
        diff_pairs.append((i, j))
    if len(diff_pairs) < target_total:
        made = max(0, len(diff_pairs) - count_sd_dg)
        print(f"[pairs] WARNING diff(DD,DG): {made}/{count_dd_dg}")

    return np.array(same_pairs, dtype=np.int32), np.array(diff_pairs, dtype=np.int32)


# ============================================================
# 5) FAST theta coefficient computation (NumPy)
# ============================================================
def compute_global_B(a_arr, b_arr, c_arr, P: int) -> int:
    """
    Safe global bound B so that all (x,y) with Q(x,y) < P are included.
    """
    a = a_arr.astype(np.float64)
    b = b_arr.astype(np.float64)
    c = c_arr.astype(np.float64)
    lam_min = (a + c - np.sqrt((a - c) ** 2 + b ** 2)) / 2.0
    lam_min = np.maximum(lam_min, 1e-6)
    lam = float(lam_min.min())
    B = int(math.ceil(math.sqrt((P - 1) / lam)))
    return max(B, 1)


def theta_precompute_matrices(B: int):
    xs = np.arange(-B, B + 1, dtype=np.int32)
    X = xs[:, None]
    Y = xs[None, :]
    X2 = (X * X).astype(np.int32)
    Y2 = (Y * Y).astype(np.int32)
    XY = (X * Y).astype(np.int32)
    return X2, XY, Y2


def theta_counts_for_form(
    a: int,
    b: int,
    c: int,
    X2: np.ndarray,
    XY: np.ndarray,
    Y2: np.ndarray,
    P: int,
) -> np.ndarray:
    Q = a * X2 + b * XY + c * Y2
    vals = Q[Q < P].ravel()
    cnt = np.bincount(vals, minlength=P).astype(np.float32)
    return cnt


def precompute_theta_matrix(
    a_arr,
    b_arr,
    c_arr,
    P: int,
    X2,
    XY,
    Y2,
    progress_every: int = 2000,
) -> np.ndarray:
    n = len(a_arr)
    out = np.zeros((n, P), dtype=np.float32)
    for i in range(n):
        out[i] = theta_counts_for_form(int(a_arr[i]), int(b_arr[i]), int(c_arr[i]), X2, XY, Y2, P)
        if (i + 1) % progress_every == 0 or (i + 1) == n:
            print(f"  [theta] {i+1}/{n} computed...")
    return out


def load_or_build_theta(name: str, a_arr, b_arr, c_arr, X2, XY, Y2, P: int) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}_theta_MAXP{P}.npy")
    if USE_DISK_CACHE and os.path.exists(path):
        print(f"[theta] loading cached {path}")
        mat = np.load(path, mmap_mode=None)
        if mat.shape == (len(a_arr), P):
            return mat.astype(np.float32, copy=False)
        print("[theta] cache shape mismatch, recomputing...")

    print(f"[theta] computing {name} theta matrix (n={len(a_arr)}, P={P}) ...")
    mat = precompute_theta_matrix(a_arr, b_arr, c_arr, P, X2, XY, Y2)
    if USE_DISK_CACHE:
        np.save(path, mat)
        print(f"[theta] saved cache {path}")
    return mat


# ============================================================
# 6) Normalization + vectorized dataset construction
# ============================================================
def fit_zscore(theta_id: np.ndarray, fit_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit z-score statistics using only fit_indices."""
    X = theta_id[fit_indices, :]
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_norm_matrix(
    X: np.ndarray,
    norm: str,
    mean_full: Optional[np.ndarray],
    std_full: Optional[np.ndarray],
) -> np.ndarray:
    if norm == "raw":
        return X
    if norm == "log1p":
        return np.log1p(X)
    if norm == "zscore":
        assert mean_full is not None and std_full is not None
        return (X - mean_full[: X.shape[1]]) / std_full[: X.shape[1]]
    raise ValueError(norm)


def build_classical_Xy(
    theta_mat: np.ndarray,
    same_pairs: np.ndarray,
    diff_pairs: np.ndarray,
    P: int,
    norm: str,
    z_mean: Optional[np.ndarray],
    z_std: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    pairs = np.vstack([same_pairs, diff_pairs]).astype(np.int32)
    y = np.concatenate(
        [
            np.zeros(len(same_pairs), dtype=np.int64),
            np.ones(len(diff_pairs), dtype=np.int64),
        ]
    )
    i = pairs[:, 0]
    j = pairs[:, 1]
    c1 = theta_mat[i, :P].astype(np.float32, copy=False)
    c2 = theta_mat[j, :P].astype(np.float32, copy=False)
    c1 = apply_norm_matrix(c1, norm, z_mean, z_std)
    c2 = apply_norm_matrix(c2, norm, z_mean, z_std)
    X = np.concatenate([c1, c2, np.abs(c1 - c2)], axis=1).astype(np.float32, copy=False)
    return X, y


def build_seq_Xy(
    theta_mat: np.ndarray,
    same_pairs: np.ndarray,
    diff_pairs: np.ndarray,
    P: int,
    norm: str,
    z_mean: Optional[np.ndarray],
    z_std: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    pairs = np.vstack([same_pairs, diff_pairs]).astype(np.int32)
    y = np.concatenate(
        [
            np.zeros(len(same_pairs), dtype=np.int64),
            np.ones(len(diff_pairs), dtype=np.int64),
        ]
    )
    i = pairs[:, 0]
    j = pairs[:, 1]
    c1 = theta_mat[i, :P].astype(np.float32, copy=False)
    c2 = theta_mat[j, :P].astype(np.float32, copy=False)
    c1 = apply_norm_matrix(c1, norm, z_mean, z_std)
    c2 = apply_norm_matrix(c2, norm, z_mean, z_std)
    X = np.stack([c1, c2], axis=2).astype(np.float32, copy=False)  # (N, P, 2)
    return X, y


# ============================================================
# 7) Models + evaluation
# ============================================================
@dataclass
class PerSeedResult:
    seed: int
    P: int
    norm: str
    model: str
    id_acc: float
    ood_acc: float
    ood_auroc: float


def eval_sklearn(clf, X_id, y_id, X_ood, y_ood) -> Tuple[float, float, float]:
    id_pred = clf.predict(X_id)
    ood_pred = clf.predict(X_ood)
    id_acc = accuracy_score(y_id, id_pred)
    ood_acc = accuracy_score(y_ood, ood_pred)

    id_conf = clf.predict_proba(X_id).max(axis=1)
    ood_conf = clf.predict_proba(X_ood).max(axis=1)
    labels = np.concatenate([np.ones_like(id_conf), np.zeros_like(ood_conf)])
    scores = np.concatenate([id_conf, ood_conf])
    auroc = roc_auc_score(labels, scores)
    return id_acc, ood_acc, auroc


class ThetaPairTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int = TF_D_MODEL,
        nhead: int = TF_NHEAD,
        num_layers: int = TF_NUM_LAYERS,
        ff_dim: int = TF_FF_DIM,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_linear = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.input_linear(x) + self.pos_embedding
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


@torch.no_grad()
def tf_conf_pred(model: nn.Module, X: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    dl = DataLoader(TensorDataset(torch.tensor(X)), batch_size=TF_BATCH, shuffle=False)
    model.eval()
    confs, preds = [], []
    for (xb,) in dl:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        maxp, pred = torch.max(probs, dim=1)
        confs.append(maxp.cpu().numpy())
        preds.append(pred.cpu().numpy())
    return np.concatenate(confs), np.concatenate(preds)


def eval_transformer(model: nn.Module, X_id, y_id, X_ood, y_ood, device: torch.device) -> Tuple[float, float, float]:
    id_conf, id_pred = tf_conf_pred(model, X_id, device)
    ood_conf, ood_pred = tf_conf_pred(model, X_ood, device)
    id_acc = accuracy_score(y_id, id_pred)
    ood_acc = accuracy_score(y_ood, ood_pred)
    labels = np.concatenate([np.ones_like(id_conf), np.zeros_like(ood_conf)])
    scores = np.concatenate([id_conf, ood_conf])
    auroc = roc_auc_score(labels, scores)
    return id_acc, ood_acc, auroc


def train_transformer_earlystop(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    device: torch.device,
    max_epochs: int,
) -> nn.Module:
    rng = np.random.default_rng(12345)
    n = len(y_tr)
    perm = rng.permutation(n)
    n_val = max(1000, int(0.1 * n))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    X_val = X_tr[val_idx]
    y_val = y_tr[val_idx]
    X_train = X_tr[tr_idx]
    y_train = y_tr[tr_idx]

    model = ThetaPairTransformer(seq_len=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TF_LR)
    crit = nn.CrossEntropyLoss()

    dl = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=TF_BATCH,
        shuffle=True,
    )

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            vdl = DataLoader(
                TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
                batch_size=TF_BATCH,
                shuffle=False,
            )
            vloss = 0.0
            for xb, yb in vdl:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                vloss += float(loss.item())
            vloss /= max(1, len(vdl))

        if ep % 3 == 0 or ep == 1 or ep == max_epochs:
            print(
                f"    ep {ep:02d}/{max_epochs} | "
                f"train_loss={total_loss / len(dl):.4f} | val_loss={vloss:.4f}"
            )

        improved = vloss < best_val - 1e-4
        if improved:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep >= TF_MIN_EPOCHS and bad >= TF_PATIENCE:
            print(f"    early stop at ep={ep} (best_val={best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# 8) Main
# ============================================================
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 110)
    print(f"[System] device={device} | FAST Complex | seeds={SEEDS} | P_LIST={P_LIST} | MAX_P={MAX_P}")
    print(f"[Ablation] norms={NORMALIZATION_MODES} | RUN_TRANSFORMER={RUN_TRANSFORMER}")
    print(f"[Cache] dir={CACHE_DIR} | USE_DISK_CACHE={USE_DISK_CACHE}")
    print("=" * 110)

    # ------------------------------------------------------------
    # (A) Catalog load/build
    # ------------------------------------------------------------
    id_cat_path = os.path.join(CACHE_DIR, f"id_catalog_{ID_MIN_ABS_DISC}_{ID_MAX_ABS_DISC}.npz")
    ood_cat_path = os.path.join(CACHE_DIR, f"ood_catalog_{OOD_MIN_ABS_DISC}_{OOD_MAX_ABS_DISC}.npz")

    if USE_DISK_CACHE and os.path.exists(id_cat_path):
        print("[Catalog] loading cached ID catalog...")
        a_id, b_id, c_id, absD_id, genus_id_int, genus_str_id, disc_map_id = load_catalog_npz(id_cat_path)
    else:
        print("[Catalog] building ID catalog...")
        a_id, b_id, c_id, absD_id, genus_id_int, genus_str_id, disc_map_id = build_catalog(
            ID_MIN_ABS_DISC, ID_MAX_ABS_DISC, True
        )
        if USE_DISK_CACHE:
            save_catalog_npz(id_cat_path, a_id, b_id, c_id, absD_id, genus_id_int, genus_str_id)
            print(f"[Catalog] saved {id_cat_path}")
    print(f"  ID forms = {len(a_id)}")

    if USE_DISK_CACHE and os.path.exists(ood_cat_path):
        print("[Catalog] loading cached OOD catalog...")
        a_ood, b_ood, c_ood, absD_ood, genus_ood_int, genus_str_ood, disc_map_ood = load_catalog_npz(ood_cat_path)
    else:
        print("[Catalog] building OOD catalog...")
        a_ood, b_ood, c_ood, absD_ood, genus_ood_int, genus_str_ood, disc_map_ood = build_catalog(
            OOD_MIN_ABS_DISC, OOD_MAX_ABS_DISC, True
        )
        if USE_DISK_CACHE:
            save_catalog_npz(ood_cat_path, a_ood, b_ood, c_ood, absD_ood, genus_ood_int, genus_str_ood)
            print(f"[Catalog] saved {ood_cat_path}")
    print(f"  OOD forms = {len(a_ood)}")

    # ------------------------------------------------------------
    # (B) Theta load/build
    # ------------------------------------------------------------
    print("\n[Theta] computing safe global bound B ...")
    B_id = compute_global_B(a_id, b_id, c_id, MAX_P)
    B_ood = compute_global_B(a_ood, b_ood, c_ood, MAX_P)
    B = max(B_id, B_ood)
    print(f"  B_id={B_id}, B_ood={B_ood} -> using B={B} (grid size={(2*B+1)}x{(2*B+1)})")
    X2, XY, Y2 = theta_precompute_matrices(B)

    theta_id = load_or_build_theta("id", a_id, b_id, c_id, X2, XY, Y2, MAX_P)
    theta_ood = load_or_build_theta("ood", a_ood, b_ood, c_ood, X2, XY, Y2, MAX_P)

    # ------------------------------------------------------------
    # (C) Seed loop
    # ------------------------------------------------------------
    all_results: List[PerSeedResult] = []

    for seed in SEEDS:
        print("\n" + "#" * 110)
        print(f"[SEED RUN] seed={seed}")
        print("#" * 110)
        set_seed(seed)
        rng = np.random.default_rng(seed)

        # 1) form-disjoint split
        print("\n[1/5] form-disjoint split (ID)...")
        train_mask, test_mask = form_disjoint_split_id(
            n_forms=len(a_id),
            test_size=ID_FORM_TEST_SIZE,
            seed=seed,
            disc_map_id=disc_map_id,
            min_ge2_groups=1500,
            max_tries=50,
        )
        disc_map_train = restrict_disc_map(disc_map_id, train_mask)
        disc_map_test = restrict_disc_map(disc_map_id, test_mask)

        # 2) complex pairs
        print("\n[2/5] sampling Complex pairs...")
        tr_same, tr_diff = make_complex_pairs(
            disc_map_train,
            ID_TRAIN_SAME_D_SAME_G,
            ID_TRAIN_SAME_D_DIFF_G,
            ID_TRAIN_DIFF_D_DIFF_G,
            rng,
        )
        id_same, id_diff = make_complex_pairs(
            disc_map_test,
            ID_TEST_SAME_D_SAME_G,
            ID_TEST_SAME_D_DIFF_G,
            ID_TEST_DIFF_D_DIFF_G,
            rng,
        )
        ood_same, ood_diff = make_complex_pairs(
            disc_map_ood,
            OOD_SAME_D_SAME_G,
            OOD_SAME_D_DIFF_G,
            OOD_DIFF_D_DIFF_G,
            rng,
        )
        print(f"  train pairs: same={len(tr_same)}, diff={len(tr_diff)}")
        print(f"  ID-test pairs: same={len(id_same)}, diff={len(id_diff)}")
        print(f"  OOD pairs: same={len(ood_same)}, diff={len(ood_diff)}")

        # 3) z-score fit
        print("\n[3/5] fitting z-score from TRAIN only...")
        if ZSCORE_FIT_ON_ALL_TRAIN_FORMS:
            fit_idx = np.where(train_mask)[0].astype(np.int32)
        else:
            fit_idx = np.unique(np.concatenate([tr_same.ravel(), tr_diff.ravel()])).astype(np.int32)
        z_mean, z_std = fit_zscore(theta_id, fit_idx)

        # 4) P x norm x models
        print("\n[4/5] train/eval (P x norm x models) ...")
        for P in P_LIST:
            print("\n" + "=" * 100)
            print(f"[seed={seed}] P={P}")
            print("=" * 100)

            for norm in NORMALIZATION_MODES:
                print("-" * 100)
                print(f"[norm={norm}] building datasets (vectorized) ...")

                X_tr, y_tr = build_classical_Xy(theta_id, tr_same, tr_diff, P, norm, z_mean, z_std)
                X_id, y_id = build_classical_Xy(theta_id, id_same, id_diff, P, norm, z_mean, z_std)
                X_ood, y_ood = build_classical_Xy(theta_ood, ood_same, ood_diff, P, norm, z_mean, z_std)

                # Decision Tree
                dt = DecisionTreeClassifier(max_depth=25, random_state=seed)
                dt.fit(X_tr, y_tr)
                id_acc, ood_acc, auroc = eval_sklearn(dt, X_id, y_id, X_ood, y_ood)
                all_results.append(PerSeedResult(seed, P, norm, "DecisionTree", id_acc, ood_acc, auroc))

                # Random Forest
                rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
                rf.fit(X_tr, y_tr)
                id_acc, ood_acc, auroc = eval_sklearn(rf, X_id, y_id, X_ood, y_ood)
                all_results.append(PerSeedResult(seed, P, norm, "RandomForest", id_acc, ood_acc, auroc))

                # MLP
                mlp = MLPClassifier(
                    hidden_layer_sizes=MLP_HIDDEN,
                    max_iter=MLP_MAX_ITER,
                    random_state=seed,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                )
                mlp.fit(X_tr, y_tr)
                id_acc, ood_acc, auroc = eval_sklearn(mlp, X_id, y_id, X_ood, y_ood)
                all_results.append(PerSeedResult(seed, P, norm, "MLP", id_acc, ood_acc, auroc))

                # Transformer
                if RUN_TRANSFORMER:
                    print(f"[norm={norm}] Transformer: building seq tensors ...")
                    X_tr_seq, y_tr_seq = build_seq_Xy(theta_id, tr_same, tr_diff, P, norm, z_mean, z_std)
                    X_id_seq, y_id_seq = build_seq_Xy(theta_id, id_same, id_diff, P, norm, z_mean, z_std)
                    X_ood_seq, y_ood_seq = build_seq_Xy(theta_ood, ood_same, ood_diff, P, norm, z_mean, z_std)

                    set_seed(seed + 1000 + P)
                    max_epochs = TF_MAX_EPOCHS_BY_P.get(P, 15)
                    print(f"   Training Transformer (max_epochs={max_epochs}, early_stop) ...")
                    tf_model = train_transformer_earlystop(X_tr_seq, y_tr_seq, device, max_epochs)
                    id_acc, ood_acc, auroc = eval_transformer(
                        tf_model, X_id_seq, y_id_seq, X_ood_seq, y_ood_seq, device
                    )
                    all_results.append(PerSeedResult(seed, P, norm, "Transformer", id_acc, ood_acc, auroc))

                    del X_tr_seq, y_tr_seq, X_id_seq, y_id_seq, X_ood_seq, y_ood_seq, tf_model
                    gc.collect()

                del X_tr, y_tr, X_id, y_id, X_ood, y_ood, dt, rf, mlp
                gc.collect()

        print("\n[5/5] seed done.")

    # ------------------------------------------------------------
    # (D) Save per-seed results
    # ------------------------------------------------------------
    print("\n" + "=" * 110)
    print("[SAVE] results_per_seed.csv")
    print("=" * 110)
    with open("results_per_seed.csv", "w", encoding="utf-8") as f:
        f.write("seed,P,norm,model,id_acc,ood_acc,ood_auroc\n")
        for r in all_results:
            f.write(
                f"{r.seed},{r.P},{r.norm},{r.model},{r.id_acc:.6f},{r.ood_acc:.6f},{r.ood_auroc:.6f}\n"
            )

    # ------------------------------------------------------------
    # (E) Mean ± std summary
    # ------------------------------------------------------------
    bucket = defaultdict(list)
    for r in all_results:
        bucket[(r.P, r.norm, r.model)].append((r.id_acc, r.ood_acc, r.ood_auroc))

    summary = []
    for key, vals in bucket.items():
        arr = np.array(vals, dtype=np.float64)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(3)
        P, norm, model = key
        summary.append(
            (
                P,
                norm,
                model,
                mean[0],
                std[0],
                mean[1],
                std[1],
                mean[2],
                std[2],
                arr.shape[0],
            )
        )

    summary.sort(key=lambda x: (x[0], x[1], x[2]))

    print("\n" + "=" * 110)
    print("FINAL SUMMARY (mean ± std over seeds)")
    print("=" * 110)
    print(
        f"{'P':>3} | {'Norm':<7} | {'Model':<12} | "
        f"{'ID Acc':>16} | {'OOD Acc':>16} | {'OOD AUROC':>18} | n"
    )
    print("-" * 110)
    for row in summary:
        P, norm, model, id_m, id_s, ood_m, ood_s, au_m, au_s, n = row
        print(
            f"{P:>3} | {norm:<7} | {model:<12} | "
            f"{id_m:7.4f} ± {id_s:6.4f} | "
            f"{ood_m:7.4f} ± {ood_s:6.4f} | "
            f"{au_m:8.4f} ± {au_s:7.4f} | {n}"
        )
    print("-" * 110)

    print("[SAVE] results_summary_mean_std.csv")
    with open("results_summary_mean_std.csv", "w", encoding="utf-8") as f:
        f.write(
            "P,norm,model,id_acc_mean,id_acc_std,ood_acc_mean,ood_acc_std,"
            "ood_auroc_mean,ood_auroc_std,n\n"
        )
        for row in summary:
            P, norm, model, id_m, id_s, ood_m, ood_s, au_m, au_s, n = row
            f.write(
                f"{P},{norm},{model},{id_m:.6f},{id_s:.6f},{ood_m:.6f},{ood_s:.6f},"
                f"{au_m:.6f},{au_s:.6f},{n}\n"
            )

    print("\nDONE.")


if __name__ == "__main__":
    main()
