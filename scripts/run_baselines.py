# -*- coding: utf-8 -*-
"""
Baseline experiments for the genus-comparison benchmark.

This script runs the baseline models reported in the manuscript:
  - L1 threshold
  - L2 threshold
  - cosine threshold
  - logistic regression on low-dimensional canonical pair features
  - optional linear SVM, disabled by default

The benchmark task is pairwise genus comparison from truncated theta-series
coefficient vectors. Pair order is controlled by deterministic canonical
ordering during pair sampling. Threshold baselines use symmetric scalar scores,
while logistic regression uses low-dimensional features computed from the
canonical ordered pair.

Default experimental settings:
  - ID discriminant window: |D| in [4, 4000]
  - OOD discriminant window: |D| in [4001, 8000]
  - truncation lengths:
      P in {10, 20, ..., 150, 200, 300, 400, 500, 600}
  - preprocessing modes: raw, log1p, z-score
  - random seeds: 42, 43, 44, 45, 46
  - z-score statistics fitted on forms appearing in sampled ID training pairs
    for the corresponding seed

Outputs by default:
  - results/baseline/results_baselines_per_seed.csv
  - results/baseline/results_baselines_summary_mean_std.csv
  - results/baseline/run_config_baseline.json

These CSV files are used to generate:
  - Table 3
  - Table 5
  - Figure 1
  - Figure 3
  - Figure 4

Example:
  sage -python scripts/run_baselines.py
"""

# ---------- Sage ----------
from sage.all import *
from sage.quadratic_forms.binary_qf import BinaryQF_reduced_representatives

# ---------- Standard libs ----------
import argparse
import json
import os, math, random, gc
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ---------- sklearn ----------
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# ============================================================
# 0) Default experimental configuration
# ============================================================
SEEDS = [42, 43, 44, 45, 46]
P_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 300, 400, 500, 600]
MAX_P = max(P_LIST)

# |D| ranges
ID_MIN_ABS_DISC = 4
ID_MAX_ABS_DISC = 4000
OOD_MIN_ABS_DISC = 4001
OOD_MAX_ABS_DISC = 8000

# Form-disjoint split ratio
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

# If False, fit z-score statistics only on forms appearing in sampled ID training pairs.
ZSCORE_FIT_ON_ALL_TRAIN_FORMS = False

# Disk cache
CACHE_DIR = "fast_cache_theta"
USE_DISK_CACHE = True

# Theta computation is single-process; disk caching is recommended.
THETA_PROGRESS_EVERY = 2000

# Baseline options
RUN_LINEAR_SVM = False  # Set True to include the optional LinearSVM baseline.

# Default output filenames. Command-line execution places these under --outdir.
OUT_PER_SEED = "results_baselines_per_seed.csv"
OUT_SUMMARY  = "results_baselines_summary_mean_std.csv"


# ============================================================
# 1) Reproducibility
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# 2) Catalog construction (a,b,c + genus_id(int) + disc_map)
# ============================================================
def genus_id_string(Q) -> str:
    try:
        symbols = Q.CS_genus_symbol_list()
        return " | ".join(str(s) for s in symbols)
    except Exception:
        from sage.quadratic_forms.genera.genus import Genus
        G = Genus(Q.Gram_matrix())
        return f"{G.signature_pair()} | {G.discriminant_form()}"

def binaryqf_to_quadratic_form(bqf):
    a, b, c = list(bqf)
    return QuadraticForm(ZZ, 2, [a, b, c])

def build_catalog(min_abs_disc: int, max_abs_disc: int, primitive_only: bool = True):
    """
    Returns:
      a_arr,b_arr,c_arr : (n_forms,)
      absD_arr          : (n_forms,)
      genus_int_arr     : (n_forms,)
      genus_str_list    : genus_int -> genus_str
      disc_map          : absD -> genus_int -> [form_idx,...]
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

def save_catalog_npz(path: str,
                     a_arr, b_arr, c_arr, absD_arr, genus_int_arr,
                     genus_str_list: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        a=a_arr, b=b_arr, c=c_arr,
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
# 3) Form-disjoint split and restricted disc_map
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
    for d, gdict in disc_map_subset.items():
        for g, idxs in gdict.items():
            if len(idxs) >= 2:
                cnt += 1
    return cnt

def form_disjoint_split_id(n_forms: int,
                           test_size: float,
                           seed: int,
                           disc_map_id,
                           min_ge2_groups: int = 1500,
                           max_tries: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: train_mask, test_mask (bool arrays)
    """
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
        disc_map_test  = restrict_disc_map(disc_map_id, test_mask)
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
def make_complex_pairs(disc_map_subset,
                       count_sd_sg: int, count_sd_dg: int, count_dd_dg: int,
                       rng: np.random.Generator,
                       max_attempts_factor: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      same_pairs: (count_sd_sg, 2) label 0
      diff_pairs: (count_sd_dg+count_dd_dg, 2) label 1
    """
    discs = list(disc_map_subset.keys())
    if not discs:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)

    eligible_A = []
    A_genera = {}
    eligible_B = []
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
        print(f"[pairs] WARNING diff(DD,DG): got {len(diff_pairs)-count_sd_dg}/{count_dd_dg}")

    return np.array(same_pairs, dtype=np.int32), np.array(diff_pairs, dtype=np.int32)


# ============================================================
# 5) FAST theta coefficient computation (NumPy) + cache
# ============================================================
def compute_global_B(a_arr, b_arr, c_arr, P: int) -> int:
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

def theta_counts_for_form(a: int, b: int, c: int, X2, XY, Y2, P: int) -> np.ndarray:
    Q = a * X2 + b * XY + c * Y2
    vals = Q[Q < P].ravel()
    cnt = np.bincount(vals, minlength=P).astype(np.float32)
    return cnt

def precompute_theta_matrix(a_arr, b_arr, c_arr, P: int, X2, XY, Y2, progress_every: int = 2000) -> np.ndarray:
    n = len(a_arr)
    out = np.zeros((n, P), dtype=np.float32)
    for i in range(n):
        out[i] = theta_counts_for_form(int(a_arr[i]), int(b_arr[i]), int(c_arr[i]), X2, XY, Y2, P)
        if (i + 1) % progress_every == 0 or (i + 1) == n:
            print(f" [theta] {i+1}/{n} computed...")
    return out

def load_or_build_theta(name: str, a_arr, b_arr, c_arr, X2, XY, Y2, P: int) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}_theta_MAXP{P}.npy")
    if USE_DISK_CACHE and os.path.exists(path):
        print(f"[theta] loading cached {path}")
        mat = np.load(path, mmap_mode=None)
        if mat.shape == (len(a_arr), P):
            return mat.astype(np.float32, copy=False)
        print("[theta] cache shape mismatch -> recomputing...")
    print(f"[theta] computing {name} theta matrix (n={len(a_arr)}, P={P}) ...")
    mat = precompute_theta_matrix(a_arr, b_arr, c_arr, P, X2, XY, Y2, progress_every=THETA_PROGRESS_EVERY)
    if USE_DISK_CACHE:
        np.save(path, mat)
        print(f"[theta] saved cache {path}")
    return mat


# ============================================================
# 6) Normalization + vectorized pair-array construction
# ============================================================
def fit_zscore(theta_id: np.ndarray, fit_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = theta_id[fit_indices, :]  # (n_fit, MAX_P)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std

def apply_norm_matrix(X: np.ndarray, norm: str, mean_full: Optional[np.ndarray], std_full: Optional[np.ndarray]) -> np.ndarray:
    if norm == "raw":
        return X
    if norm == "log1p":
        return np.log1p(X)
    if norm == "zscore":
        assert mean_full is not None and std_full is not None
        return (X - mean_full[:X.shape[1]]) / std_full[:X.shape[1]]
    raise ValueError(norm)

def build_pair_arrays(theta_mat: np.ndarray,
                      same_pairs: np.ndarray,
                      diff_pairs: np.ndarray,
                      P: int,
                      norm: str,
                      z_mean: Optional[np.ndarray],
                      z_std: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      c1: (N,P), c2: (N,P), y: (N,)  (y=0 same, y=1 diff)
    """
    pairs = np.vstack([same_pairs, diff_pairs]).astype(np.int32)
    y = np.concatenate([
        np.zeros(len(same_pairs), dtype=np.int64),
        np.ones(len(diff_pairs), dtype=np.int64)
    ])
    i = pairs[:, 0]
    j = pairs[:, 1]
    c1 = theta_mat[i, :P].astype(np.float32, copy=False)
    c2 = theta_mat[j, :P].astype(np.float32, copy=False)
    c1 = apply_norm_matrix(c1, norm, z_mean, z_std)
    c2 = apply_norm_matrix(c2, norm, z_mean, z_std)
    return c1, c2, y


# ============================================================
# 7) Baselines: threshold scores + logistic regression + optional LinearSVM
# ============================================================
def l1_dist(c1, c2):
    return np.sum(np.abs(c1 - c2), axis=1)

def l2_dist(c1, c2):
    return np.sqrt(np.sum((c1 - c2) ** 2, axis=1))

def cos_sim(c1, c2, eps=1e-12):
    num = np.sum(c1 * c2, axis=1)
    den = np.linalg.norm(c1, axis=1) * np.linalg.norm(c2, axis=1)
    den = np.maximum(den, eps)
    return num / den

def pick_threshold(scores: np.ndarray, y: np.ndarray, direction: str) -> float:
    """
    Select a scalar threshold using training labels.

    Labels:
        y = 0: same genus
        y = 1: different genus

    direction:
        "high_is_one": larger score predicts different genus; used for L1/L2 distances.
        "low_is_one":  smaller score predicts different genus; used for cosine similarity.
    """
    s = scores.astype(np.float64)
    uniq = np.unique(s)
    if uniq.size == 1:
        return float(uniq[0])

    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = np.concatenate(([uniq[0] - 1e-9], mids, [uniq[-1] + 1e-9]))

    best_t, best_acc = candidates[0], -1.0
    for t in candidates:
        if direction == "high_is_one":
            yhat = (s >= t).astype(np.int64)
        elif direction == "low_is_one":
            yhat = (s <= t).astype(np.int64)
        else:
            raise ValueError(direction)
        acc = accuracy_score(y, yhat)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return float(best_t)

def threshold_predict(scores: np.ndarray, t: float, direction: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return predicted labels and a simple confidence score.

    The confidence score is max(p, 1-p) after a monotone rescaling of the
    underlying scalar score. It is used only for ID-vs-OOD AUROC reporting.
    """
    s = scores.astype(np.float64)

    if direction == "high_is_one":
        yhat = (s >= t).astype(np.int64)
        # confidence: 멀수록(큰 점수) 1 쪽 확신, 작을수록 0 쪽 확신
        # conf를 0~1로 스케일 후, yhat에 맞게 '확신'으로 변환
        smin, smax = float(s.min()), float(s.max())
        denom = (smax - smin) if smax > smin else 1.0
        p1 = (s - smin) / denom  # 0~1
    elif direction == "low_is_one":
        yhat = (s <= t).astype(np.int64)
        smin, smax = float(s.min()), float(s.max())
        denom = (smax - smin) if smax > smin else 1.0
        p1 = 1.0 - (s - smin) / denom
    else:
        raise ValueError(direction)

    # "max-softmax" 스타일 confidence = max(p, 1-p)
    conf = np.maximum(p1, 1.0 - p1)
    conf = np.clip(conf, 0.0, 1.0)
    return yhat, conf

def weak_features(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    d1 = l1_dist(c1, c2)
    d2 = l2_dist(c1, c2)
    cs = cos_sim(c1, c2)
    n1 = np.linalg.norm(c1, axis=1)
    n2 = np.linalg.norm(c2, axis=1)
    m1 = c1.mean(axis=1)
    m2 = c2.mean(axis=1)
    diff = np.abs(c1 - c2)
    maxdiff = diff.max(axis=1)
    meandiff = diff.mean(axis=1)
    return np.stack([d1, d2, cs, n1, n2, m1, m2, maxdiff, meandiff], axis=1).astype(np.float32)

def ood_auroc_from_conf(conf_id: np.ndarray, conf_ood: np.ndarray) -> float:
    labels = np.concatenate([np.ones_like(conf_id), np.zeros_like(conf_ood)])
    scores = np.concatenate([conf_id, conf_ood])
    return float(roc_auc_score(labels, scores))


# ============================================================
# 8) Result structure
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



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline experiments for the genus-comparison benchmark."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=CACHE_DIR,
        help="Directory for cached catalogs and theta-coefficient matrices.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/baseline",
        help="Directory in which result CSV files and run_config_baseline.json are written.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Random seeds used for form splits, pair sampling, and model randomness.",
    )
    parser.add_argument(
        "--p-list",
        type=int,
        nargs="+",
        default=P_LIST,
        help="Theta-prefix truncation lengths.",
    )
    parser.add_argument(
        "--run-linear-svm",
        action="store_true",
        help="Also run the optional LinearSVM baseline. This is disabled in the main reported tables.",
    )
    parser.add_argument(
        "--no-disk-cache",
        action="store_true",
        help="Disable reading/writing cached catalogs and theta matrices.",
    )
    return parser.parse_args()


def write_run_config(outdir: str) -> None:
    config = {
        "script": "run_baselines.py",
        "task": "pairwise genus comparison from truncated theta-series coefficients",
        "seeds": SEEDS,
        "p_list": P_LIST,
        "max_p": MAX_P,
        "id_abs_disc_range": [ID_MIN_ABS_DISC, ID_MAX_ABS_DISC],
        "ood_abs_disc_range": [OOD_MIN_ABS_DISC, OOD_MAX_ABS_DISC],
        "id_form_test_size": ID_FORM_TEST_SIZE,
        "pair_budgets": {
            "id_train": {
                "same_D_same_genus": ID_TRAIN_SAME_D_SAME_G,
                "same_D_diff_genus": ID_TRAIN_SAME_D_DIFF_G,
                "diff_D": ID_TRAIN_DIFF_D_DIFF_G,
            },
            "id_test": {
                "same_D_same_genus": ID_TEST_SAME_D_SAME_G,
                "same_D_diff_genus": ID_TEST_SAME_D_DIFF_G,
                "diff_D": ID_TEST_DIFF_D_DIFF_G,
            },
            "ood_test": {
                "same_D_same_genus": OOD_SAME_D_SAME_G,
                "same_D_diff_genus": OOD_SAME_D_DIFF_G,
                "diff_D": OOD_DIFF_D_DIFF_G,
            },
        },
        "normalization_modes": NORMALIZATION_MODES,
        "zscore_fit_on_all_train_forms": ZSCORE_FIT_ON_ALL_TRAIN_FORMS,
        "zscore_default_fit_scope": "forms appearing in sampled ID training pairs",
        "cache_dir": CACHE_DIR,
        "use_disk_cache": USE_DISK_CACHE,
        "theta_progress_every": THETA_PROGRESS_EVERY,
        "models": ["Baseline_L1Thresh", "Baseline_L2Thresh", "Baseline_CosThresh", "Baseline_LogReg"]
                  + (["Baseline_LinearSVM"] if RUN_LINEAR_SVM else []),
        "logistic_regression_C_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
        "run_linear_svm": RUN_LINEAR_SVM,
        "outputs": {
            "per_seed": "results_baselines_per_seed.csv",
            "summary": "results_baselines_summary_mean_std.csv",
        },
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "run_config_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

# ============================================================
# 9) Main execution
# ============================================================
def main():
    args = parse_args()

    global CACHE_DIR, SEEDS, P_LIST, MAX_P, RUN_LINEAR_SVM, USE_DISK_CACHE
    CACHE_DIR = args.cache_dir
    SEEDS = args.seeds
    P_LIST = args.p_list
    MAX_P = max(P_LIST)
    RUN_LINEAR_SVM = args.run_linear_svm
    USE_DISK_CACHE = not args.no_disk_cache

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    write_run_config(outdir)

    out_per_seed = os.path.join(outdir, OUT_PER_SEED)
    out_summary = os.path.join(outdir, OUT_SUMMARY)

    # ---------------------------
    # (A) Catalog: load from disk cache when available
    # ---------------------------
    id_cat_path  = os.path.join(CACHE_DIR, f"id_catalog_{ID_MIN_ABS_DISC}_{ID_MAX_ABS_DISC}.npz")
    ood_cat_path = os.path.join(CACHE_DIR, f"ood_catalog_{OOD_MIN_ABS_DISC}_{OOD_MAX_ABS_DISC}.npz")

    if USE_DISK_CACHE and os.path.exists(id_cat_path):
        print("[Catalog] loading cached ID catalog...")
        a_id, b_id, c_id, absD_id, genus_id_int, genus_str_id, disc_map_id = load_catalog_npz(id_cat_path)
    else:
        print("[Catalog] building ID catalog...")
        a_id, b_id, c_id, absD_id, genus_id_int, genus_str_id, disc_map_id = build_catalog(ID_MIN_ABS_DISC, ID_MAX_ABS_DISC, True)
        if USE_DISK_CACHE:
            save_catalog_npz(id_cat_path, a_id, b_id, c_id, absD_id, genus_id_int, genus_str_id)
            print(f"[Catalog] saved {id_cat_path}")
    print(f" ID forms = {len(a_id)}")

    if USE_DISK_CACHE and os.path.exists(ood_cat_path):
        print("[Catalog] loading cached OOD catalog...")
        a_ood, b_ood, c_ood, absD_ood, genus_ood_int, genus_str_ood, disc_map_ood = load_catalog_npz(ood_cat_path)
    else:
        print("[Catalog] building OOD catalog...")
        a_ood, b_ood, c_ood, absD_ood, genus_ood_int, genus_str_ood, disc_map_ood = build_catalog(OOD_MIN_ABS_DISC, OOD_MAX_ABS_DISC, True)
        if USE_DISK_CACHE:
            save_catalog_npz(ood_cat_path, a_ood, b_ood, c_ood, absD_ood, genus_ood_int, genus_str_ood)
            print(f"[Catalog] saved {ood_cat_path}")
    print(f" OOD forms = {len(a_ood)}")

    # ---------------------------
    # (B) Global theta bound + theta matrix precomputation/load
    # ---------------------------
    print("\n[Theta] computing safe global bound B ...")
    B_id  = compute_global_B(a_id, b_id, c_id, MAX_P)
    B_ood = compute_global_B(a_ood, b_ood, c_ood, MAX_P)
    B = max(B_id, B_ood)
    print(f" B_id={B_id}, B_ood={B_ood} -> using B={B} (grid={(2*B+1)}x{(2*B+1)})")
    X2, XY, Y2 = theta_precompute_matrices(B)

    theta_id  = load_or_build_theta("id",  a_id,  b_id,  c_id,  X2, XY, Y2, MAX_P)
    theta_ood = load_or_build_theta("ood", a_ood, b_ood, c_ood, X2, XY, Y2, MAX_P)

    # ---------------------------
    # (C) Seed loop
    # ---------------------------
    all_results: List[PerSeedResult] = []

    for seed in SEEDS:
        print("\n" + "#" * 110)
        print(f"[SEED RUN] seed={seed}")
        print("#" * 110)
        set_seed(seed)
        rng = np.random.default_rng(seed)

        # 1) Form-disjoint split
        print("\n[1/5] form-disjoint split (ID)...")
        train_mask, test_mask = form_disjoint_split_id(
            n_forms=len(a_id),
            test_size=ID_FORM_TEST_SIZE,
            seed=seed,
            disc_map_id=disc_map_id,
            min_ge2_groups=1500,
            max_tries=50
        )
        disc_map_train = restrict_disc_map(disc_map_id, train_mask)
        disc_map_test  = restrict_disc_map(disc_map_id, test_mask)

        # 2) Complex pairs
        print("\n[2/5] sampling Complex pairs...")
        tr_same, tr_diff = make_complex_pairs(
            disc_map_train,
            ID_TRAIN_SAME_D_SAME_G, ID_TRAIN_SAME_D_DIFF_G, ID_TRAIN_DIFF_D_DIFF_G,
            rng
        )
        id_same, id_diff = make_complex_pairs(
            disc_map_test,
            ID_TEST_SAME_D_SAME_G, ID_TEST_SAME_D_DIFF_G, ID_TEST_DIFF_D_DIFF_G,
            rng
        )
        ood_same, ood_diff = make_complex_pairs(
            disc_map_ood,
            OOD_SAME_D_SAME_G, OOD_SAME_D_DIFF_G, OOD_DIFF_D_DIFF_G,
            rng
        )
        print(f" train pairs: same={len(tr_same)}, diff={len(tr_diff)}")
        print(f" ID-test pairs: same={len(id_same)}, diff={len(id_diff)}")
        print(f" OOD pairs: same={len(ood_same)}, diff={len(ood_diff)}")

        # 3) z-score fit (ID training data only)
        print("\n[3/5] fitting z-score from TRAIN only...")
        if ZSCORE_FIT_ON_ALL_TRAIN_FORMS:
            fit_idx = np.where(train_mask)[0].astype(np.int32)
        else:
            fit_idx = np.unique(np.concatenate([tr_same.ravel(), tr_diff.ravel()])).astype(np.int32)
        z_mean, z_std = fit_zscore(theta_id, fit_idx)

        # 4) P x normalization x baselines
        print("\n[4/5] baselines (P × norm) ...")
        for P in P_LIST:
            print("\n" + "=" * 100)
            print(f"[seed={seed}] P={P}")
            print("=" * 100)

            for norm in NORMALIZATION_MODES:
                print("-" * 100)
                print(f"[norm={norm}] building pair arrays ...")

                # train / id-test / ood arrays
                c1_tr, c2_tr, y_tr = build_pair_arrays(theta_id,  tr_same, tr_diff,  P, norm, z_mean, z_std)
                c1_id, c2_id, y_id = build_pair_arrays(theta_id,  id_same, id_diff,  P, norm, z_mean, z_std)
                c1_od, c2_od, y_od = build_pair_arrays(theta_ood, ood_same, ood_diff, P, norm, z_mean, z_std)

                # ---------- Threshold baselines ----------
                # L1: larger distance indicates higher likelihood of different genus (1).
                s_tr = l1_dist(c1_tr, c2_tr)
                t = pick_threshold(s_tr, y_tr, direction="high_is_one")
                yhat_id, conf_id = threshold_predict(l1_dist(c1_id, c2_id), t, "high_is_one")
                yhat_od, conf_od = threshold_predict(l1_dist(c1_od, c2_od), t, "high_is_one")
                all_results.append(PerSeedResult(seed, P, norm, "Baseline_L1Thresh",
                                                 float(accuracy_score(y_id, yhat_id)),
                                                 float(accuracy_score(y_od, yhat_od)),
                                                 ood_auroc_from_conf(conf_id, conf_od)))

                # L2
                s_tr = l2_dist(c1_tr, c2_tr)
                t = pick_threshold(s_tr, y_tr, direction="high_is_one")
                yhat_id, conf_id = threshold_predict(l2_dist(c1_id, c2_id), t, "high_is_one")
                yhat_od, conf_od = threshold_predict(l2_dist(c1_od, c2_od), t, "high_is_one")
                all_results.append(PerSeedResult(seed, P, norm, "Baseline_L2Thresh",
                                                 float(accuracy_score(y_id, yhat_id)),
                                                 float(accuracy_score(y_od, yhat_od)),
                                                 ood_auroc_from_conf(conf_id, conf_od)))

                # Cosine: smaller similarity indicates higher likelihood of different genus (1).
                s_tr = cos_sim(c1_tr, c2_tr)
                t = pick_threshold(s_tr, y_tr, direction="low_is_one")
                yhat_id, conf_id = threshold_predict(cos_sim(c1_id, c2_id), t, "low_is_one")
                yhat_od, conf_od = threshold_predict(cos_sim(c1_od, c2_od), t, "low_is_one")
                all_results.append(PerSeedResult(seed, P, norm, "Baseline_CosThresh",
                                                 float(accuracy_score(y_id, yhat_id)),
                                                 float(accuracy_score(y_od, yhat_od)),
                                                 ood_auroc_from_conf(conf_id, conf_od)))

                # ---------- Logistic regression on low-dimensional canonical pair features ----------
                Xtr = weak_features(c1_tr, c2_tr)
                Xid = weak_features(c1_id, c2_id)
                Xod = weak_features(c1_od, c2_od)

                # Select C on ID training accuracy over a small grid.
                Cs = [0.01, 0.1, 1.0, 10.0, 100.0]
                best = None
                for C in Cs:
                    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, class_weight="balanced")
                    lr.fit(Xtr, y_tr)
                    # Selection uses ID training accuracy only; ID test and OOD data are not used.
                    acc_tr = accuracy_score(y_tr, lr.predict(Xtr))
                    if best is None or acc_tr > best[0]:
                        best = (acc_tr, lr)

                lr = best[1]
                # classification
                yhat_id = lr.predict(Xid)
                yhat_od = lr.predict(Xod)
                # confidence: max(p,1-p)
                p_id = lr.predict_proba(Xid)[:, 1]
                p_od = lr.predict_proba(Xod)[:, 1]
                conf_id = np.maximum(p_id, 1.0 - p_id)
                conf_od = np.maximum(p_od, 1.0 - p_od)

                all_results.append(PerSeedResult(seed, P, norm, "Baseline_LogReg",
                                                 float(accuracy_score(y_id, yhat_id)),
                                                 float(accuracy_score(y_od, yhat_od)),
                                                 ood_auroc_from_conf(conf_id, conf_od)))

                # ---------- Optional: Linear SVM ----------
                if RUN_LINEAR_SVM:
                    # LinearSVC has no probabilities; use absolute decision value as confidence.
                    svm = LinearSVC(C=1.0, class_weight="balanced")
                    svm.fit(Xtr, y_tr)
                    yhat_id = (svm.decision_function(Xid) >= 0).astype(np.int64)
                    yhat_od = (svm.decision_function(Xod) >= 0).astype(np.int64)
                    conf_id = np.abs(svm.decision_function(Xid))
                    conf_od = np.abs(svm.decision_function(Xod))
                    # Normalize to [0,1] for numerical stability in AUROC reporting.
                    def norm01(x):
                        x = x.astype(np.float64)
                        mn, mx = float(x.min()), float(x.max())
                        d = (mx - mn) if mx > mn else 1.0
                        return (x - mn) / d
                    all_results.append(PerSeedResult(seed, P, norm, "Baseline_LinearSVM",
                                                     float(accuracy_score(y_id, yhat_id)),
                                                     float(accuracy_score(y_od, yhat_od)),
                                                     ood_auroc_from_conf(norm01(conf_id), norm01(conf_od))))

                # Memory cleanup
                del c1_tr, c2_tr, y_tr, c1_id, c2_id, y_id, c1_od, c2_od, y_od
                del Xtr, Xid, Xod
                gc.collect()

        print("\n[5/5] seed done.")
        gc.collect()

    # ---------------------------
    # (D) Save per-seed results
    # ---------------------------
    print("\n" + "=" * 110)
    print(f"[SAVE] {out_per_seed}")
    print("=" * 110)
    with open(out_per_seed, "w", encoding="utf-8") as f:
        f.write("seed,P,norm,model,id_acc,ood_acc,ood_auroc\n")
        for r in all_results:
            f.write(f"{r.seed},{r.P},{r.norm},{r.model},{r.id_acc:.6f},{r.ood_acc:.6f},{r.ood_auroc:.6f}\n")

    # ---------------------------
    # (E) Mean +/- std summary
    # ---------------------------
    bucket = defaultdict(list)
    for r in all_results:
        bucket[(r.P, r.norm, r.model)].append((r.id_acc, r.ood_acc, r.ood_auroc))

    summary = []
    for key, vals in bucket.items():
        arr = np.array(vals, dtype=np.float64)  # (nSeeds, 3)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(3)
        P, norm, model = key
        summary.append((P, norm, model, mean[0], std[0], mean[1], std[1], mean[2], std[2], arr.shape[0]))

    summary.sort(key=lambda x: (x[0], x[1], x[2]))

    print("\n" + "=" * 110)
    print("FINAL BASELINE SUMMARY (mean ± std over seeds)")
    print("=" * 110)
    print(f"{'P':>3} | {'Norm':<7} | {'Model':<18} | "
          f"{'ID Acc':>16} | {'OOD Acc':>16} | {'OOD AUROC':>18} | n")
    print("-" * 110)
    for row in summary:
        P, norm, model, id_m, id_s, ood_m, ood_s, au_m, au_s, n = row
        print(f"{P:>3} | {norm:<7} | {model:<18} | "
              f"{id_m:7.4f} ± {id_s:6.4f} | "
              f"{ood_m:7.4f} ± {ood_s:6.4f} | "
              f"{au_m:8.4f} ± {au_s:7.4f} | {n}")
    print("-" * 110)

    print("\n" + "=" * 110)
    print(f"[SAVE] {out_summary}")
    print("=" * 110)
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write("P,norm,model,id_acc_mean,id_acc_std,ood_acc_mean,ood_acc_std,ood_auroc_mean,ood_auroc_std,n\n")
        for row in summary:
            P, norm, model, id_m, id_s, ood_m, ood_s, au_m, au_s, n = row
            f.write(f"{P},{norm},{model},{id_m:.6f},{id_s:.6f},{ood_m:.6f},{ood_s:.6f},{au_m:.6f},{au_s:.6f},{n}\n")

    print("\nDONE.")



if __name__ == "__main__":
    main()
