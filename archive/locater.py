import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- Eq. (2.12): normalized pairing distance (uses Hungarian) ----------
def _pairing_distance(U_centers: np.ndarray,
                      S_centers: np.ndarray,
                      diam_D: float) -> float:
    """
    Normalized set distance from Eq. (2.12) in the paper.
    U_centers, S_centers: shape (k, d), same k and d
    diam_D: positive float, diameter of domain D
    """
    U_centers = np.asarray(U_centers, dtype=float)
    S_centers = np.asarray(S_centers, dtype=float)
    assert U_centers.shape == S_centers.shape, "Sets must have the same shape."
    k = U_centers.shape[0]
    if k == 0:
        return 0.0
    # Cost = L2 distances
    diff = U_centers[:, None, :] - S_centers[None, :, :]
    cost = np.linalg.norm(diff, axis=2)  # (k, k)
    r, c = linear_sum_assignment(cost)
    avg = cost[r, c].sum() / float(k)
    return float(avg / diam_D)


# ---------- Algorithm 2: Function Cluster(U) ----------
def cluster_points(U: np.ndarray, Cmax: int, sil1: float, random_state: int = 0) -> np.ndarray:
    """
    Implements Algorithm 2's Cluster(U):
      - Try k=2..Cmax, pick the one with best silhouette.
      - If best silhouette < sil1, return the single mean as 1 center.
      - Else return KMeans(k) cluster centers.

    Parameters
    ----------
    U : (N, d) ndarray of collected points (PSNN >= threshold)
    Cmax : int >= 2
    sil1 : float in (0,1)
    random_state : int

    Returns
    -------
    centers : (k, d) ndarray (k>=1); k may be 1 if fallback is triggered.
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[0] == 0:
        return np.empty((0, U.shape[1] if U.ndim == 2 else 0))

    n_samples, d = U.shape

    # Too few samples to form 2 clusters → mean
    # if n_samples < 3:
    #     return U.mean(axis=0, keepdims=True)

    if n_samples == 0:
        return np.empty((0, d))
    if n_samples <= 2:
        return U.copy()  # each point is its own "cluster center"

    best_k, best_sil = None, -np.inf
    upper = min(Cmax, n_samples - 1)
    for k in range(2, max(2, upper) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(U)
        if len(set(labels)) < 2:
            continue
        try:
            sil = silhouette_score(U, labels, metric='euclidean')
        except Exception:
            sil = -np.inf
        if sil > best_sil:
            best_sil = sil
            best_k = k

    if best_k is not None and best_sil >= sil1:
        centers = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit(U).cluster_centers_
    else:
        centers = U.mean(axis=0, keepdims=True)

    return centers


# ---------- Algorithm 2: Function Averageerror(L) ----------
# def average_error(L: float,
#                   psnn,
#                   U_all: np.ndarray,
#                   O_search: list,
#                   Cmax: int,
#                   sil1: float,
#                   diam_D: float,
#                   random_state: int = 0) -> float:
#     U_all = torch.from_numpy(U_all).float().to(device)
#     if len(O_search) == 0:
#         return 0.0

#     total_err = 0.0
#     for case in O_search:
#         theta_i = case["Theta"]
#         S_i = np.vstack([sol['u'] for sol in case['U']]) if len(case['U']) > 0 else np.empty((0, U_all.shape[1]))
#         # Direct evaluation (no batching)
#         # scores = psnn(U_all, torch.tensor(theta_i).float().to(device)).flatten().cpu().detach().numpy()
#         # scores = psnn(
#         #     torch.from_numpy(U_all).float().to(device),
#         #     torch.tensor(theta_i).float().to(device).unsqueeze(0).repeat(U_all.shape[0], 1)
#         # ).flatten().cpu().detach().numpy()

#         # mask = scores >= L
#         # U_collected = U_all[mask, :]  # here U_all must still be numpy


#         U_all_t = torch.from_numpy(U_all).float().to(device)

#         theta_t = torch.tensor(theta_i).float().to(device)
#         theta_t = theta_t.unsqueeze(0).repeat(U_all_t.shape[0], 1)  # (B,2)

#         with torch.no_grad():
#             scores_t = psnn(U_all_t, theta_t).flatten()

#         # scores = scores_t.cpu().numpy()
#         # Collect above threshold
#         U_collected = U_all[scores_t >= L, :]

#         # Cluster
#         centers_i = cluster_points(U_collected.cpu().numpy(), Cmax=Cmax, sil1=sil1, random_state=random_state)
#         # Error add: pairing distance if counts match; else penalty 1
#         if centers_i.shape[0] == len(S_i):
#             total_err += _pairing_distance(centers_i, S_i, diam_D=diam_D)
#         else:
#             total_err += 1.0

#     return total_err / float(len(O_search))

def average_error(L: float,
                  psnn,
                  U_all: np.ndarray,
                  O_search: list,
                  Cmax: int,
                  sil1: float,
                  diam_D: float,
                  random_state: int = 0) -> float:
    # Keep a NumPy copy for shapes and later use
    U_all_np = np.asarray(U_all, dtype=float)
    # One-time torch conversion
    U_all_t = torch.from_numpy(U_all_np).float().to(device)

    if len(O_search) == 0:
        return 0.0

    total_err = 0.0
    for case in O_search:
        theta_i = case["Theta"]  # shape (2,)
        # True steady states S_i
        if len(case["U"]) > 0:
            S_i = np.vstack([sol["u"] for sol in case["U"]])  # (k,d)
        else:
            S_i = np.empty((0, U_all_np.shape[1]))

        # Build batched Theta: (B, dim_theta)
        theta_t = torch.tensor(theta_i, dtype=torch.float32, device=device)
        theta_t = theta_t.unsqueeze(0).repeat(U_all_t.shape[0], 1)

        # Evaluate PSNN on all U_all for this theta
        with torch.no_grad():
            scores_t = psnn(U_all_t, theta_t).flatten()  # (B,)

        # Thresholding
        mask = scores_t >= L
        U_collected_t = U_all_t[mask, :]         # (N_sel, d) tensor
        U_collected   = U_collected_t.cpu().numpy()

        # Cluster
        centers_i = cluster_points(U_collected, Cmax=Cmax, sil1=sil1, random_state=random_state)

        # Error: pairing distance if counts match; else penalty 1
        if centers_i.shape[0] == len(S_i):
            total_err += _pairing_distance(centers_i, S_i, diam_D=diam_D)
        else:
            total_err += 1.0

    return total_err / float(len(O_search))

# ---------- Optional: scan L to pick the best cut ----------
def pick_best_cut(psnn,
                  U_all: np.ndarray,
                  O_search: list,
                  Cmax: int,
                  sil1: float,
                  diam_D: float,
                  L_grid: np.ndarray,
                  random_state: int = 0):
    """
    Utility to pick L_cut by scanning a grid and minimizing Averageerror(L).
    Returns (L_best, errors_array).
    """
    errs = []
    for L in L_grid:
        print(f"Evaluating L={L:.4f}...", end='', flush=True)
        err = average_error(L, psnn, U_all, O_search, Cmax, sil1, diam_D, random_state=random_state)
        errs.append(err)
    errs = np.array(errs, dtype=float)
    return float(L_grid[int(np.argmin(errs))]), errs