
import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

EPS = 1e-9
ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "exp"
FIG_DIR = ROOT / "figures"

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "lines.linewidth": 2.2,
    "lines.markersize": 7,
})


@dataclass
class Box:
    lower: np.ndarray
    upper: np.ndarray


def ensure_dirs():
    for p in [EXP_DIR, FIG_DIR, EXP_DIR / "stage1", EXP_DIR / "stage2", EXP_DIR / "stage3"]:
        p.mkdir(parents=True, exist_ok=True)


def median_nan(s: pd.Series) -> float:
    arr = s.to_numpy(dtype=float)
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def datasets_round1():
    out = {}
    bc = load_breast_cancer()
    out["breast_cancer"] = (bc.data.astype(float), bc.target.astype(int), list(bc.feature_names))

    wine = load_wine()
    out["wine_binary"] = (wine.data.astype(float), (wine.target == 0).astype(int), list(wine.feature_names))

    iris = load_iris()
    out["iris_binary"] = (iris.data.astype(float), (iris.target == 0).astype(int), list(iris.feature_names))

    digits = load_digits()
    m = np.isin(digits.target, [0, 1])
    X01 = digits.data[m].astype(float)
    y01 = (digits.target[m] == 1).astype(int)
    out["digits_0_vs_1"] = (X01, y01, [f"pixel_{i}" for i in range(X01.shape[1])])

    Xs, ys = make_classification(
        n_samples=1600,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_classes=2,
        class_sep=1.3,
        flip_y=0.02,
        random_state=7,
    )
    out["synthetic_20"] = (Xs.astype(float), ys.astype(int), [f"feat_{i}" for i in range(Xs.shape[1])])
    return out


def leaf_boxes(clf: DecisionTreeClassifier):
    t = clf.tree_
    n_features = clf.n_features_in_
    boxes = {0: [], 1: []}

    def dfs(node, lower, upper):
        if t.children_left[node] == t.children_right[node]:
            pred = int(np.argmax(t.value[node][0]))
            boxes.setdefault(pred, []).append(Box(lower.copy(), upper.copy()))
            return
        f = int(t.feature[node])
        thr = float(t.threshold[node])

        ul = upper.copy()
        ul[f] = min(ul[f], thr)
        dfs(t.children_left[node], lower, ul)

        lr = lower.copy()
        # sklearn trees use float32 thresholds internally; tiny nextafter can vanish after casting.
        strict_thr = thr + max(1e-6, 1e-6 * abs(thr))
        lr[f] = max(lr[f], strict_thr)
        dfs(t.children_right[node], lr, upper)

    dfs(0, np.full(n_features, -np.inf), np.full(n_features, np.inf))
    return boxes


def project_box(x, lower, upper):
    z = x.copy()
    lm = np.isfinite(lower) & (z < lower)
    um = np.isfinite(upper) & (z > upper)
    z[lm] = lower[lm]
    z[um] = upper[um]
    d = z - x
    return z, float(np.dot(d, d))


def solve_exact(x, boxes, clf=None, target=None):
    if not boxes:
        return None, float("inf")
    lo = np.vstack([b.lower for b in boxes])
    up = np.vstack([b.upper for b in boxes])
    x2 = np.broadcast_to(x, lo.shape)
    z = np.where(x2 < lo, lo, x2)
    z = np.where(z > up, up, z)
    d = z - x2
    d2 = np.einsum("ij,ij->i", d, d)
    if clf is not None and target is not None:
        for i in np.argsort(d2):
            cand = z[int(i)]
            if int(clf.predict(cand.reshape(1, -1))[0]) == int(target):
                return cand, float(d2[int(i)])
        return None, float("inf")
    i = int(np.argmin(d2))
    return z[i], float(d2[i])


def solve_bruteforce(x, boxes, clf=None, target=None):
    best, best_d2 = None, float("inf")
    for b in boxes:
        z, d2 = project_box(x, b.lower, b.upper)
        if clf is not None and target is not None:
            if int(clf.predict(z.reshape(1, -1))[0]) != int(target):
                continue
        if d2 < best_d2:
            best, best_d2 = z, d2
    return best, best_d2


def solve_loop_variant(x, boxes, order=True, pruning=True, clf=None, target=None):
    if not boxes:
        return None, float("inf")
    idx = list(range(len(boxes)))
    if order:
        rank = []
        for i, b in enumerate(boxes):
            _, d2 = project_box(x, b.lower, b.upper)
            rank.append((d2, i))
        rank.sort(key=lambda t: t[0])
        idx = [i for _, i in rank]

    best, best_d2 = None, float("inf")
    for i in idx:
        b = boxes[i]
        z = x.copy()
        d2 = 0.0
        for j in range(x.shape[0]):
            v, lo, up = x[j], b.lower[j], b.upper[j]
            if v < lo:
                dv = lo - v
                z[j] = lo
                d2 += dv * dv
            elif v > up:
                dv = v - up
                z[j] = up
                d2 += dv * dv
            if pruning and d2 >= best_d2:
                break
        if clf is not None and target is not None:
            if int(clf.predict(z.reshape(1, -1))[0]) != int(target):
                continue
        if d2 < best_d2:
            best, best_d2 = z, d2
    return best, best_d2


def solve_nn(x, target, X_train, pred_train):
    idx = np.where(pred_train == target)[0]
    if idx.size == 0:
        return None, float("inf")
    c = X_train[idx]
    d = c - x
    d2 = np.einsum("ij,ij->i", d, d)
    i = int(np.argmin(d2))
    return c[i], float(d2[i])


def solve_random(x, target, clf, feat_std, rng, n_samples=480):
    best, best_d2 = None, float("inf")
    scales = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
    per = max(10, n_samples // len(scales))
    for s in scales:
        noise = rng.normal(0.0, s * feat_std, size=(per, x.shape[0]))
        c = x[None, :] + noise
        p = clf.predict(c)
        ok = c[p == target]
        if ok.shape[0] == 0:
            continue
        d = ok - x
        d2 = np.einsum("ij,ij->i", d, d)
        i = int(np.argmin(d2))
        if d2[i] < best_d2:
            best, best_d2 = ok[i], float(d2[i])
    return best, best_d2

def run_stage1():
    out = EXP_DIR / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    data = datasets_round1()

    for dname, (X, y, _) in data.items():
        for seed in [0, 1, 2, 3, 4]:
            rng = np.random.default_rng(seed)
            Xtr0, Xte0, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr0)
            Xte = sc.transform(Xte0)
            feat_std = np.clip(Xtr.std(axis=0), 1e-3, None)

            clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed)
            clf.fit(Xtr, ytr)
            ptr = clf.predict(Xtr)
            pte = clf.predict(Xte)
            boxes = leaf_boxes(clf)

            qidx = rng.choice(np.arange(Xte.shape[0]), size=min(60, Xte.shape[0]), replace=False)
            for qn, i in enumerate(qidx):
                x = Xte[i]
                p0 = int(pte[i])
                target = 1 - p0
                if not boxes.get(target, []):
                    continue
                qkey = f"{dname}|{seed}|{qn}"

                for method in ["treesat_exact", "leaf_enum_bruteforce", "nearest_neighbor", "random_search"]:
                    t0 = time.perf_counter()
                    if method == "treesat_exact":
                        z, d2 = solve_exact(x, boxes.get(target, []), clf=clf, target=target)
                    elif method == "leaf_enum_bruteforce":
                        z, d2 = solve_bruteforce(x, boxes.get(target, []), clf=clf, target=target)
                    elif method == "nearest_neighbor":
                        z, d2 = solve_nn(x, target, Xtr, ptr)
                    else:
                        rr = np.random.default_rng(seed * 100000 + qn * 131 + 17)
                        z, d2 = solve_random(x, target, clf, feat_std, rr)
                    rt = (time.perf_counter() - t0) * 1000.0

                    if z is None:
                        success, pc, dist, l0 = 0, -1, float("nan"), -1
                    else:
                        pc = int(clf.predict(z.reshape(1, -1))[0])
                        success = int(pc == target)
                        dist = float(np.sqrt(max(d2, 0.0)))
                        l0 = int(np.sum(np.abs(z - x) > EPS))

                    rows.append(
                        {
                            "stage": "stage1",
                            "dataset": dname,
                            "seed": seed,
                            "query_id": qn,
                            "query_key": qkey,
                            "method": method,
                            "pred_orig": p0,
                            "target_class": target,
                            "pred_cf": pc,
                            "success": success,
                            "distance_l2": dist,
                            "l0_changes": l0,
                            "runtime_ms": rt,
                            "n_leaves": int(clf.get_n_leaves()),
                            "n_nodes": int(clf.tree_.node_count),
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    summary = (
        df.groupby(["dataset", "method"], as_index=False)
        .agg(
            n=("success", "size"),
            success_rate=("success", "mean"),
            median_distance=("distance_l2", median_nan),
            median_l0=("l0_changes", median_nan),
            mean_runtime_ms=("runtime_ms", "mean"),
            median_runtime_ms=("runtime_ms", median_nan),
        )
        .sort_values(["dataset", "method"])
    )
    summary.to_csv(out / "summary.csv", index=False)

    ex = df[df.method == "treesat_exact"][["query_key", "distance_l2"]].rename(columns={"distance_l2": "exact"})
    br = df[df.method == "leaf_enum_bruteforce"][["query_key", "distance_l2"]].rename(columns={"distance_l2": "brute"})
    gap = ex.merge(br, on="query_key", how="inner")
    gap["optimality_gap"] = (gap["exact"] - gap["brute"]).abs()
    gap.to_csv(out / "optimality_gap.csv", index=False)

    agg = (
        summary.groupby("method", as_index=False)
        .agg(
            success_rate=("success_rate", "mean"),
            median_distance=("median_distance", median_nan),
            median_runtime_ms=("median_runtime_ms", median_nan),
        )
        .sort_values("method")
    )

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].bar(agg.method, agg.success_rate, color="#4C78A8")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title("Success Rate")
    ax[0].tick_params(axis="x", rotation=35)

    ax[1].bar(agg.method, agg.median_distance, color="#F58518")
    ax[1].set_title("Median L2 Distance")
    ax[1].tick_params(axis="x", rotation=35)

    ax[2].bar(agg.method, agg.median_runtime_ms, color="#54A24B")
    ax[2].set_title("Median Runtime (ms)")
    ax[2].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage1_main_metrics.png", dpi=180)
    plt.close(fig)

def run_stage2():
    out = EXP_DIR / "stage2"
    out.mkdir(parents=True, exist_ok=True)

    depth_rows = []
    for depth in [2, 4, 6, 8, 10, 12]:
        X, y = make_classification(
            n_samples=2800,
            n_features=30,
            n_informative=15,
            n_redundant=8,
            n_classes=2,
            class_sep=1.1,
            flip_y=0.01,
            random_state=depth,
        )
        Xtr0, Xte0, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=depth, stratify=y)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr0)
        Xte = sc.transform(Xte0)
        clf = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=depth)
        clf.fit(Xtr, ytr)
        pte = clf.predict(Xte)
        boxes = leaf_boxes(clf)
        rng = np.random.default_rng(1000 + depth)
        qidx = rng.choice(np.arange(Xte.shape[0]), size=min(160, Xte.shape[0]), replace=False)

        for method in ["treesat_exact", "leaf_enum_bruteforce"]:
            rt = []
            for i in qidx:
                x = Xte[i]
                target = 1 - int(pte[i])
                if not boxes.get(target, []):
                    continue
                t0 = time.perf_counter()
                if method == "treesat_exact":
                    _ = solve_exact(x, boxes.get(target, []), clf=clf, target=target)
                else:
                    _ = solve_bruteforce(x, boxes.get(target, []), clf=clf, target=target)
                rt.append((time.perf_counter() - t0) * 1000.0)
            depth_rows.append(
                {
                    "depth": depth,
                    "method": method,
                    "mean_runtime_ms": float(np.mean(rt)),
                    "median_runtime_ms": float(np.median(rt)),
                    "n_nodes": int(clf.tree_.node_count),
                    "n_leaves": int(clf.get_n_leaves()),
                    "n_queries": len(qidx),
                }
            )

    feat_rows = []
    for nf in [5, 10, 20, 40, 80, 120]:
        X, y = make_classification(
            n_samples=3200,
            n_features=nf,
            n_informative=max(4, nf // 3),
            n_redundant=max(1, nf // 5),
            n_classes=2,
            class_sep=1.1,
            flip_y=0.01,
            random_state=2000 + nf,
        )
        Xtr0, Xte0, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=2000 + nf, stratify=y)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr0)
        Xte = sc.transform(Xte0)
        clf = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=3000 + nf)
        clf.fit(Xtr, ytr)
        pte = clf.predict(Xte)
        boxes = leaf_boxes(clf)
        rng = np.random.default_rng(4000 + nf)
        qidx = rng.choice(np.arange(Xte.shape[0]), size=min(160, Xte.shape[0]), replace=False)

        for method in ["treesat_exact", "leaf_enum_bruteforce"]:
            rt = []
            for i in qidx:
                x = Xte[i]
                target = 1 - int(pte[i])
                if not boxes.get(target, []):
                    continue
                t0 = time.perf_counter()
                if method == "treesat_exact":
                    _ = solve_exact(x, boxes.get(target, []), clf=clf, target=target)
                else:
                    _ = solve_bruteforce(x, boxes.get(target, []), clf=clf, target=target)
                rt.append((time.perf_counter() - t0) * 1000.0)
            feat_rows.append(
                {
                    "n_features": nf,
                    "method": method,
                    "mean_runtime_ms": float(np.mean(rt)),
                    "median_runtime_ms": float(np.median(rt)),
                    "n_nodes": int(clf.tree_.node_count),
                    "n_leaves": int(clf.get_n_leaves()),
                    "n_queries": len(qidx),
                }
            )

    depth_df = pd.DataFrame(depth_rows)
    feat_df = pd.DataFrame(feat_rows)
    depth_df.to_csv(out / "scalability_depth.csv", index=False)
    feat_df.to_csv(out / "scalability_features.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in depth_df.method.unique():
        s = depth_df[depth_df.method == m].sort_values("depth")
        ax.plot(s.depth, s.median_runtime_ms, marker="o", label=m)
    ax.set_xlabel("Tree Depth")
    ax.set_ylabel("Median Runtime (ms)")
    ax.set_title("Scalability vs Depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage2_scalability_depth.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in feat_df.method.unique():
        s = feat_df[feat_df.method == m].sort_values("n_features")
        ax.plot(s.n_features, s.median_runtime_ms, marker="o", label=m)
    ax.set_xlabel("Feature Count")
    ax.set_ylabel("Median Runtime (ms)")
    ax.set_title("Scalability vs Features")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage2_scalability_features.png", dpi=180)
    plt.close(fig)

    ab_rows = []
    for dname, (X, y, _) in datasets_round1().items():
        for seed in [0, 1, 2, 3, 4]:
            rng = np.random.default_rng(5000 + seed)
            Xtr0, Xte0, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr0)
            Xte = sc.transform(Xte0)
            clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed)
            clf.fit(Xtr, ytr)
            pte = clf.predict(Xte)
            base = leaf_boxes(clf)
            qidx = rng.choice(np.arange(Xte.shape[0]), size=min(70, Xte.shape[0]), replace=False)

            for qn, i in enumerate(qidx):
                x = Xte[i]
                target = 1 - int(pte[i])
                if not base.get(target, []):
                    continue
                qkey = f"{dname}|{seed}|{qn}"
                for method in ["full", "no_cache", "no_order", "no_pruning"]:
                    t0 = time.perf_counter()
                    if method == "full":
                        z, d2 = solve_exact(x, base.get(target, []), clf=clf, target=target)
                    elif method == "no_cache":
                        fresh = leaf_boxes(clf)
                        z, d2 = solve_exact(x, fresh.get(target, []), clf=clf, target=target)
                    elif method == "no_order":
                        z, d2 = solve_loop_variant(
                            x, base.get(target, []), order=False, pruning=True, clf=clf, target=target
                        )
                    else:
                        z, d2 = solve_loop_variant(
                            x, base.get(target, []), order=True, pruning=False, clf=clf, target=target
                        )
                    rt = (time.perf_counter() - t0) * 1000.0
                    succ = int(z is not None and int(clf.predict(z.reshape(1, -1))[0]) == target)
                    dist = float(np.sqrt(max(d2, 0.0))) if z is not None else float("nan")
                    ab_rows.append(
                        {
                            "dataset": dname,
                            "seed": seed,
                            "query_key": qkey,
                            "method": method,
                            "runtime_ms": rt,
                            "distance_l2": dist,
                            "success": succ,
                        }
                    )

    ab = pd.DataFrame(ab_rows)
    ab.to_csv(out / "ablation_results.csv", index=False)
    ab_summary = (
        ab.groupby(["dataset", "method"], as_index=False)
        .agg(
            n=("success", "size"),
            success_rate=("success", "mean"),
            median_runtime_ms=("runtime_ms", median_nan),
            median_distance=("distance_l2", median_nan),
        )
        .sort_values(["dataset", "method"])
    )
    ab_summary.to_csv(out / "ablation_summary.csv", index=False)

    full = ab[ab.method == "full"][["query_key", "distance_l2"]].rename(columns={"distance_l2": "dist_full"})
    for m in ["no_cache", "no_order", "no_pruning"]:
        comp = ab[ab.method == m][["query_key", "distance_l2"]].rename(columns={"distance_l2": f"dist_{m}"})
        full = full.merge(comp, on="query_key", how="left")
        full[f"gap_{m}"] = (full["dist_full"] - full[f"dist_{m}"]).abs()
    full.to_csv(out / "ablation_distance_gap.csv", index=False)

    g = (
        ab.groupby("method", as_index=False)
        .agg(median_runtime_ms=("runtime_ms", median_nan), success_rate=("success", "mean"))
        .sort_values("median_runtime_ms")
    )
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    ax[0].bar(g.method, g.median_runtime_ms, color="#72B7B2")
    ax[0].set_title("Ablation Runtime")
    ax[0].set_ylabel("Median Runtime (ms)")
    ax[0].tick_params(axis="x", rotation=30)

    ax[1].bar(g.method, g.success_rate, color="#E45756")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Ablation Success Rate")
    ax[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage2_ablation.png", dpi=180)
    plt.close(fig)

def run_stage3():
    out = EXP_DIR / "stage3"
    out.mkdir(parents=True, exist_ok=True)
    data = datasets_round1()

    st_rows = []
    for dname, (X, y, _) in data.items():
        for seed in list(range(20)):
            rng = np.random.default_rng(7000 + seed)
            Xtr0, Xte0, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr0)
            Xte = sc.transform(Xte0)
            feat_std = np.clip(Xtr.std(axis=0), 1e-3, None)
            clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed)
            clf.fit(Xtr, ytr)
            ptr = clf.predict(Xtr)
            pte = clf.predict(Xte)
            boxes = leaf_boxes(clf)

            qidx = rng.choice(np.arange(Xte.shape[0]), size=min(30, Xte.shape[0]), replace=False)
            for qn, i in enumerate(qidx):
                x = Xte[i]
                target = 1 - int(pte[i])
                if not boxes.get(target, []):
                    continue
                for method in ["treesat_exact", "nearest_neighbor", "random_search"]:
                    t0 = time.perf_counter()
                    if method == "treesat_exact":
                        z, d2 = solve_exact(x, boxes.get(target, []), clf=clf, target=target)
                    elif method == "nearest_neighbor":
                        z, d2 = solve_nn(x, target, Xtr, ptr)
                    else:
                        rr = np.random.default_rng(seed * 40000 + qn * 97 + 23)
                        z, d2 = solve_random(x, target, clf, feat_std, rr, n_samples=360)
                    rt = (time.perf_counter() - t0) * 1000.0
                    succ = int(z is not None and int(clf.predict(z.reshape(1, -1))[0]) == target)
                    dist = float(np.sqrt(max(d2, 0.0))) if z is not None else float("nan")
                    st_rows.append(
                        {
                            "dataset": dname,
                            "seed": seed,
                            "query_id": qn,
                            "method": method,
                            "runtime_ms": rt,
                            "distance_l2": dist,
                            "success": succ,
                        }
                    )

    st = pd.DataFrame(st_rows)
    st.to_csv(out / "stability_results.csv", index=False)
    st_seed = (
        st.groupby(["dataset", "seed", "method"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            median_distance=("distance_l2", median_nan),
            median_runtime_ms=("runtime_ms", median_nan),
        )
        .sort_values(["dataset", "seed", "method"])
    )
    st_seed.to_csv(out / "stability_seed_summary.csv", index=False)

    exact = st_seed[st_seed.method == "treesat_exact"]
    dnames = sorted(exact.dataset.unique())
    box_data = [exact[exact.dataset == ds].median_distance.dropna().to_numpy() for ds in dnames]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(box_data, tick_labels=dnames, showfliers=False)
    ax.set_ylabel("Median L2 Distance Across Seeds")
    ax.set_title("Stability Distribution (TreeSAT Exact)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage3_stability_boxplot.png", dpi=180)
    plt.close(fig)

    nz_rows = []
    sigmas = [0.0, 0.01, 0.03, 0.05, 0.1]
    for dname, (X, y, _) in data.items():
        for seed in [0, 1, 2, 3, 4]:
            rng = np.random.default_rng(9000 + seed)
            Xtr0, Xte0, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr0)
            Xte = sc.transform(Xte0)
            clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed)
            clf.fit(Xtr, ytr)
            pte = clf.predict(Xte)
            boxes = leaf_boxes(clf)

            qidx = rng.choice(np.arange(Xte.shape[0]), size=min(35, Xte.shape[0]), replace=False)
            for qn, i in enumerate(qidx):
                x0 = Xte[i]
                target = 1 - int(pte[i])
                if not boxes.get(target, []):
                    continue
                z0, d20 = solve_exact(x0, boxes.get(target, []), clf=clf, target=target)
                d0 = float(np.sqrt(max(d20, 0.0))) if z0 is not None else float("nan")
                for s in sigmas:
                    xn = x0 + rng.normal(0.0, s, size=x0.shape)
                    t0 = time.perf_counter()
                    z, d2 = solve_exact(xn, boxes.get(target, []), clf=clf, target=target)
                    rt = (time.perf_counter() - t0) * 1000.0
                    succ = int(z is not None and int(clf.predict(z.reshape(1, -1))[0]) == target)
                    dist = float(np.sqrt(max(d2, 0.0))) if z is not None else float("nan")
                    infl = dist / d0 if d0 > EPS and np.isfinite(dist) else float("nan")
                    nz_rows.append(
                        {
                            "dataset": dname,
                            "seed": seed,
                            "query_id": qn,
                            "noise_sigma": s,
                            "runtime_ms": rt,
                            "success": succ,
                            "distance_l2": dist,
                            "clean_distance_l2": d0,
                            "distance_inflation": infl,
                        }
                    )

    nz = pd.DataFrame(nz_rows)
    nz.to_csv(out / "noise_robustness_results.csv", index=False)
    nz_sum = (
        nz.groupby(["dataset", "noise_sigma"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            median_distance=("distance_l2", median_nan),
            median_inflation=("distance_inflation", median_nan),
        )
        .sort_values(["dataset", "noise_sigma"])
    )
    nz_sum.to_csv(out / "noise_robustness_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for ds in sorted(nz_sum.dataset.unique()):
        s = nz_sum[nz_sum.dataset == ds].sort_values("noise_sigma")
        ax.plot(s.noise_sigma, s.median_inflation, marker="o", label=ds)
    ax.set_xlabel("Gaussian Noise Sigma")
    ax.set_ylabel("Median Distance Inflation")
    ax.set_title("Noise Robustness of TreeSAT Exact")
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stage3_noise_robustness.png", dpi=180)
    plt.close(fig)

    cs_rows = []
    for dname, (X, y, fnames) in data.items():
        Xtr0, Xte0, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr0)
        Xte = sc.transform(Xte0)
        clf = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=0)
        clf.fit(Xtr, ytr)
        pte = clf.predict(Xte)
        boxes = leaf_boxes(clf)

        for i in range(Xte.shape[0]):
            x = Xte[i]
            target = 1 - int(pte[i])
            z, d2 = solve_exact(x, boxes.get(target, []), clf=clf, target=target)
            if z is None:
                continue
            x_raw = sc.inverse_transform(x.reshape(1, -1))[0]
            z_raw = sc.inverse_transform(z.reshape(1, -1))[0]
            delta = np.abs(z_raw - x_raw)
            rank = np.argsort(-delta)
            top = [j for j in rank if delta[j] > 1e-12][:3]
            if not top:
                top = [int(rank[0])]
            txt = []
            for j in top:
                txt.append(f"{fnames[j]}: {x_raw[j]:.4f} -> {z_raw[j]:.4f} (|d|={delta[j]:.4f})")
            cs_rows.append(
                {
                    "dataset": dname,
                    "seed": 0,
                    "test_index": int(i),
                    "target_class": int(target),
                    "distance_l2": float(np.sqrt(max(d2, 0.0))),
                    "n_changed_features": int(np.sum(np.abs(z - x) > EPS)),
                    "top_feature_changes": " | ".join(txt),
                }
            )
            break

    pd.DataFrame(cs_rows).to_csv(out / "case_studies.csv", index=False)


def write_repro():
    meta = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "matplotlib_version": plt.matplotlib.__version__,
        "timing_policy": "solver runtime only; excludes model training",
        "timeout_policy": "no hard timeout; all queries completed",
    }
    (EXP_DIR / "reproducibility.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def final_assessment():
    checks = {}
    p_s1 = EXP_DIR / "stage1" / "summary.csv"
    p_gap = EXP_DIR / "stage1" / "optimality_gap.csv"
    p_depth = EXP_DIR / "stage2" / "scalability_depth.csv"
    p_feat = EXP_DIR / "stage2" / "scalability_features.csv"
    p_ab = EXP_DIR / "stage2" / "ablation_summary.csv"
    p_st = EXP_DIR / "stage3" / "stability_seed_summary.csv"
    p_nz = EXP_DIR / "stage3" / "noise_robustness_summary.csv"
    p_cs = EXP_DIR / "stage3" / "case_studies.csv"

    if p_s1.exists():
        s1 = pd.read_csv(p_s1)
        checks["dataset_count"] = int(s1.dataset.nunique())
        checks["method_count"] = int(s1.method.nunique())
    if p_gap.exists():
        gap = pd.read_csv(p_gap)
        checks["max_optimality_gap"] = float(gap.optimality_gap.max()) if len(gap) else float("inf")
    if p_depth.exists() and p_feat.exists():
        ddf = pd.read_csv(p_depth)
        fdf = pd.read_csv(p_feat)
        checks["scalability_depth_points"] = int(ddf.depth.nunique())
        checks["scalability_feature_points"] = int(fdf.n_features.nunique())
    if p_ab.exists():
        ab = pd.read_csv(p_ab)
        checks["ablation_methods"] = int(ab.method.nunique())
    if p_st.exists():
        st = pd.read_csv(p_st)
        checks["stability_seed_count"] = int(st.seed.nunique())
    if p_nz.exists():
        nz = pd.read_csv(p_nz)
        checks["noise_levels"] = int(nz.noise_sigma.nunique())
    if p_cs.exists():
        cs = pd.read_csv(p_cs)
        checks["case_study_datasets"] = int(cs.dataset.nunique())

    achieved = (
        checks.get("dataset_count", 0) >= 5
        and checks.get("method_count", 0) >= 4
        and checks.get("max_optimality_gap", 1.0) <= 1e-8
        and checks.get("scalability_depth_points", 0) >= 6
        and checks.get("scalability_feature_points", 0) >= 6
        and checks.get("ablation_methods", 0) >= 4
        and checks.get("stability_seed_count", 0) >= 20
        and checks.get("noise_levels", 0) >= 5
        and checks.get("case_study_datasets", 0) >= 5
    )

    out = {"target": "SCI/SSCI Q2 experimental evidence bar", "checks": checks, "achieved": bool(achieved)}
    (EXP_DIR / "final_assessment.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="SCI Q2-oriented workflow for Idea 35")
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3", "all"], default="all")
    args = parser.parse_args()

    ensure_dirs()
    write_repro()

    if args.stage in ["stage1", "all"]:
        run_stage1()
    if args.stage in ["stage2", "all"]:
        run_stage2()
    if args.stage in ["stage3", "all"]:
        run_stage3()

    final_assessment()
    print("Workflow run complete.")


if __name__ == "__main__":
    main()
