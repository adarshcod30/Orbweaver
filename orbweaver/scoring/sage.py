"""A GraphSAGE scorer, reported beside the gradient-boosted one.

The account scorer is a component, not the contribution, and the ring
extractor downstream is scorer-agnostic. So the question here is narrow: does
a graph neural network, which can propagate information along edges rather
than only summarising a neighbourhood into fixed features, score accounts
better than XGBoost on this graph?

GADBench (NeurIPS 2023) found that gradient boosting over graph-aggregated
features usually wins on real anomaly-detection graphs, and PromoGuardian's
own baseline table has GraphSAGE at F1 0.2810 against FRAUDAR's 0.4715 on this
very dataset. I expect XGBoost to hold up. Whichever way it falls, both
numbers go in the results.

The graph is far too large for full-batch training on 16 GB, so this uses
neighbour sampling: each step draws a small batch of seed accounts and samples
a bounded neighbourhood around them, which keeps memory flat regardless of
graph size. Training uses the early window and scoring uses the late one, and
the held-out accounts are absent from training exactly as they are for the
gradient-boosted scorer.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.data.windows import EARLY, LATE
from orbweaver.features.node_features import FEATURE_NAMES


def pick_device():
    """CPU by default, because it is the only one that reproduces exactly.

    Metal's scatter reductions accumulate in a non-deterministic order, so the
    same seed on MPS gives a slightly different model every run. Measured:
    twelve identical training batches diverge at batch 7 by 6e-8, and across
    full runs held-out AUPRC moved between 0.3825 and 0.3833. That is far too
    small to change any conclusion here, and still enough to break the promise
    that a run is reproducible, so the default is the device that keeps it.

    Set ORBWEAVER_SAGE_DEVICE=mps for the roughly 3x faster run when exact
    reproducibility is not what you need.
    """
    import os

    import torch

    want = os.environ.get("ORBWEAVER_SAGE_DEVICE", "cpu").lower()
    if want == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_data(cfg: Config, tag: str, n_users: int):
    """Standardised features plus a CSR adjacency for the sampler."""
    from orbweaver.scoring.sampler import build_csr

    proc = cfg.abs_path(cfg.paths.processed)
    t = pq.read_table(proc / f"features_week2_{tag}.parquet")
    uid = t["user_id"].to_numpy()
    X = np.zeros((n_users, len(FEATURE_NAMES)), dtype=np.float32)
    cols = np.column_stack([t[c].to_numpy() for c in FEATURE_NAMES]).astype(np.float32)
    keep = uid < n_users
    X[uid[keep]] = cols[keep]
    # Standardise: raw degrees and order counts span orders of magnitude and a
    # GNN, unlike a tree, is sensitive to that.
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd

    e = pq.read_table(proc / f"edges_week2_{tag}.parquet", columns=["src", "dst"])
    indptr, indices = build_csr(e["src"].to_numpy().astype(np.int64),
                                e["dst"].to_numpy().astype(np.int64), n_users)
    return X, indptr, indices


class SAGE:
    """Two-layer GraphSAGE. Kept small deliberately - the labelled signal here
    is weak and a larger model just memorises the training accounts."""

    def __init__(self, in_dim: int, hidden: int = 64, seed: int = 0):
        import torch
        from torch_geometric.nn import SAGEConv

        torch.manual_seed(seed)

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = SAGEConv(in_dim, hidden)
                self.c2 = SAGEConv(hidden, hidden)
                self.out = torch.nn.Linear(hidden, 1)
                self.drop = torch.nn.Dropout(0.3)

            def forward(self, x, edge_index):
                x = torch.relu(self.c1(x, edge_index))
                x = self.drop(x)
                x = torch.relu(self.c2(x, edge_index))
                return self.out(x).squeeze(-1)

        self.net = Net()


def train_sage(cfg: Config | None = None, epochs: int = 3) -> dict:
    import torch

    from eval.metrics import evaluate, ltv_proxy
    from eval.split import make_split
    from orbweaver.scoring.sampler import batches, sample_subgraph

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    n = split.labels.size
    device = pick_device()
    print(f"device: {device}")

    X, indptr, indices = build_data(cfg, EARLY, n)
    y_all = np.zeros(n, dtype=np.float32)
    for idx in (split.train, split.val, split.test):
        y_all[idx] = split.y(idx).astype(np.float32)

    model = SAGE(len(FEATURE_NAMES), seed=cfg.seed).net.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    pos = float(split.y(split.train).sum())
    neg = float(split.train.size - pos)
    lossf = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(neg / max(pos, 1)).to(device))

    rng = np.random.default_rng(cfg.seed)
    fanouts = (15, 10)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        total, seen = 0.0, 0
        for seeds in batches(split.train, 1024, rng):
            nodes, ei, n_seed = sample_subgraph(indptr, indices, seeds, fanouts, rng)
            xb = torch.from_numpy(X[nodes]).to(device)
            eb = torch.from_numpy(ei).to(device)
            yb = torch.from_numpy(y_all[nodes[:n_seed]]).to(device)
            opt.zero_grad()
            out = model(xb, eb)[:n_seed]
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * n_seed
            seen += n_seed
        print(f"  epoch {ep + 1}/{epochs}  loss {total / max(seen, 1):.4f}  "
              f"({time.time() - t0:.0f}s)")

    # Score on the late window, the window the other scorer is judged on.
    Xl, lptr, lidx = build_data(cfg, LATE, n)
    model.eval()
    scores = np.zeros(n, dtype=np.float32)
    targets = np.concatenate([split.test, split.train_pool])
    with torch.no_grad():
        for seeds in batches(targets, 4096, rng, shuffle=False):
            nodes, ei, n_seed = sample_subgraph(lptr, lidx, seeds, fanouts, rng)
            xb = torch.from_numpy(Xl[nodes]).to(device)
            eb = torch.from_numpy(ei).to(device)
            out = torch.sigmoid(model(xb, eb)[:n_seed])
            scores[nodes[:n_seed]] = out.cpu().numpy()

    f = pq.read_table(proc / f"features_week2_{EARLY}.parquet",
                      columns=["user_id", "n_orders"])
    orders_n = np.zeros(n)
    uid = f["user_id"].to_numpy(); keep = uid < n
    orders_n[uid[keep]] = f["n_orders"].to_numpy()[keep]
    ltv = ltv_proxy(orders_n, cfg.cost.assumed_avg_order_value_inr)

    out = {
        "device": str(device), "epochs": epochs,
        "fanout": [15, 10], "hidden": 64,
        "train_seconds": round(time.time() - t0, 1),
        "results": {
            "test_heldout__labelled_only": evaluate(
                split.y(split.test), scores[split.test], ltv[split.test]),
            "train_pool__labelled_only": evaluate(
                split.y(split.train_pool), scores[split.train_pool], ltv[split.train_pool]),
        },
    }
    np.save(proc / "scores_sage_week2.npy", scores)
    return out


def main() -> None:
    # torch and torch_geometric are optional extras, so this stage skips
    # rather than failing when they are absent. It still belongs in
    # `make reproduce`: leaving it out meant the comparison table in
    # docs/results.md depended on an artefact nothing in the pipeline
    # produced, and clearing data/processed silently deleted the GNN row.
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
    except ImportError as exc:
        print(f"skipping the GNN scorer: {exc.name} is not installed. "
              "It is an optional extra; see requirements.txt.")
        return

    cfg = load_config()
    out = train_sage(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "sage_report.json"
    dest.write_text(json.dumps(out, indent=2))
    for k, v in out["results"].items():
        b = v["at_best_f1"]
        print(f"{k:36s} AUPRC {v['auprc']:.4f} (x{v['auprc_lift_over_random']}) "
              f"P {b['precision']:.4f} R {b['recall']:.4f} F1 {b['f1']:.4f}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
