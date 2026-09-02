"""Fragmentation: what happens when a ring deliberately breaks itself up.

The PromoGuardian authors name this as the obvious next move for an attacker:
*"fraudsters may fragment groups to conceal the cohesion patterns."* If a
fifty-account ring splits into ten cells of five that share nothing across
cells, the density that makes it findable is gone.

So the useful question is not "does this survive fragmentation" — it does not,
nothing does — but **at what cell size does it stop working**. That number is
the honest boundary of the method, and it is what a team deciding whether to
deploy this would actually want to know.

The protocol, following the multi-round adversarial setup in SNAM 2025:

1. Take the ground-truth groups: connected components of the subgraph induced
   by fraud-labelled accounts.
2. Split each into cells of size `c`, assigning members round-robin so cells
   are balanced.
3. **Delete every edge that crosses two cells.** The accounts remain and their
   behaviour is untouched; only the cohesion between cells is severed.
4. Re-extract rings on the damaged graph and measure recall of the fragmented
   accounts.

This is defence-only. It operates on a public labelled dataset to measure a
detector's degradation, and produces nothing usable against a real system.
"""
from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.rings.peel import EdgeList

FRAUD = 1


def fraud_components(labels: np.ndarray, src: np.ndarray, dst: np.ndarray,
                     min_size: int = 5) -> list[np.ndarray]:
    """Ground-truth groups: connected components of fraud-labelled accounts."""
    import igraph as ig

    fraud = labels == FRAUD
    m = fraud[src] & fraud[dst]
    fs, fd = src[m], dst[m]
    if fs.size == 0:
        return []
    nodes = np.unique(np.concatenate([fs, fd]))
    ei = np.stack([np.searchsorted(nodes, fs), np.searchsorted(nodes, fd)], axis=1)
    g = ig.Graph(n=nodes.size, edges=ei)
    out = []
    for comp in g.connected_components():
        if len(comp) >= min_size:
            out.append(nodes[np.array(comp, dtype=np.int64)])
    return out


def fragment_graph(edges: EdgeList, components: list[np.ndarray],
                   cell_size: int, seed: int) -> tuple[EdgeList, np.ndarray]:
    """Split each component into cells and cut every edge between cells.

    Returns the damaged graph and the accounts that were fragmented.
    """
    rng = np.random.default_rng(seed)
    cell_of = np.full(edges.n_nodes, -1, dtype=np.int64)
    touched = []
    next_cell = 0
    for comp in components:
        if comp.size <= cell_size:
            continue                     # already smaller than the target cell
        members = comp.copy()
        rng.shuffle(members)
        n_cells = int(np.ceil(members.size / cell_size))
        # Round-robin keeps cells balanced; contiguous chunks would leave a
        # ragged last cell that is easier to detect than the rest.
        for i, acct in enumerate(members):
            cell_of[acct] = next_cell + (i % n_cells)
        next_cell += n_cells
        touched.append(members)

    if not touched:
        return edges, np.empty(0, dtype=np.int64)

    fragmented = np.concatenate(touched)
    cs, cd = cell_of[edges.src], cell_of[edges.dst]
    # Cut an edge only when both ends are inside the fragmented population and
    # they landed in different cells. Edges to the rest of the graph survive.
    cross = (cs >= 0) & (cd >= 0) & (cs != cd)
    keep = ~cross
    return (EdgeList(edges.src[keep], edges.dst[keep], edges.weight[keep],
                     edges.n_nodes),
            fragmented)


def run_fragmentation(cfg: Config | None = None,
                      cell_sizes: tuple[int, ...] = (3, 5, 10, 20)) -> dict:
    from eval.run_rings import load_edges, prune
    from eval.split import make_split
    from orbweaver.rings.peel import extract_rings_batch

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    n = split.labels.size
    labels = split.labels

    scores = np.zeros(n)
    s = pq.read_table(proc / "scores_week2.parquet")
    scores[s["user_id"].to_numpy()] = s["score"].to_numpy()

    report = json.loads((proc / "ring_report.json").read_text())
    best = report.get("best_cell", {"tau": 0.5, "lambda": 5.0})
    top_k = report["graph"]["top_k"]

    full = load_edges("late", cfg, n)
    comps = fraud_components(labels, full.src, full.dst)
    print(f"ground-truth fraud groups (>=5 accounts): {len(comps)}, "
          f"covering {sum(c.size for c in comps):,} accounts")

    def measure(edges: EdgeList, population: np.ndarray) -> dict:
        sub = prune(edges, scores, best["tau"])
        rings = extract_rings_batch(sub, scores, lambda_=best["lambda"],
                                    k_min=cfg.rings.k_min, k_max=cfg.rings.k_max,
                                    top_k=top_k, g_min=cfg.rings.g_min)
        caught = np.zeros(n, dtype=bool)
        for r in rings:
            caught[r.members] = True
        members = np.unique(np.concatenate([r.members for r in rings])) if rings \
            else np.empty(0, dtype=np.int64)
        lab = labels[members]
        known = int((lab == FRAUD).sum() + (lab == 0).sum())
        return {
            "n_rings": len(rings),
            "accounts_in_rings": int(members.size),
            "ring_precision": round(float((lab == FRAUD).sum() / known), 4) if known else None,
            "recall_of_population": round(float(caught[population].mean()), 4)
            if population.size else None,
            "population": int(population.size),
        }

    baseline_pop = np.concatenate(comps) if comps else np.empty(0, dtype=np.int64)
    rows = {"intact": measure(full, baseline_pop)}
    print(f"  intact          recall of grouped fraudsters "
          f"{rows['intact']['recall_of_population']}  "
          f"precision {rows['intact']['ring_precision']}")

    for c in cell_sizes:
        damaged, frag = fragment_graph(full, comps, c, cfg.seed)
        cut = full.src.size - damaged.src.size
        r = measure(damaged, frag if frag.size else baseline_pop)
        r["cell_size"] = c
        r["edges_cut"] = int(cut)
        r["accounts_fragmented"] = int(frag.size)
        rows[f"cells_of_{c}"] = r
        print(f"  cells of {c:<3}    recall {r['recall_of_population']}  "
              f"precision {r['ring_precision']}  ({cut:,} edges cut)")

    return {
        "operating_point": best,
        "ground_truth_groups": len(comps),
        "accounts_in_groups": int(sum(c.size for c in comps)),
        "results": rows,
        "protocol": ("Ground-truth groups are connected components of the "
                     "fraud-labelled subgraph. Each is split into balanced cells "
                     "of the given size and every edge crossing two cells is "
                     "deleted; accounts and their behaviour are untouched."),
    }


def main() -> None:
    cfg = load_config()
    out = run_fragmentation(cfg)
    dest = cfg.abs_path(cfg.paths.processed) / "fragmentation.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
