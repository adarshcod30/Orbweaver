# Where machine learning is used, and where it deliberately is not

The pipeline has exactly one learned component, and the boundary around it is
the most deliberate design decision in this project.

## The split

| Stage | How it works | Learned? |
|---|---|---|
| Graph construction | Two accounts are linked when they share an entity; the edge weight is `1/log(2 + users(e))` | No — arithmetic |
| Account scoring | XGBoost over 39 transaction and graph features, isotonic-calibrated | **Yes** |
| Ring extraction | Greedy peeling on a density objective, with a proved approximation bound | No — deterministic |
| Evidence | Shared entities, coverage, rarity, day concentration, counted from the orders | No — counting |
| Rupees at stake and false-positive cost | Arithmetic over stated assumptions | No |
| The decision to act | A human reads the case file | No |

**A model scores accounts. It does not decide who is in a ring.** The ring is
whatever the peeling objective returns, and that objective is forty lines of
arithmetic I can print on one screen.

## Why the boundary sits there

**I can answer "why is this account in this ring?"** The answer is a number:
removing it lowers the group's density from *x* to *y*, and it shares these
three entities with these nine members. That is checkable by hand. If ring
membership came out of a learned model, the honest answer would be "the model
put it there", and an analyst deciding whether to freeze someone's account
deserves better.

**There is an approximation guarantee, and it means something.** Greedy
peeling returns a set within a factor of 2 of the densest possible one
(Charikar 2000; Hooi et al., FRAUDAR, KDD 2016), and the batch variant used at
scale is within 2(1+ε). No learned model over this objective could offer a
comparable statement. The guarantee is about the *search*, not the data, which
is exactly the part I want to be certain about.

**It degrades in a way I can predict.** When the scorer is weak, ring quality
falls back toward pure structure rather than collapsing — `λ = 0` is a valid
operating mode that uses no model output at all. On this dataset that matters:
the FRAUDAR baseline in the PromoGuardian paper (F1 0.4715, structure only)
beats their GraphSAGE baseline (F1 0.2810, learned node features), which is a
strong hint that on promotion-abuse graphs the structure carries more signal
than the node features do.

**λ makes the trade-off explicit rather than hidden.** The objective is
`(Σ edge weight + λ · Σ score) / |S|`. At `λ = 0` it is pure structure; as λ
grows the model's opinion dominates. I sweep λ and report the whole curve
instead of tuning to a single flattering value, so a reader can see exactly
how much the learned component is contributing.

## Why XGBoost and not a graph neural network

GADBench (NeurIPS 2023) found that gradient boosting over graph-aggregated
features is a top performer on real anomaly-detection graphs and often beats
GNNs outright. It trains in about a minute on a laptop, its feature
importances are readable, and — most importantly — the ring extraction
downstream is scorer-agnostic, so a better scorer can be dropped in later
without touching the component that produces the output.

A GNN is worth trying as an alternative scorer, and I report its numbers
beside XGBoost's whichever way they fall. It is not on the critical path.

**The GNN is trained in mini-batches, and it has to be.** A full-batch layer
is not slow here, it is impossible: `SAGEConv` materialises one hidden vector
per directed edge before aggregating, which on 71.4 million directed edges at
64 hidden dimensions is **18.3 GB for a single layer**, 36.6 GB for two, and
about 73 GB once autograd retains them for the backward pass. The machine has
16 GB, so one layer alone exceeds the whole thing.

Neighbour sampling bounds the batch instead of the graph. At fanout
`[15, 10]` with 1024 seed accounts, a batch reaches at most ~170,000 edges —
roughly **43 MB of messages** — and that figure does not change if the graph
gets ten times larger. Measured peak resident memory for the whole training
run, including the feature matrix and adjacency held in full, is **3.56 GB**,
and it trains in about 95 seconds on the M4's GPU.

The sampler is written by hand against a CSR adjacency
(`orbweaver/scoring/sampler.py`, about forty lines) because PyTorch
Geometric's own `NeighborLoader` requires `pyg-lib` or `torch-sparse`, and
neither publishes a wheel for Python 3.13 on arm64 — the same wall that
stopped the authors' checkpoint.

## No language model anywhere in the detection or decision path

No large language model scores an account, chooses a ring member, sets a
threshold, or decides anything else that affects an output.

The reason is not distaste for the technology. It is that this component has
to answer three questions that a language model cannot answer well: *what
guarantee does this give?*, *why exactly is this account here?*, and *what
happens at the margin if I change this input slightly?* Densest-subgraph
peeling answers all three. It also runs in seconds on 35 million edges, costs
nothing per call, and gives the same answer every time — none of which is true
of a model behind an API.

There is one place a language model would genuinely help: writing a finished
case file up in plain English for whoever reads it. That is read-only, strictly
downstream of every decision, and entirely skippable. It is not implemented,
and if it were, removing it would change no number in this repository.
