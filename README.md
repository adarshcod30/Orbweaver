# Orbweaver

*Orbweaver — the one that feels the whole web.*

Detecting coordinated promotion-abuse rings in transaction graphs.

A food-delivery app gives ₹100 off your first order. A group runs fifty accounts between them — a few people, a few phones, a stack of SIMs registered in other people's names — and takes ₹5,000. Every order looks normal on its own; the fraud only exists in the connections between accounts: a shared delivery address, a shared device, the same UPI ID paying for all of them, orders placed minutes apart. A system that scores one transaction at a time cannot see it. Orbweaver looks at the web instead of the strand.

## What it does

- Builds a multi-relation graph of users from the entities they share, with each edge weighted by how rare the shared entity is
- Scores accounts with a gradient-boosted model over transaction and graph-neighbourhood features
- Extracts rings with a score-weighted densest-subgraph algorithm — greedy peeling with a provable approximation bound
- Attaches evidence to every ring: what the members share, when they acted, the rupees at stake, and the cost of acting if the ring is not what it looks like
- Evaluates on real labelled data with a strict temporal split, and reports every detection number next to its false-positive cost

## Status

Early. Data loading, graph construction and the ring extractor are in place; scoring and evaluation are next. Results will appear here as they are produced by `make reproduce` — nothing in this file is typed in by hand.

## Data

PPA, the Public Promotion Abuse dataset released with PromoGuardian (IEEE S&P 2026). The paper describes 5.69 million users and 29 million edges, but the public release is the second week only: **3,267,961 users and 10,012,449 edges**, with 68,533 accounts labelled fraud and 237,084 labelled normal. Three of the eight relation types have no values in the released order files at all. I measured all of this from the files rather than taking the paper's figures, and every number I report is against the released data.

The data is not in this repository. `docs/data.md` explains how to obtain it, what is actually in the files, and the four ways it differs from its own documentation.

## Context

Built for the Razorpay AI Buildathon, Track 02.
