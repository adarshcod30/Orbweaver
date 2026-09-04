# The PPA dataset, as actually released

**[← README](../README.md)** · [Why this dataset](why-this-data.md) · [Architecture](architecture.md) · [Results](results.md)

I measured the release file by file instead of trusting its paper or its
readme. Five things came out of that, and one of them changed the entire
evaluation design.

| | Finding | Consequence |
|---|---|---|
| **A** | The released graph covers the **second week only** | The evaluation lives inside week 2 |
| **B** | The release is **57.4% of the users** the paper reports | Every number here is against what is downloadable, never the paper's larger set |
| **C** | **Three of eight relations are empty** in the order files — including one worth 38.41% of the authors' edges | Two graph views are reported: what I can rebuild, and what they shipped |
| **D** | **CRLF makes the last column lie** — would have fabricated ~14 billion phantom edges | Caught by a test that now pins it |
| **E** | **The two order files are separately re-indexed from zero** | **The single most consequential finding** — see [below](#finding-e--the-two-order-files-do-not-share-an-id-space) |

Download: `make download` fetches all 4.00 GB from
[OSF](https://osf.io/rasje/?view_only=671050154acf4c0fa6b86a9337e74c2c) and
md5-checks every file.

## Files

| File | Bytes | Rows | What it is |
|---|---:|---:|---|
| `node.csv` | 142,373,615 | 3,267,961 | Users, labels, placeholder feature matrix |
| `edge.csv` | 675,362,394 | 10,012,449 | The authors' pre-built user–user graph |
| `Transactions/order_train.csv` | 1,625,195,771 | 22,456,547 | Week-1 raw orders |
| `Transactions/order_test.csv` | 1,559,383,792 | 21,478,704 | Week-2 raw orders |

## A — the graph covers week 2 only

`order_test.csv` contains 3,267,961 distinct users with ids `0…3,267,960`, and
`node.csv` has 3,267,961 rows with contiguous ids `0…3,267,960`. **Same user
set.** The shipped graph is the week-2 slice.

| | Week 1 | Week 2 |
|---|---:|---:|
| Orders | 22,456,547 | 21,478,704 |
| Distinct users | 3,785,628 (`0…3,785,627`) | 3,267,961 (`0…3,267,960`) |
| Date range | `1000-05-13` → `1000-05-20` | `1000-05-20` → `1000-05-28` |

`1000-05-20` is the only shared date — 2,935,107 orders in week 1 and **75** in
week 2 (0.000349% of it). Week 1 is cut `<= 1000-05-20`, week 2 `>= 1000-05-21`.

*(Timestamps are year-shifted to 1000, so `pandas.to_datetime` cannot represent
them — and they are day-resolution, which is why nothing in this repository
claims anything about sub-day bursts.)*

## B — the release is 57% of the paper's size

| | Paper | Release |
|---|---:|---:|
| Users | 5,693,351 | **3,267,961** (57.4%) |
| Edges | ~29,000,000 | **10,012,449** (34.5%) |
| Fraud-labelled | ~95,700 (1.68%) | **68,533** (2.097%) |
| Normal-labelled | ~405,000 (7.12%) | **237,084** (7.255%) |
| Unlabelled | rest | **2,962,344** (90.648%) |

Not a complaint — the fraud *rate* is actually higher in the release (2.10% vs
1.68%). But nothing in this project describes itself as running on 5.7M
accounts. It runs on 3.27M users and 10.0M edges.

## C — three of eight relations cannot be rebuilt

`r2`, `r4` and `r5` have **zero** non-null values across all 43.9M orders —
yet `r4` alone is 38.41% of the authors' `edge.csv`.

| Rel | Meaning | Present in orders | Share of `edge.csv` |
|---|---|---|---:|
| r1 | order-location geohash | 99.15% / 99.26% | 5.52% |
| r2 | shared links | **empty** | 0.82% |
| r3 | delivery info | 99.99% / 99.99% | 0.04% |
| r4 | retail store | **empty** | **38.41%** |
| r5 | group id | **empty** | 0.94% |
| r6 | promotion id | 48.85% / 47.30% | **73.95%** |
| r7 | coupon type | 99.99% / 99.99% | 8.55% |
| r8 | stimulation id | 99.24% / 99.13% | 5.82% |

So the pipeline reports **two views**: the five-relation graph I can rebuild
from orders, and the authors' eight-relation `edge.csv` as shipped. What the
missing three are worth is then a *measured* number, not a guess
([results.md](results.md)).

## Finding E — the two order files do not share an id space

> **Week-1 user 12,345 and week-2 user 12,345 are different people, and no key
> joins them.**

I found this the hard way: a scorer that trained cleanly and learned exactly
nothing, because it joined week-1 behaviour to week-2 labels through an id that
does not carry across ([the full account](../FAILURES.md)). Three checks
establish it:

1. **Both id spaces are perfectly dense.** Week 1 covers exactly `0…3,785,627`;
   week 2 covers exactly `0…3,267,960`. Not one gap in either. A smaller set
   drawn from a shared population would be *sparse*, not contiguous — both
   files were renumbered from zero.
2. **Behaviour does not correlate across the boundary.** For the same id,
   week-1 vs week-2 `n_orders` correlates at **+0.0101**, `degree` at +0.0013,
   `core_number` at +0.0008 — indistinguishable from independent.
3. **Week 2 is internally consistent.** The graph built from `order_test.csv`
   shares 3,079,704 pairs with the shipped `edge.csv` (30.8%, far above
   chance). Orders, graph and labels agree. Only week 1 stands apart.

**What follows.** Labels live in the week-2 id space, so a week-1-train /
week-2-test split is **not possible** on this release — the obstacle is the id
column, not the calendar. Week 2 carries the evaluation, split account-disjoint
*and* temporally forward within itself: train on `05-21…05-24`, score held-out
accounts on `05-25…05-28`. Both properties the original design wanted, using
the only data that can support them.

## D — CRLF makes the last column lie

Every line ends `\r\n`, and `r8` is the last column. When `r8` is empty the
final field is not `""` but `"\r"`. My first pass reported r8 present on
**100.00%** of orders. It is **99.24%**.

Why this matters more than it sounds: the graph builder connects every pair of
users sharing an entity. Had the loader kept the `\r`, every order with no r8
would have shared one phantom entity — and the builder would have emitted a
clique across all of them. **Roughly 14 billion fabricated edges from one
unstripped byte**, every one looking like dense, suspicious, ring-shaped
structure. Worse, the entity cap would have silently dropped the phantom entity
as "too common", so the bug would have vanished from the output without ever
being fixed. `tests/test_schema.py` now pins r8 between 98% and 99.5%.

## Entity sizes — why capping *is* the algorithm

| Rel | Distinct entities | Largest entity | Uncapped pairs | **@cap 100** |
|---|---:|---:|---:|---:|
| r1 | 2,023,601 | **862,317** | 371,808,329,847 | 11,460,780 |
| r3 | 3,274,955 | **7** | 12,122 | 12,122 |
| r6 | 390,153 | 7,421 | 3,193,225,490 | 48,405,948 |
| r7 | 37,030 | **3,187,247** | 5,086,273,447,218 | 2,248,983 |
| r8 | 10,496,292 | 1,618 | 36,071,829 | 9,682,878 |
| **Total** | | | **5,461,311,086,506** | **71,810,711** |

1. **Uncapped is not slow — it is impossible.** 5.46 *trillion* pairs on week 2.
   `r7` alone is 5.09 trillion of them, because one coupon type is shared by
   3,187,247 users: **97.5% of the entire user base**. A relation connecting
   97.5% of users is not evidence of anything.
2. **A cap of 500 was ~8× too loose** — 568M pairs, which does not fit in 16 GB.
   **100** leaves 71.8M and is the configured default, in `config/default.yaml`.
3. **Relations are wildly unequal as evidence.** `r3` never exceeds seven users
   per entity — close to a private key, so two users sharing one is strong
   evidence (and it contributes just 12,122 pairs total). `r7` is near-universal
   and nearly worthless. The rarity weight `1/log(2+|users(e)|)` orders them
   correctly without hand-ranking.

**Two entities look like sentinels, not real values** — the largest `r1` holds
26.4% of everyone and the largest `r7` holds 97.5%. Almost certainly "unknown"
placeholders. The cap removes them as a side effect, but they are excluded
explicitly too, so the reason lives in the code rather than in where a
threshold happened to land.

## What this means downstream

1. Week 2 carries the evaluation (finding E). Account-disjoint, temporally forward within the week.
2. **Every node feature is engineered from orders** — the dataset ships none (every `node.csv` feature column is literally 1.0).
3. Two graph views are always reported, which is what makes the three unbuildable relations a measurement rather than a guess.
4. Labels are static, user-level, week-2 only. `-1` means unknown, is excluded from metrics, and is **never** counted as normal.
5. Day-resolution timestamps — so no hour-of-day features and no sub-day burst claims anywhere.

---

**[← README](../README.md)** · Next: [why this dataset at all →](why-this-data.md)
