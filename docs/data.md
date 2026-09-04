# The PPA dataset, as actually released

## In one minute

What is actually on disk is **3,267,961 users and 10,012,449 edges** — the
released week-2 slice — against the **5,693,351 users and ~29,000,000 edges**
the paper describes (finding B).

The single most consequential thing on this page is
[finding E](#finding-e--the-two-order-files-are-separately-re-indexed-from-zero):
the two order files are independently re-indexed from zero, so
`order_train.csv` and `order_test.csv` do not share a user id space, which
rules out joining week-1 behaviour to week-2 labels by id at all.

Three of the eight relations, r2, r4 and r5, are entirely absent from both
order files, so they cannot be rebuilt from raw orders even though r4 alone
accounts for 38.41 % of the edges in the authors' `edge.csv`.

---

The data comes from the PromoGuardian authors' OSF project at
`https://osf.io/rasje/?view_only=671050154acf4c0fa6b86a9337e74c2c`; running
`make download` fetches all 4.00 GB of it and checks every file against its
published md5, and `make schema` then runs `scripts/inspect_ppa.py` over the
raw files to produce `data/ppa_schema_facts.json`. Every number on this page
comes out of that JSON — I have not typed any of them in by hand.

I wrote this page because the dataset's own `readme.md` is short and, in two
places, wrong, and because the paper describes a larger dataset than the one
you can actually download. Five differences changed how I built the pipeline,
and they are marked as findings below. **Finding E is the one that matters
most** — the two order files use different user id spaces, which rules out the
train/test split the paper's protocol implies.

---

## Files

| File | Bytes | Rows | What it is |
|---|---:|---:|---|
| `readme.md` | 990 | — | The authors' four-paragraph description |
| `node.csv` | 142,373,615 | 3,267,961 | Users, labels, and a placeholder feature matrix |
| `edge.csv` | 675,362,394 | 10,012,449 | The authors' pre-built user–user graph |
| `Transactions/order_train.csv` | 1,625,195,771 | 22,456,547 | Week-1 raw orders |
| `Transactions/order_test.csv` | 1,559,383,792 | 21,478,704 | Week-2 raw orders |

All five files use CRLF (`\r\n`) line endings, which matters more than it
sounds — see finding D.

---

## Finding A — the released graph covers the second week only

The dataset's `readme.md` says it plainly, in a sentence that is easy to skim
past: *"This graph is obtained by filtering the original orders of the test
set, and the original orders are stored in the folder named `Transactions`."*

It checks out exactly. `order_test.csv` contains 3,267,961 distinct users with
ids `0…3,267,960`, and `node.csv` has 3,267,961 rows with contiguous ids
`0…3,267,960`. Same user set.

So `node.csv` and `edge.csv` are not a two-week graph waiting to be split.
They are a week-2 object, and there is no split inside them. The temporal
split has to be built from the two order files, because `order_time` only
exists there.

| | Week 1 (`order_train.csv`) | Week 2 (`order_test.csv`) |
|---|---|---|
| Orders | 22,456,547 | 21,478,704 |
| Distinct users | 3,785,628 (ids 0…3,785,627) | 3,267,961 (ids 0…3,267,960) |
| Date range | `1000-05-13` → `1000-05-20` | `1000-05-20` → `1000-05-28` |
| Full days | 05-14 … 05-20 (7) | 05-21 … 05-28 (8) |
| Boundary day | 05-13: 291 orders | 05-20: 75 orders |

`1000-05-20` is the only date that appears in both files — 2,935,107 orders in
week 1 and **75** in week 2, which is 0.000349 % of week 2. The 291 orders on
`1000-05-13` are a similar leading fragment in week 1.

I split on `order_time`, not on which file a row came from. Week 1 is
`<= 1000-05-20`, week 2 is `>= 1000-05-21`, and those 75 boundary orders are
dropped rather than reassigned. If I had used the filenames as the split, the
75 rows would have been a leak in the less obvious direction — test-file rows
sitting in training data. `tests/test_split_no_leak.py` asserts the two date
sets are disjoint and that every raw row is either in a week or explicitly
dropped.

**How I use the two graphs.** Week-1 orders build training features; week-2
orders build the graph I extract rings from. The authors' `edge.csv` is a
legitimate week-2 object, so using it at extraction time is not leakage — but
it cannot be used for training features, and it cannot be rebuilt (finding C).

## Finding B — the release is 57 % of the size the paper reports

| Quantity | Reported in the paper | Actually released |
|---|---:|---:|
| Users | 5,693,351 | **3,267,961** (57.4 %) |
| Edges | ~29,000,000 | **10,012,449** (34.5 %) |
| Fraud-labelled | ~95,700 (1.68 %) | **68,533 (2.097 %)** |
| Normal-labelled | ~405,000 (7.12 %) | **237,084 (7.255 %)** |
| Unlabelled | rest | **2,962,344 (90.648 %)** |

This follows from finding A: the paper's figures describe the full two-week
internal dataset, and the public release is the test-week slice of it. The
fraud *rate* is actually higher in the release, 2.10 % against 1.68 %.

Every number I report is therefore "on the released PPA week-2 graph,
3.27 M users and 10.0 M edges". I never describe this work as running on
5.7 M users, because that is not what is on disk.

## Finding C — three of the eight relations cannot be rebuilt

`r2`, `r4` and `r5` have **zero** non-null values across all 43.9 M orders in
both files. But they are present in `edge.csv`, and `r4` is the second densest
relation there at 38.4 % of all edges.

| Relation | Meaning | In the order files | In `edge.csv` |
|---|---|---|---|
| r1 | order-location geohash | 99.15 % / 99.26 % | 5.52 % of edges |
| r2 | shared links | **empty** | 0.82 % |
| r3 | delivery info | 99.99 % / 99.99 % | 0.04 % |
| r4 | retail store | **empty** | **38.41 %** |
| r5 | group id | **empty** | 0.94 % |
| r6 | promotion id | 48.85 % / 47.30 % | **73.95 %** |
| r7 | coupon type | 99.99 % / 99.99 % | 8.55 % |
| r8 | stimulation id | 99.24 % / 99.13 % | 5.82 % |

(week 1 % / week 2 %.)

A graph built from the orders can carry **five** relations, not eight. I cannot
reproduce `edge.csv`, and there can be no week-1 version of r2, r4 or r5 at all.

I decided this is worth measuring rather than working around, because it is a
direct instance of a limitation the PromoGuardian authors state themselves:
*"In cases where key relations are missing, detection performance may
degrade."* So the pipeline runs over two graph views:

- **View A, the one I build:** from week-2 orders, five relations, my own
  entity-rarity weights, entities capped. It has a matching week-1 graph, so
  features are honestly temporal.
- **View B, the authors' `edge.csv`:** eight relations, their weights, week 2
  only, no week-1 counterpart.

The gap between A and B is a measurement of what those three missing relations
are worth. It also keeps me honest: my numbers come from a strictly smaller
relation set than the paper's.

## Finding E — the two order files are separately re-indexed from zero

**`order_train.csv` and `order_test.csv` do not share a user id space.**
Week-1 user 12,345 and week-2 user 12,345 are different people, and there is
no key that joins them.

I found this the hard way — a scorer that trained cleanly and learned exactly
nothing, because it was joining week-1 behaviour to week-2 labels through an
id that does not carry across. `FAILURES.md` has the full account. Three
checks establish it:

1. **Both id spaces are perfectly dense.** Week 1 holds 3,785,628 distinct ids
   covering exactly `0…3,785,627`; week 2 holds 3,267,961 covering exactly
   `0…3,267,960`. Neither has a single gap. A smaller set of users drawn from a
   shared larger population would be sparse, not contiguous. Both files were
   renumbered from zero.
2. **Behaviour does not correlate across the boundary.** For the same id,
   week-1 against week-2 `n_orders` correlates at **+0.0101**, `degree` at
   +0.0013, `core_number` at +0.0008 — indistinguishable from independent.
3. **Week 2 is internally consistent.** The graph built from `order_test.csv`
   shares 3,079,704 pairs with the shipped `edge.csv` — 30.8 % of it, far
   above chance. `order_test.csv`, `node.csv` and `edge.csv` agree with one
   another. Only week 1 stands apart.

**What follows.** Labels live in the week-2 id space, so a labelled account
cannot be located in week 1 and week-1 features for a labelled account do not
exist. A week-1-train / week-2-test split is not possible on this release, and
no amount of care about dates changes that — the obstacle is the id column,
not the calendar.

So week 2 carries the evaluation, because it is the slice where orders, graph
and labels share one id space. The split is **account-disjoint** — stratified
by label, with held-out accounts absent from training and calibration — and
**temporally forward within week 2**: training features and graph come from
`1000-05-21…05-24`, and held-out accounts are scored on features and a graph
built from `1000-05-25…05-28`. That preserves both properties the original
design was after — no account seen twice, and nothing from the future in a
training feature — using the only data that can actually support them.

Week 1 remains useful as an unlabelled population for checking that graph
construction behaves consistently, but it cannot contribute supervision.

## Finding D — CRLF makes the last column lie

Every line ends `\r\n`, and `r8` is the last column. When `r8` is empty the
final field is not `""` but `"\r"`. My first pass over the file reported r8
present on **100.00 %** of orders. It is **99.24 %**.

This is worth stating loudly because of what it would have done downstream. The
graph builder connects every pair of users who share an entity. Had the loader
kept the `\r`, every order with no r8 would have shared one phantom entity —
the string `"\r"` — and the builder would have emitted a clique across all of
them. That is roughly 14 billion fabricated edges from one unstripped byte, and
every one of them would have looked like dense, suspicious, ring-shaped
structure. Worse, the entity cap would have silently dropped the phantom
entity as "too common", so the bug would have disappeared from the output
without ever being fixed. `tests/test_schema.py` pins r8 between 98 % and
99.5 % so this cannot come back.

---

## `node.csv`

`id, label, vec_0 … vec_7` — 10 columns, 3,267,961 rows.

- **`id`** — contiguous `0…3,267,960`, so `id` equals the row index. I index
  arrays with it directly and never remap; `tests/test_schema.py` asserts it.
- **`label`** — one of:

| Value | Meaning | Count | Share |
|---:|---|---:|---:|
| `1` | fraud | 68,533 | 2.097 % |
| `0` | normal | 237,084 | 7.255 % |
| `-1` | **unlabelled** | 2,962,344 | 90.648 % |

  `-1` means unknown, not normal. It outnumbers the two real classes about
  ten to one, so treating it as a negative silently redefines every metric. I
  compute metrics over labelled users only and say so. The authors' own
  `test.py` counts `-1` as negative, which deflates their reported precision.
- **`vec_0…vec_7`** — every value is 1.0, in all 3,267,961 rows. There are no
  node features in this dataset. The authors' `test.py` does not even read
  these columns; it builds `torch.ones(n, 52)` instead.

  So every node feature in this project is engineered from the order stream.
  That is the open problem the paper names in its future work: *"Designing
  discriminative node features remains an open challenge."*

## `edge.csv`

`src, dst, r1_score … r8_score` — 10 columns, 10,012,449 rows.

The dataset `readme.md` calls these columns `r1`…`r8`. They are actually named
`r1_score`…`r8_score` — the authors' own `edge_feature_KG.py` reads the
`_score` names. The files and their code agree; the readme is wrong.

- 10,012,449 rows are 10,012,449 distinct undirected pairs, with **no
  self-loops**. Already deduplicated, one row per pair, one direction only.
  Their `test.py` calls `dgl.add_reverse_edges`, confirming it is undirected.
- All 3,267,961 nodes appear in at least one edge. There are no isolated nodes.
- Multi-relation edges are common: 70.6 % of edges carry one relation, 25.2 %
  carry two, 3.8 % three, and 213 edges carry six. Six of eight is the maximum
  observed.

| Column | Non-zero edges | % | min | max | mean | distinct values |
|---|---:|---:|---:|---:|---:|---:|
| `r1_score` | 552,601 | 5.52 % | 0.307692 | 1.000000 | 0.6365 | 146 |
| `r2_score` | 82,254 | 0.82 % | 0.307692 | 1.000000 | 0.6999 | 113 |
| `r3_score` | 4,246 | 0.04 % | 0.317525 | 0.999994 | 0.7437 | 19 |
| `r4_score` | 3,845,316 | 38.41 % | 0.304348 | 1.000000 | 0.5417 | 209 |
| `r5_score` | 94,055 | 0.94 % | 0.317525 | 0.997527 | 0.6176 | 38 |
| `r6_score` | 7,404,702 | 73.95 % | 0.303030 | 1.000000 | 0.5564 | 175 |
| `r7_score` | 855,891 | 8.55 % | 0.317525 | 0.999877 | 0.6661 | 49 |
| `r8_score` | 582,558 | 5.82 % | 0.307692 | 0.999999 | 0.5740 | 95 |

These weights are not raw counts. They sit in roughly (0.30, 1.0] and take
very few distinct values — 19 to 209 — so they are a quantised transform of
some small-integer statistic. The value `0.7310585786300049` appears exactly,
which is `sigmoid(1)` to machine precision, so it is a logistic of a
co-occurrence count. The exact form is not documented anywhere I could find.

I do not use these weights for View A. I compute my own entity-rarity weight,
`w_r(e) = 1 / log(2 + |users(e)|)`, which I can explain and inspect and which
is derived from values I can see. The authors' weights are carried through on
View B only, and labelled as theirs.

## `Transactions/order_{train,test}.csv`

`order_time, sku_id, id, r1 … r8` — 11 columns.

- **`order_time`** — `YYYY-MM-DD`, with the year shifted to `1000`. Dates are
  anonymised but strictly ordered, and the resolution is one day; there is no
  time of day.

  Two practical consequences. First, `pandas.to_datetime` raises
  `OutOfBoundsDatetime` on these values, because `datetime64[ns]` only spans
  1677–2262 and the shift puts every timestamp about 700 years outside it — so
  I parse with `datetime.date` and store an integer day ordinal instead.
  Second, and more importantly for the results: there is no hour-of-day
  feature and no sub-day burst here. Temporal features are day-level — active
  days, busiest-day concentration, longest gap in days — and any claim about a
  ring ordering "within a few hours" is unsupportable on this data. I say
  "same day" instead, because that is the finest thing the data can show.
- **`sku_id`** — the product ordered.
- **`id`** — the user, but **the two files do not share an id space**. See
  finding E; this is the single most consequential thing on this page.
- **`r1…r8`** — the raw anonymised entity ids, written as floats with a `.0`
  suffix, empty when the order does not involve that relation. These are what
  View A is built from.

### Entity sizes, and why capping is the whole ballgame

This table is the reason the graph builder has a cap at all. It shows, per
relation, how many distinct users share one entity and how many user-pairs
that would induce. Week 2:

| Rel | Distinct entities | Largest entity | Entities >100 | Uncapped pairs | @cap 1000 | @cap 500 | **@cap 100** |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 | 2,023,601 | **862,317** | 164 | 371,808,329,847 | 13,456,761 | 13,456,761 | 11,460,780 |
| r3 | 3,274,955 | **7** | 0 | 12,122 | 12,122 | 12,122 | 12,122 |
| r6 | 390,153 | 7,421 | 19,479 | 3,193,225,490 | 1,130,435,688 | 500,633,398 | 48,405,948 |
| r7 | 37,030 | **3,187,247** | 1,587 | 5,086,273,447,218 | 104,659,070 | 34,940,780 | 2,248,983 |
| r8 | 10,496,292 | 1,618 | 508 | 36,071,829 | 25,353,943 | 19,118,521 | 9,682,878 |
| **Total** | | | | **5,461,311,086,506** | 1,273,917,584 | 568,161,582 | **71,810,711** |

Week 1 totals **7,358,575,728,954** uncapped and **68,170,327** at a cap of 100.

Three things I took from this:

1. **Uncapped is not slow, it is impossible.** 5.46 *trillion* user-pairs on
   week 2. `r7` alone accounts for 5.09 trillion of them, because one coupon
   type is shared by 3,187,247 users — 97.5 % of the entire user base. A
   relation that connects 97.5 % of users is not evidence of anything.
2. **I started at a cap of 500 and it was about eight times too loose.** It
   leaves 568 M pairs, which does not fit in the 16 GB I have. A cap of
   **100** leaves 71.8 M pairs and is the configured default. The cap lives in
   `config/default.yaml` and I report how sensitive the results are to it.
3. **The relations are wildly unequal as evidence.** `r3` never exceeds seven
   users per entity — it is close to a private key, so two users sharing one
   is strong evidence, and it contributes just 12,122 pairs in total. `r7` is
   near-universal and nearly worthless at the common end. The rarity weight
   `1/log(2+|users(e)|)` orders them correctly without my having to hand-rank
   them.

**Two entities look like sentinels rather than real values.** The largest `r1`
entity holds 862,317 users, 26.4 % of everyone, and the largest `r7` holds
97.5 %. Those are almost certainly "unknown" or default placeholders, not a
location a quarter of the user base shares. The cap removes them as a side
effect, but I exclude them explicitly as well, so the reason is in the code
rather than being an accident of where a threshold landed.

---

## What this means for the rest of the pipeline

1. Week 2 carries the evaluation, because it is the only slice where orders,
   graph and labels share an id space (finding E). The split is
   account-disjoint and temporally forward *within* week 2: train on
   `05-21…05-24`, score held-out accounts on `05-25…05-28`.
2. Every node feature is engineered from orders. The dataset ships none.
3. View A carries five relations with my rarity weights and a cap of 100.
   View B is `edge.csv` as shipped. Reporting both measures what the three
   unbuildable relations are worth.
4. Labels are static and user-level, and only cover users active in week 2.
   `-1` means unknown and is excluded from metrics, never counted as normal.
5. Timestamps are day-resolution. No hour-of-day features, and no claims about
   sub-day bursts anywhere in this repository.

---

Back to [README](../README.md).
