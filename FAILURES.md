# What broke

An honest log, written as things happen rather than reconstructed at the end.
Each entry records what I believed, why it was wrong, and what fixed it.

---

## 2 September — the first version of this project was the wrong project

**What broke:** my first direction was an agent-payment mandate gate. It did
not survive my own review. The gap I was aiming at turned out to be only
partly novel, the evaluation would have rested on synthetic data I generated
myself, and the headline metric was "losses avoided" — a counterfactual with
nothing real to point at.

**What I believed:** that a narrow, well-argued gap in a fashionable area was
enough to carry a project, and that I could bolt credible evaluation onto it
later.

**What fixed it:** starting again from the literature instead of from the
idea. I found PromoGuardian (IEEE S&P 2026), which released a real labelled
promotion-abuse dataset and, in its own limitations section, named three open
problems. I rebuilt the project around data I did not make up, a formal core
(densest subgraph — NP-hard, with a provable greedy bound), and an output a
human can actually act on.

**What I take from it:** pick the problem that has a dataset, not the problem
that has the best story. Evaluation is not something you add at the end.

---

## 2 September — a column that was full of values it did not have

**What broke:** my first pass over `order_train.csv` reported that relation
`r8` was present on **100.00 %** of all 22,456,547 orders. Every single one.

**What I believed:** that `r8` (the stimulation id) was a mandatory field the
platform stamps on every order. That is a perfectly plausible story for a
sales-strategy tag, and I was about to write it into the schema notes as a
fact about the data.

One thing bothered me. The sample rows I had printed a minute earlier ended in
a bare comma — `...,,,,76260.0,` — which is a visibly *empty* `r8` on a row my
counter had just called non-empty. Both could not be true.

**What was actually happening:** the files use CRLF line endings. `r8` is the
last column, so on a row where `r8` is empty the final field is not `""` but
`"\r"`. My test was `if ($i != "")`, and `"\r"` is not `""`. Every empty `r8`
in the file counted as present.

**What fixed it:** `od -c` on the first two lines, which showed `,  \r  \n`
ending the record. Re-running with `sub(/\r$/, "")` gives the true figure:
**99.24 %**, not 100 %.

**Why it mattered much more than 0.76 %:** the graph builder draws an edge
between every pair of users who share an entity. If the loader had kept the
`\r`, then all ~170,000 orders with no `r8` would have "shared" one phantom
entity — the string `"\r"` — and the builder would have emitted a clique
across every one of them. That is on the order of 14 billion fabricated edges
from a single unstripped byte, and every one of them would have looked like
exactly the dense, suspicious structure I am trying to find. The worst part is
how it would have ended: my entity cap would have thrown the phantom entity
away for being too common, so the bug would have vanished from the output
without ever being fixed, and I would have trusted the result.

`tests/test_schema.py` now pins r8 between 98 % and 99.5 % so this cannot
come back quietly.

**What I take from it:** a suspiciously round number is a bug until proven
otherwise. 100.00 % is not a measurement, it is a smell. And what caught it
was reading two lines of raw bytes — not any amount of thinking about what the
column was supposed to mean.

---

## 2 September — the dataset is not the dataset in the paper

**What broke:** I had planned the whole pipeline around the figures in the
PromoGuardian paper: 5,693,351 users, ~29 M edges, eight relation types. My
memory budget, my sampling strategy and the README headline all assumed them.

**What I believed:** that the public OSF release is the dataset the paper
describes. It is the obvious assumption and I never questioned it.

**What it actually is:** the release is the **test week only** —
**3,267,961 users** (57 %) and **10,012,449 edges** (35 %). The dataset's own
`readme.md` says so, in one sentence that is very easy to skim past: *"This
graph is obtained by filtering the original orders of the test set."* I
confirmed it independently — the user id set of `order_test.csv` is exactly
the id set of `node.csv`, contiguous `0…3,267,960`.

Then a worse one. **`r2`, `r4` and `r5` have zero values across all 43.9 M
orders in both files**, yet `r4` is 38.4 % of the edges in the shipped
`edge.csv`. Three of the eight relations cannot be rebuilt from the released
data at all, and none of them can have a week-1 counterpart.

**What fixed it:** two changes, neither of which cost me anything in the end.
First, the temporal split moved off `node.csv`/`edge.csv` — which are a week-2
object with no split in them — and onto the two order files, which is the only
place `order_time` exists. Second, the pipeline now runs over two graph views:
the one I build from week-2 orders with five relations and my own rarity
weights, which has a real week-1 counterpart; and the authors' shipped
`edge.csv` with all eight. The gap between them is a measurement of what the
three missing relations are worth, which happens to be a limitation the
authors state themselves: *"in cases where key relations are missing,
detection performance may degrade."*

**What I take from it:** the paper describes the data the authors had. The
release is what they were able to publish. Read the release's own readme
before trusting the paper's numbers — and then check the readme against the
files, because this one was also wrong about the column names (it says
`r1`…`r8`; the files and the authors' own code use `r1_score`…`r8_score`).

---

## 2 September — my entity cap was eight times too loose

**What broke:** nothing, and only because I measured before building. This is
the short entry it is because of the order I did things in.

**What I believed:** that capping shared entities was a tidiness measure — a
guard against a handful of pathological values. I had picked 500 as a starting
point without much thought.

**What it actually is:** on week-2 orders the uncapped user-pair count is
**5,461,311,086,506**. Five and a half *trillion* pairs. One coupon type is
shared by **3,187,247 users — 97.5 % of the entire user base** — and
contributes 5.09 trillion pairs on its own. At my starting cap of 500 the
graph is still **568,161,582** pairs, which does not fit in the 16 GB I have.

**What fixed it:** sweeping the cap before writing the builder rather than
after. A cap of **100** gives **71,810,711** pairs on week 2 and 68,170,327 on
week 1 — tractable, and small enough that iteration is fast. The cap is a
config value and I report how sensitive results are to it, so the choice is
visible rather than baked in.

**What I take from it:** "cap huge entities" was already my plan, but as a
principle. Turning it into a measured table changed the number by a factor of
five, and the measurement took four minutes.

---

## 2 September — I trained a model on 183,370 strangers

**What broke:** my first scorer came out with an AUPRC of 0.2338 against a base
rate of 0.2242. Precision, recall and F1 were all exactly 0.0000. The model had
learned nothing at all.

**What I believed:** that I had a modelling problem. Wrong class weights, too
few trees, features too weak, early stopping firing at iteration 116. I was
about to start tuning hyperparameters.

The thing that stopped me was printing per-feature means for fraud against
normal. Every one of the 27 features had a ratio of almost exactly 1.000 —
`n_orders` 5.9739 against 5.9980, `degree` 34.19 against 34.50, and so on down
the list. Twenty-seven features cannot all be identical to three decimal places
by accident. That is not a weak signal, that is no relationship whatsoever,
and no amount of tuning fixes a table that looks like that.

**What it actually is: the two order files use different user id spaces.**
`order_train.csv` and `order_test.csv` are each independently re-indexed from
zero. Week-1 user 12,345 and week-2 user 12,345 are two different people. So
I had been joining week-1 behaviour to week-2 labels through an id that means
nothing across the boundary, and training on pure noise. It ran, it converged,
it produced a number.

Three checks, each independently conclusive:

1. **Both id spaces are perfectly dense.** Week 1 has 3,785,628 distinct ids
   covering exactly `0…3,785,627`, and week 2 has 3,267,961 covering exactly
   `0…3,267,960`. If week 2 were a subset of a shared id space, its ids would
   have gaps — a smaller set of users drawn from a larger population is not
   contiguous. Neither has a single gap. Both were renumbered from zero.
2. **Behaviour does not correlate across the boundary.** For the same id,
   week-1 against week-2 `n_orders` correlates at **+0.0101**; `degree` at
   +0.0013; `core_number` at +0.0008. Heavy users stay heavy — unless they are
   not the same users.
3. **Week-2 ids are internally consistent.** The graph I build from
   `order_test.csv` shares 3,079,704 pairs with the authors' shipped
   `edge.csv`, 30.8 % of theirs, which is far from chance. So `order_test.csv`,
   `node.csv` and `edge.csv` agree with each other. It is only week 1 that is
   in its own world.

**What this costs:** the split I had planned — and had already written tests
for — is impossible. Labels live in the week-2 id space, so a labelled
account cannot be located in week 1, so week-1 features for a labelled account
do not exist. There is no key to join on, and none can be reconstructed.

**A second thing this quietly invalidated.** An hour earlier I had recorded
that "every labelled week-2 account was also active in week 1" and treated it
as a finding about user retention that made memorisation a risk. It was an
artefact of this same bug: every week-2 id in `0…3,267,960` trivially falls
inside week 1's dense range `0…3,785,627`, so the intersection was guaranteed
to be total no matter what the data said. I had measured the id ranges, not
the users. That "finding" is withdrawn.

**How I got out:** week 2 is internally consistent — orders, graph, and labels
all share one id space — so it is the slice that can carry the whole
evaluation. The split is now **account-disjoint within week 2**, stratified by
label, so no account appears in both training and test. To keep a real
temporal guarantee rather than just asserting one, I also split week 2's eight
days in half: features and graph for training come from `05-21…05-24`, and the
held-out accounts are scored on features and a graph built from
`05-25…05-28`. That is both account-disjoint and strictly forward in time,
which is what the original week-1/week-2 design was for.

**What I take from it:** I verified that the id column was contiguous, and
wrote that down as a schema fact, and then read it as evidence that ids were
*shared*. Those are different claims and I collapsed them without noticing.
The bug was invisible at every level above the data — the loader was correct,
the split was correct, the tests passed, the training converged. Only the
per-class feature means showed it, and I only printed those because the result
was too bad to tune my way out of. A model that fails loudly is a gift; the
same bug with 60 % of the signal intact would have shipped.

---

## 2 September — the subsample quietly destroyed the rings

**What broke:** my regional subsample ran and looked healthy — 378,440 nodes,
9,671 of them labelled fraud, a reasonable-looking slice to develop against.
Then the preservation check I had written into it reported that of the 2,151
labelled-fraud components it touched, only **556 survived whole. 25.8 %.**
Three quarters of every ring in the sample had been cut in half.

**What I believed:** that anchoring on `r1` location entities would keep rings
intact, because PromoGuardian reports that abuse rings are spatially cohesive.
My rule for myself was "never sample nodes, sample entities", and this
satisfied it — it *is* sampling entities.

**What it actually is:** in this data rings are not held together by location.
`r1` contributes 11.5 M pairs to the week-2 graph; `r6`, the promotion id,
contributes **48.4 M**. Ring members share a *promotion* and scatter across
locations. So cutting on locations cuts across rings almost as effectively as
cutting on nodes would have. I had followed the letter of my own rule and
missed its entire purpose.

**What fixed it, partly:** adding a 1-hop closure — after seeding on location
entities, pull in every graph neighbour. Rings are dense, so in a near-clique
every member is one hop from any other and seeding one member should retrieve
the rest. Preservation went from 25.8 % to 44.9 %, but the sample grew from
378 k to 1.18 M nodes, which is 66 % of the full graph's edges and no longer a
fast sample. Tuned down to 571,849 nodes, preservation sits at 33.4 %.

The closure deliberately does not look at labels. Selecting nodes by fraud
label would have leaked the exact structure the sample is used to measure.

**Where it actually landed:** no local sample of this graph is ring-faithful.
Mean degree is around 49, so any cut either splits rings or expands to most of
the graph. So I demoted the subsample. It is a development-speed convenience,
its 33.4 % preservation is recorded in its own manifest, and **every ring
metric I report comes from the full week-2 graph** — 2,824,697 nodes and
69,260,001 edges. Peeling is near-linear, so I can afford the full graph. The
sample was never supposed to be load-bearing; it was only supposed to be
faster.

**What I take from it:** a rule written as a mechanism ("don't sample nodes")
can be obeyed exactly while the thing it exists to protect ("don't destroy
rings") is destroyed anyway. The only reason I caught this is that I had made
`ring_preservation` an *output* of the sampler instead of an assumption.
Measure the property the rule is protecting, not compliance with the rule.
