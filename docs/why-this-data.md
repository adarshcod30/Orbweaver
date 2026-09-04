# Why this dataset

## In one minute

PPA is the only public, labelled, promotion-abuse-ring dataset I could find —
not "the best one", the only one. I did not modify it, augment it, or add a
single synthetic label anywhere near a headline number, because a detector's
numbers are only honest if the ground truth they are checked against is real.
The release is smaller than its own paper describes it to be, and every
number in this project is against what is actually on disk — see
`docs/data.md` for the five findings that came out of measuring the raw files
by hand rather than trusting the paper or the dataset's own readme. The
hostel test is the closest this project gets to an India-specific check
against the release's own limits, and three further, unrelated datasets
(Amazon, YelpChi, IEEE-CIS) test whether the method transfers past PPA at
all. [Jump to transfer](#how-amazon-yelpchi-and-ieee-cis-test-transfer).

---

## Why PPA

A promotion-abuse *ring* — several accounts controlled by one operator,
farming an offer together, with the label on the account rather than the
order — is a narrow and unusual thing to find labelled data for. Per-order
fraud datasets are common. Account-level fraud datasets are common. A dataset
with ring-shaped structure *and* account labels *and* the relations that let
you rebuild the graph is not. I looked for an alternative before committing
to this one — a payments dataset with group labels, a delivery or ride-share
release with referral-ring ground truth, anything closer to home than a
Chinese food-delivery platform — and did not find one that is both public and
labelled. PPA, released alongside PromoGuardian (Ma et al., IEEE S&P 2026,
[arXiv:2510.12652](https://arxiv.org/abs/2510.12652)), is what exists: real
accounts, real orders, real fraud labels, and eight relation types an
operator's accounts can share.

That is a statement about what I found, not a claim that nothing else exists
anywhere. If a better-fitting public dataset turns up, the pipeline is built
to point at a different loader, not to be rewritten — `orbweaver/data/`
already has three loaders in it (PPA, GADBench, IEEE-CIS) built to the same
shape for exactly that reason.

## Why the data being Chinese does not invalidate the method

Nothing this project measures is about China, or about food delivery
specifically. The signal the whole pipeline is built around — accounts that
share a rare entity are more likely to be coordinated than accounts that
share a common one, and coordinated accounts form denser structure than
chance — is a property of *how people evade detection when they are
sharing resources*, not a property of one country's payment habits or one
platform's promotion mechanics. The graph construction, the rarity
weighting, the densest-subgraph extraction and its approximation guarantee
do not read a country field anywhere.

What genuinely *would* need Indian data to confirm is anything about the
**base rates of Indian legitimate sharing** — how often a real family, a real
hostel, a real office shares a delivery address here, at what scale, with
what overlap. That is exactly why the hostel test exists rather than being
assumed away: it does not ask "does the method work", it asks "when this
method is pointed at the kind of coordinated-looking, overwhelmingly
innocent group that is routine in India, what does it do". [Below](#the-hostel-test).

Three further datasets — Amazon, YelpChi, IEEE-CIS — are run completely
unchanged through the same pipeline to ask a related but different question:
does the *result* replicate on unrelated data at all, regardless of geography.
[Below](#how-amazon-yelpchi-and-ieee-cis-test-transfer).

## Why the dataset was not modified or synthetically augmented

I did not change a label, add a row, remove an outlier, or synthesise a
single edge anywhere near a number this project reports as a result. The
reason is simple and I think under-stated in most write-ups: the entire value
of a labelled evaluation is that the labels were assigned by someone with no
stake in how my method performs. The moment I touch the ground truth —
relabel a borderline account, drop an inconvenient ring, inject a synthetic
fraud pattern I know my own method will catch — every number downstream stops
being a measurement and becomes a demonstration of whatever I decided to
show. That trade is not worth making even once, because it cannot be
partially undone: a reader has no way to tell a real 0.7292 from a curated
one, so *any* known instance of relabelling would put every number in this
project in doubt, not just the one that was touched.

This project's one deliberate exception proves the same rule rather than
breaking it. The adversarial-evaluation module adds *synthetic* edges to
argue what a payment aggregator's cross-merchant relation would be worth,
because PPA does not contain that relation at all. Every figure and table
produced from it is labelled, in the output itself, "simulated relation —
sensitivity analysis", and it is never quoted as a headline result — the
real, label-preserving version of the same question is answered later, on
YelpChi's real `net_rur` relation, against real labels. Simulation is used
exactly once, for a question the real data cannot answer, and it is fenced
off from everything else with a label a reader cannot miss.

The other side of "not modified" is what was *left broken*. `docs/data.md`
documents five things about the release that are inconvenient for a clean
write-up — the two order files do not share a user id space, the release is
57% of the users the paper describes, three of eight relations are entirely
empty in the files you can actually download, a CRLF parsing bug would have
fabricated 14 billion phantom edges from one unstripped byte if I had not
caught it. None of those became a reason to patch the data instead of the
pipeline. The pipeline adapted to what the data actually is; the data did
not get adjusted to make the pipeline's story cleaner.

## What the released data actually is, against what its own paper claims

In brief, because `docs/data.md` is the full page for this and I do not want
to duplicate it and risk the two drifting apart:

- **The release is smaller than the paper describes.** 3,267,961 users and
  10,012,449 edges, against 5,693,351 users and roughly 29,000,000 edges in
  the paper — 57.4% of the users, 34.5% of the edges. Every number in this
  project is against the smaller, actually-downloadable set; nowhere does
  this project describe itself as running on 5.7 million accounts.
- **Three of the graph's eight relations cannot be rebuilt from the raw
  order files at all** — `r2`, `r4` and `r5` are entirely empty there, even
  though `r4` alone is 38.41% of the authors' own pre-built `edge.csv`. The
  pipeline reports both a five-relation view built from the orders and the
  authors' eight-relation view, and what the missing three relations are
  worth is a measured number, not a guess (`docs/results.md`, "What the
  relations I cannot rebuild are worth").
- **The two order files are independently re-indexed**, which is the single
  most consequential finding on the data page: it rules out the
  train-on-week-1, test-on-week-2 protocol the paper's own description
  implies, and is the direct cause of the worst mistake in this project's own
  log (`FAILURES.md`).

None of this is a complaint about the dataset. A public, labelled release
with these properties is already unusual and useful; the point of measuring
it this closely is that every downstream number in this project should be
traceable to *files that exist*, not to a paper's description of a larger
dataset that is not the one anyone can download.

## The hostel test

The most obvious way this method could fail specifically in India is a false
positive on a shared delivery address that is not fraud at all — a hostel, a
paying-guest place, a joint family, an office. PPA's own labels cannot
directly confirm what a shared address means culturally, so the hostel test
asks the released data the closest question it can actually answer: among
groups that share one location entity, are large enough to look coordinated,
and whose labelled members are overwhelmingly normal — the structural
signature of an innocent shared-address population — how many does the
pipeline actually flag?

**2,446 such groups, 2 of them touched — 0.08%.** The full breakdown of what
separates the two that were touched from the 2,444 that were not is in
`docs/results.md`. This is the closest thing to an India-specific safety
check this release can support; it is not a substitute for testing on Indian
data, and `README.md` says so plainly under "what these numbers do not
prove". The same test is repeated on IEEE-CIS's billing address, a genuinely
different population, and the answer there is worse — 4 of 7 apartment
clusters touched — for a reason `docs/results.md` and
`docs/design-decisions.md` both explain: on a payment-processor graph, the
billing address is at once the most informative relation and the thing that
legitimately ties every card in a building together, and no reweighting
separates the two.

## How Amazon, YelpChi and IEEE-CIS test transfer

Three more datasets, chosen because they are nothing like PPA, run through
the identical extractor with no method-specific tuning:

- **Amazon reviewers and YelpChi reviews** (Dou et al., CIKM 2020; two of
  GADBench's ten benchmark datasets, Tang et al., NeurIPS 2023) are review-fraud
  graphs, not promotion-abuse graphs, from a different country and a different
  kind of platform. The same extractor, unpruned, again lands below the base
  rate — *exactly zero* on YelpChi: 25 rings, 1,914 accounts, none of them
  fraudulent. Pruned first, the identical code reaches 14.3× the base rate on
  Amazon and 6.9× on YelpChi. This is evidence that "dense is not the same as
  fraudulent, prune before you peel" is a property of the method, not a
  property of PPA — it is the finding this project would defend hardest, and
  it did not come from PPA at all.
- **IEEE-CIS** is real anonymised card-transaction data from a payment
  processor (Kaggle's IEEE-CIS Fraud Detection release). The account there is
  a card fingerprint, not a person, and this project says so explicitly
  everywhere the dataset is used. The same pipeline, unchanged, reaches
  0.5079 ring precision at 18.138× that dataset's own base rate. This is the
  closest thing in this project to evidence about payments specifically,
  and it is also where the method's weak point (the hostel test, above)
  shows up worse, not better — which is the point of testing transfer rather
  than only reporting the one dataset the method was built against.

Neither Amazon nor YelpChi ships timestamps, so that split is account-disjoint
but not forward in time, and both ship real node features that PPA does not —
`README.md`'s "what these numbers do not prove" section says plainly that
their much higher numbers say as much about what those datasets give you as
about the method. Transfer is evidence for the *shape* of the argument, not a
claim that any of these four datasets are interchangeable.

---

Back to [README](../README.md).
