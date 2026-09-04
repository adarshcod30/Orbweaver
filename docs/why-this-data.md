# Why this dataset

**[← README](../README.md)** · [What the release actually is](data.md) · [Results](results.md)

The question a reviewer asks first, answered in one table:

| Question | Answer |
|---|---|
| **Why PPA?** | It is the **only** public, labelled promotion-abuse-*ring* dataset I could find. Not the best one — the only one |
| **Was it modified?** | No. Not a label changed, a row added, an outlier dropped, or a synthetic edge anywhere near a headline number |
| **Does Chinese data invalidate it?** | Not for the mechanism. It would for *base rates of Indian legitimate sharing* — which is exactly why the [hostel test](#the-hostel-test) exists |
| **Is the release what its paper describes?** | No — it is smaller. Every number here is against what is actually downloadable ([measured file by file](data.md)) |
| **Does it work anywhere else?** | Three unrelated datasets say yes: [Amazon, YelpChi, IEEE-CIS](#how-amazon-yelpchi-and-ieee-cis-test-transfer) |

## Why PPA

A promotion-abuse **ring** — several accounts controlled by one operator,
farming an offer together, with the label on the *account* rather than the
order — is a narrow thing to find labelled data for. Per-order fraud datasets
are common. Account-level fraud datasets are common. A dataset with ring-shaped
structure *and* account labels *and* the relations to rebuild the graph is not.

I looked for an alternative before committing: a payments dataset with group
labels, a delivery or ride-share release with referral-ring ground truth,
anything closer to home than a Chinese food-delivery platform. I did not find
one that is both public and labelled. PPA — released with PromoGuardian
(Ma et al., IEEE S&P 2026, [arXiv:2510.12652](https://arxiv.org/abs/2510.12652))
— is what exists: real accounts, real orders, real fraud labels, and eight
relation types an operator's accounts can share.

That is a statement about what I found, not a claim nothing else exists. If a
better-fitting dataset turns up, the pipeline points at a different loader
rather than being rewritten — `orbweaver/data/` already holds three loaders
(PPA, GADBench, IEEE-CIS) built to the same shape for exactly that reason.

## Why the data being Chinese does not invalidate the method

Nothing this project measures is about China, or about food delivery. The
signal the whole pipeline is built on — *accounts that share a rare entity are
more likely to be coordinated than accounts sharing a common one, and
coordinated accounts form denser structure than chance* — is a property of how
people evade detection when sharing resources, not of one country's payment
habits. The graph construction, rarity weighting, densest-subgraph extraction
and its approximation guarantee never read a country field.

What genuinely **would** need Indian data is anything about the **base rates of
Indian legitimate sharing** — how often a real family, hostel or office shares
a delivery address here, at what scale, with what overlap. Which is why the
hostel test exists rather than being assumed away.

## Why the dataset was not modified or synthetically augmented

The entire value of a labelled evaluation is that the labels were assigned by
someone with **no stake in how my method performs**. The moment I touch the
ground truth — relabel a borderline account, drop an inconvenient ring, inject
a synthetic pattern I know my method catches — every number downstream stops
being a measurement and becomes a demonstration.

That trade cannot be partially undone: a reader has no way to tell a real
0.7292 from a curated one, so *any* known instance of relabelling puts every
number in doubt, not just the one that was touched.

**The one deliberate exception proves the rule.** The adversarial module adds
*synthetic* edges to argue what a payment aggregator's cross-merchant relation
would be worth, because PPA does not contain that relation at all. Every figure
from it is labelled, in the output itself, "simulated relation — sensitivity
analysis", and it is never quoted as a headline. The real, label-preserving
version of the same question is answered later on YelpChi's real `net_rur`
relation, against real labels.

**The other side of "not modified" is what was left broken.** [`data.md`](data.md)
documents five inconvenient things about the release — two order files with
different user id spaces, 57% of the users the paper describes, three of eight
relations entirely empty, a CRLF bug that would have fabricated 14 billion
phantom edges from one unstripped byte. None became a reason to patch the data
instead of the pipeline. **The pipeline adapted to what the data is; the data
did not get adjusted to make the pipeline's story cleaner.**

## The hostel test

The most obvious way this method fails *specifically in India* is a false
positive on a shared delivery address that is not fraud at all — a hostel, a
PG, a joint family, an office.

PPA's labels cannot confirm what a shared address means culturally, so the test
asks the closest question the data can answer: among groups that share one
location entity, are large enough to look coordinated, and whose labelled
members are overwhelmingly normal — the structural signature of an innocent
shared-address population — how many does the pipeline flag?

> **2,446 such groups. 2 touched. 0.08%.**

The full breakdown of what separates the two from the 2,444 is in
[results.md](results.md). This is the closest thing to an India-specific safety
check this release supports; it is **not** a substitute for testing on Indian
data, and the README says so plainly.

The same test on IEEE-CIS's billing address — a genuinely different population
— is **worse**: 4 of 7 apartment clusters touched. The reason is in
[results.md](results.md) and [design-decisions.md](design-decisions.md): on a
payment-processor graph the billing address is at once the most informative
relation *and* the thing that legitimately ties every card in a building
together, and no reweighting separates the two.

## How Amazon, YelpChi and IEEE-CIS test transfer

Three datasets chosen because they are nothing like PPA, run through the
identical extractor with no method-specific tuning:

| Dataset | What it is | What it showed |
|---|---|---|
| **Amazon reviewers** | Review fraud, different country, different platform (Dou et al., CIKM 2020) | Pruned first: **14.3×** base rate |
| **YelpChi reviews** | Review fraud (Dou et al., CIKM 2020) | Unpruned: **exactly zero** — 25 rings, 1,914 accounts, none fraudulent. Pruned: **6.9×** |
| **IEEE-CIS** | Real anonymised card transactions from a payment processor | **0.5079** ring precision at **18.138×** its base rate — and the hostel weakness shows up *worse*, not better |

The YelpChi result is the important one. The same extractor, unpruned, lands
below the base rate on a completely unrelated platform — evidence that
**"dense is not the same as fraudulent, prune before you peel"** is a property
of the *method*, not of PPA. It is the finding this project would defend
hardest, and it did not come from PPA at all.

On IEEE-CIS the account is a **card fingerprint, not a person**, and this
project says so everywhere the dataset is used. It is the closest thing here to
evidence about payments specifically.

**The honest caveat:** neither Amazon nor YelpChi ships timestamps, so that
split is account-disjoint but *not* forward in time, and both ship real node
features that PPA does not. Their much higher numbers say as much about what
those datasets give you as about the method. Transfer is evidence for the
*shape* of the argument — not a claim that these four datasets are
interchangeable.

---

**[← README](../README.md)** · Next: [what the release actually contains →](data.md)
