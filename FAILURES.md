# What broke

An honest log, written as things happen rather than reconstructed at the end.
Each entry records what I believed, why it was wrong, and what fixed it.

Six of these are worth reading before the rest:

- [2 September — the densest groups were the innocent ones](#2-september--the-densest-groups-were-the-innocent-ones) — the result that changed the design: dense is not the same as fraudulent
- [2 September — a column that was full of values it did not have](#2-september--a-column-that-was-full-of-values-it-did-not-have) — a line ending made an empty column look 100% present
- [2 September — I trained a model on 183,370 strangers](#2-september--i-trained-a-model-on-183370-strangers) — two files, two id spaces, and a silent leak that scored well
- [3 September — `make reproduce` wrote a third of the report before the work existed](#3-september--make-reproduce-wrote-a-third-of-the-report-before-the-work-existed) — a green run for three and a half hours that produced the wrong file
- [3 September — I measured two systems at two different budgets and read the sign backwards](#3-september--i-measured-two-systems-at-two-different-budgets-and-read-the-sign-backwards) — an unfair comparison that reversed the project's own argument
- [3 September — the model I built lost to the baseline I almost did not run](#3-september--the-model-i-built-lost-to-the-baseline-i-almost-did-not-run) — the crudest ranking beat the trained one, and stayed in

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

## 2 September — `make reproduce` worked on my machine and nowhere else

*(Four separate failures, found by the same check. The fourth was the worst.)*

**What broke:** I cloned the repository into a clean directory, pointed it at
the raw data, and ran `make reproduce`. It failed on the third target:

    FileNotFoundError: data/processed/edges_week2_early.parquet

**What I believed:** that the pipeline reproduced end to end. I had run every
stage many times, `make reproduce` had never failed, and the results in
`docs/results.md` were regenerated from it.

**What it actually is: a circular dependency I had been hiding from myself.**
The per-relation weights are fitted *from* the early-window graph. The
early-window graph is built *with* those weights applied. On my machine both
files had existed since the first run, so whichever order the targets ran in,
something valid was always on disk and the cycle was invisible. On a clean
checkout the `weights` target runs before anything has built the graph it
reads, and there is nothing to read.

**How I got out:** made the two passes explicit. `windows-weighted` builds the
windows once with neutral weights, fits the multipliers from that graph, then
rebuilds the windows with them applied. The fit only reads the relation bitmask
on each edge, which does not depend on the weights, so the bootstrap pass and
the final pass agree on what they measure — the cycle is real but it converges
in one step, and now it says so in the code.

**Then it failed twice more,** and the other two were worse:

*A prerequisite with no rule.* I added `windows-weighted` to the `reproduce`
prerequisites and to `.PHONY`, but the patch that was supposed to write the
rule itself silently did nothing — a string replace whose anchor did not match
and which I had not asserted on. Make treats a phony target with no recipe as
satisfied, so it skipped the entire stage **without an error** and the run
failed two targets later on a missing file. A missing rule should be loud. This
one was silent because I had told make the target was phony, which is exactly
the promise that stops it complaining.

*Scores that were never written.* `make score` printed detection metrics and
never saved the scores. Ring extraction, the hostel test and both adversarial
runs all read `scores_week2.parquet`, and on a clean checkout there was nothing
to open. It worked here only because I had written that file by hand, once,
from a throwaway call, hours earlier.

**What I take from it:** "every number comes from `make reproduce`" was a rule
I had written down and believed I was following. It was true in the sense that
the numbers came out of those targets, and false in the sense that nobody else
could have produced them. All three failures share one cause — my machine had
accumulated artefacts that no clean checkout has, so the pipeline was quietly
depending on files nothing in it produced.

The uncomfortable part is the ordering. I wrote the results, the README and
most of the documentation *before* running this check, all of it describing a
pipeline that could not actually run anywhere else. A fresh clone is the only
thing that separates "it works" from "it works here", and it found three real
defects in about an hour.

**And then a fourth, which was the worst of them.** With the clone finally
running green, its ring precision came out 0.7292 where mine said 0.7100. My
first thought was that something was non-deterministic and I started looking at
XGBoost's threading.

It was not non-determinism. **Every number I had published was computed from a
stale scores file.** I wrote `scores_week2.parquet` early on with a one-off
call, then later fitted the relation weights, which changed the features
underneath it — and because `make score` did not persist scores until I fixed
that an hour earlier, nothing ever overwrote the old file. Every downstream
stage, and therefore every figure in the results, inherited scores computed
from features that no longer existed.

I checked rather than assumed, in this order: the graph is bit-identical
between the two machines, both feature tables are bit-identical, XGBoost
reproduces exactly at one thread and at ten, and regenerating the scores
locally produced a file identical to the clone's. The pipeline is
deterministic. The artefact was simply out of date, and nothing in the pipeline
could tell me so, because a parquet file on disk looks exactly as valid whether
it was written five minutes or five hours ago.

Every reported number is now from the clean run. They moved a little, and all
in the same direction — precision 0.710 → 0.729, false-positive cost 0.408 →
0.371 real customers per catch, hostel clusters touched 5 → 2 — which is
almost worse than if they had moved against me, because a change that flatters
you is one you are far less likely to go looking for.

**What I take from it:** an intermediate artefact with no provenance is a
liability. The fix that mattered was not any of the three bugs; it was making
`make score` write the file it reports on, so the thing on disk and the thing
in the numbers can never again be two different objects.

---

## 3 September — a feature that is zero for 99% of accounts cannot help

**What broke:** I fed ring context back into the per-account score — was this
account in a ring last window, how confident was that ring, how many of its
neighbours were — expecting the graph view to give something back to the
per-account view. The best of eight combinations moves held-out AUPRC by
**+0.0011**. That is nothing.

**What I believed:** that ring membership is a strong signal about an account,
so carrying it forward as a feature should help the next window's scoring.
The signal is strong. That was never the problem.

**What it actually is: coverage.** Of 76,404 held-out accounts, **116 were in a
previous-window ring — 0.15%**. The most widespread of the four context
features, having a neighbour who was in one, reaches 581 accounts, 0.76%. A
feature that is zero for more than ninety-nine accounts in a hundred cannot
move an aggregate metric no matter how informative it is on the hundredth, and
I should have computed that number before building the feature rather than
after.

**The deeper reason, which connects to the replay.** Rings do not persist from
one night to the next — that is measured, with a best overlap of 0.359 against
the 0.5 needed to call it the same group — and it turns out they do not
transfer from one window to the next either. Rings are window-specific objects.
The accounts recur; the groupings are recomputed. So ring membership is a poor
thing to carry forward, and two separate experiments have now told me the same
thing from different directions.

**What did work.** `GET /check/{account}` answers in **0.01 ms at the median
and 0.059 ms at the 95th percentile**, worst case 0.308 ms over a thousand
random accounts. Every index is built once at start-up, so a lookup is array
indexing rather than a file read. That matters because the claim this project
makes is that a per-transaction system could consume the ring view — and a
claim like that is worth nothing if the answer takes a second to produce. It
takes ten microseconds.

**What I take from it:** the ceiling on a feature is how many rows it is
non-zero for, and that is a one-line calculation available before any of the
work. I built four features, wired two combination rules and a fitted blend,
and every one of them was bounded at three decimal places by a number I could
have had in a minute.

---

## 3 September — the model I built lost to the baseline I almost did not run

**What broke:** I built a ring-level model to rank the review queue by learned
confidence instead of by density. It is worse than density at the top of the
queue, and both are worse than the crudest baseline I could think of.

    rings reviewed        25       100       200
    density           0.6859    0.5976    0.5814
    mean member score 0.7667    0.6951    0.6739
    learned model     0.5480    0.5866    0.5989

**What I believed:** that density is a poor way to order a queue — which it is
— and that a model over ring-level features would therefore beat it. The
reasoning was sound and the conclusion did not follow.

**What it actually is.** Of the 479 candidate rings with enough labelled
members to train on, **434 are majority-fraud. Ninety-one per cent.** The
candidates are generated at score cut-offs of 0.3 and 0.5, and that cut-off is
precisely the mechanism that makes rings fraud-enriched. By the time a group is
a candidate the question "is this a ring?" has already been answered, so there
is almost nothing left for a ring-level model to separate.

It shows in the calibration. The model puts 441 of the rings at a predicted
confidence of 1.0, and the single bucket it does push down it gets wrong in the
other direction — predicted 0.277 against a realised 0.658. Its median
confidence on legitimate co-located clusters is 1.0, exactly the same as its
median on everything else, which my test would have passed as "not ranked
higher than average" if I had not tightened it to require the confidences to
vary at all.

**What I got out of it anyway.** The spec made me report the mean-member-score
baseline alongside my model, and that baseline **wins at every depth** — 0.7667
against density's 0.6859 at 25 rings. Reordering the queue by the mean score of
a ring's members is a free improvement over what the queue does today, it needs
no new model, and I would not have found it if I had only compared my model
against the status quo. That is the result from this piece of work, and it is
not the one I was trying to produce.

I am leaving the whole comparison in the results rather than quietly dropping a
failed experiment. A ring-level model is a reasonable thing to try and the
reason it fails here is specific and interesting: the score cut-off is doing so
much work that it leaves the downstream model with no job.

**What I take from it:** I nearly ran this with two arms — density and my model
— because those are the two I cared about. The third arm cost one line and is
the only one that produced anything useful. A baseline you are confident you
will beat is exactly the one worth running.

---

## 3 September — rings do not survive the night

**What broke:** I set out to report days-to-detection — for each ring the last
night finds, the night it first became visible. The replay ran cleanly, and the
answer came back that every single ring was first detected on the last night.
Nought per cent seen earlier.

**What I believed:** that a ring accumulates. Members join over the days, the
group gets denser, and by the final night it is the same group it was on night
two with more people in it. Matching a final ring against earlier nights at
half its members overlapping seemed generous — I picked 0.5 precisely because I
expected partial rings to be the normal case and did not want to miss them.

**What it actually is:** ring identity does not persist between nights at all.
Measuring the best overlap each final ring achieves against any ring from an
earlier night, the median is **0.124** and the maximum across all 25 is
**0.359** — never once reaching the 0.5 I asked for, and not close.

The reason is in the algorithm rather than in the data. Peeling maximises a
global density ratio, so it is not tracking groups, it is re-solving an
optimisation. A night of new edges changes densities across the whole graph, a
different set of accounts survives to the top of the queue, and the top-25
rings are recomposed rather than extended. The accounts are still there. The
grouping is not.

**How I got out — and it is a weaker result.** I dropped group identity from
the question and asked instead how much of each final ring was already being
surfaced *somewhere* on an earlier night, which needs no matching. On that
measure 28% of the final rings had half their members already inside some
surfaced ring before the last night, with 37% of their promotion spend still
ahead of them at that point. That is what a team could act on, and it is a long
way short of "this ring was visible on night two".

There is a genuine result next to it, which I nearly buried under the one that
failed. Precision by nights of data goes 0.2045 → 0.4869 → 0.6098 → 0.7292,
and the cost of being wrong goes 3.891 real customers per catch down to 0.371.
**With one night of data the system is at the base rate and worthless.** It
needs three or four days of accumulated structure before it is worth running,
which is an operational constraint I would have had no way of stating without
the replay.

**What I take from it:** I had also written a bug that would have hidden this.
The overlap variable was initialised at 1.0 and only written on a match, so
every unmatched ring reported a perfect 1.0 overlap. I only caught it because
"100% of rings matched at exactly 1.0" is not a distribution any real
measurement produces. Had I initialised it at 0.0 the output would have looked
merely disappointing rather than absurd, and I might have believed it.

---

## 2 September — the one part of the pipeline that was not reproducible

**What broke:** with everything else pinned down, I re-ran `make reproduce`
end to end and diffed the regenerated documentation against what was
published. One line moved. The GraphSAGE row: held-out AUPRC 0.3825 became
0.3830. Every other number in the file was identical.

**What I believed:** that I had already settled the determinism question. I
had checked it carefully earlier — the graph and both feature tables are
bit-identical across machines, XGBoost reproduces exactly at one thread and at
ten — and concluded the pipeline was deterministic and an artefact had simply
been stale. That was true, and I had checked only the components that were
stale.

**What it actually is:** the GNN trains on MPS, and Metal's scatter reductions
accumulate in a non-deterministic order. Running it three times gave 0.3825,
0.3830 and 0.3833.

I isolated it rather than guessing: twelve identical training batches, same
seeds, same data, run twice on each device. CPU produced bit-identical losses
both times. MPS diverged at batch 7, by 6e-8 — floating-point addition order
in the aggregation kernel, amplified over three epochs into the third decimal
place of AUPRC.

**How I got out:** the GNN now runs on CPU by default, which reproduces
exactly — two full runs give AUPRC 0.3819 and identical precision, recall and
F1 to four decimals. `ORBWEAVER_SAGE_DEVICE=mps` still gets the roughly 3x
faster run for anyone who wants it. The cost of the change is about two extra
minutes and nothing else: 0.001 of AUPRC never came close to changing the
conclusion, which is that the GNN ties the gradient-boosted model.

**What I take from it:** this one is uncomfortable because the deviation was
so small. A difference in the third decimal is exactly the size that gets
waved through as noise, and had I not diffed the whole file I would never have
looked at it. It also would have been easy to argue it did not matter — it
genuinely does not change any result — and to leave a stated principle
quietly false. Reproducible has to mean reproducible, or it means whatever is
convenient at the time.

---

## 2 September — I gave up on the generalisation check too early

**What broke:** my first attempt at running on other datasets got nowhere.
GADBench distributes its ten datasets as one Google Drive bundle, the link does
not script cleanly, and the GitHub API rate-limited me while I was poking at the
repository. I recorded it as blocked, substituted a comparison against the
PromoGuardian authors' own graph, and moved on.

**What I believed:** that the data was behind an obstacle I could not remove in
the time available, and that a different comparison tested the same property
well enough.

**What it actually was: two separate mistakes, both mine.**

First, I had fetched `README.md` and got a 404, twice, and read that as the
repository being awkward. The file is called `readme.md`. Lowercase. I never
listed the directory to check.

Second, and more useful: when I did find the bundle, it turned out to be in
DGL's serialisation format — `dgl.data.utils.load_graphs` — which I cannot
read on this machine for exactly the reason the authors' checkpoint failed.
That part was a real blocker. But the datasets are not *originally* GADBench's.
Amazon and YelpChi come from CARE-GNN (Dou et al., CIKM 2020) and are published
there as `.mat` files, sitting in the repository itself, 44 MB, no Drive link,
readable by scipy in one line. The obstacle was the packaging, not the data,
and I had stopped at the packaging.

**What it was worth:** the central finding of the whole project replicates on
both, independently. Unpruned, the extractor lands at 0.57× the base rate on
Amazon and at **exactly zero** on YelpChi — 25 rings, 1,914 accounts, not one
fraudster among them. With the score cut-off: 14.3× and 6.9×. Three datasets
from three unrelated platforms now say the same thing, and that is a far
stronger claim than one dataset could support.

It also put PPA's difficulty in perspective, which I had been reading as a
limitation of my method. Node scoring reaches AUPRC 0.76 and 0.86 on these
against 0.38 on PPA, because both ship real node features and PPA ships none at
all. YelpChi's strongest relation carries a 49.8× fraud lift where PPA's best
is 3.7×.

**What I take from it:** "the data is behind a Google Drive link" was a
description of one distribution channel, not of the data. I checked whether
*that* path worked and concluded the check was infeasible, when the actual
question was where the datasets originally came from — which was one search
away. Being blocked by a format is a fact; being blocked by the first
packaging you tried is a decision.

---

## 2 September — I could not run the authors' checkpoint, and stopped trying

**What I wanted:** to run PromoGuardian's released checkpoint on the same graph
I built, so their scores and mine would be a column each in the same table.
Their repository ships everything needed — `test.py`, `env.yml`,
`emb_transr_R_8.npy`, and a 128 KB `model_checkpoint_weighted_TransR.pth`.

**What broke:** their code is built on DGL, and there is no DGL wheel for this
machine. `pip install dgl` returns *"no matching distribution"*, and so does
their own wheel index at `data.dgl.ai`. Python 3.13 on arm64 macOS is simply
not a platform DGL publishes for.

**What I believed going in:** I had read their `test.py` before starting and
noted the hard DGL dependency as a risk, so this was a predicted failure rather
than a surprise. That is the only reason it cost twenty minutes instead of an
afternoon.

**What I did instead of pushing on:** I stopped. The options left were building
DGL from source, downgrading the whole project to Python 3.11, or standing up a
container — each of which is hours, and none of which improves the thing I am
actually contributing. Their published numbers (precision 0.9107, recall
0.6992, F1 0.7911) are in `docs/results.md` as a reference row, with the
comparability caveat that matters more than the gap: they evaluate on the full
graph with all eight relations, count unlabelled accounts as negatives, and
apply no account holdout. My numbers use five relations, a four-day window,
and accounts the model has never seen. Those are not the same measurement and I
do not present them as one.

**What I take from it:** reading the dependency before planning the work turned
an unbounded problem into a twenty-minute one. The temptation was to treat
"get their checkpoint running" as a milestone because it would make a nicer
table; it would not have changed a single design decision in this project.

---

## 2 September — the densest groups were the innocent ones

**What broke:** the whole premise, briefly. With the extractor fixed and
producing sensible 41-to-466-account rings, I measured ring precision for the
first time: **0.051**, against a base rate of **0.224**. The rings were *four
times worse than picking accounts at random*. Every operating point in the
first λ sweep came out between 0.12 and 0.19 — all of them below base rate.

**What I believed:** that dense means fraudulent. The entire pipeline is built
on finding dense subgraphs, and I had taken it as given that a tightly
connected group of accounts sharing rare entities is a ring. The planted-ring
tests passed, the peeling was correct, the evidence extraction worked. The
algorithm was doing exactly what I designed it to do, and what it found was
mostly innocent people.

**What it actually is.** Two measurements, both of which I should have taken
before writing the extractor rather than after:

*Fraud is assortative, but only mildly.* Fraud–fraud edges are 7.67% of
labelled-to-labelled edges against 3.19% expected by chance — a 2.4× lift.
Real, and not nearly enough on its own. Fraudsters also have *lower* mean
degree than normal accounts (22.05 against 24.10), so the densest thing in the
graph is systematically not the most fraudulent thing.

*The relation that dominates the graph is the least informative one.*
Per-relation fraud lift: location 3.68×, coupon 2.75×, stimulation 2.55×,
delivery 2.46×, **promotion 1.76×**. Promotion edges are **70% of the whole
graph**. So the densest structures are promotion cliques — hundreds of
ordinary customers who used the same offer — and my rarity weight had no way
to know that, because it only measures *how many people share a thing*, never
*what kind of thing it is*.

**How I got out,** in two steps, of which the second matters far more.

First, weight each relation by its measured lift, fitted on training accounts
only. Location edges up, promotion edges down.

Second, and this is the real fix: **filter to suspicious accounts before
peeling at all.** Restrict the graph to accounts the scorer puts above a
threshold, then look for dense structure inside that region. The effect is not
subtle:

    no cut-off      ring precision 0.070    0.31x base rate
    tau = 0.3       ring precision 0.579    2.58x base rate
    tau = 0.5       ring precision 0.729    3.25x base rate

The score cut-off was in my design as a *speed* optimisation — a way to avoid
peeling the whole graph. It turns out to be the thing that makes the output
mean anything. Density finds cohesive groups; it cannot tell you which
cohesive groups are criminal. The model answers that question, and the
deterministic objective still decides who is in the ring — which is the
property I actually cared about protecting.

**What I take from it:** I had measured that my algorithm found the densest
subgraph correctly, and mistaken that for evidence that it found rings. Those
are different claims and only the first one has a proof. The approximation
guarantee is about the *search*, not about whether the thing being searched
for is the thing you want — and I spent a long time being reassured by a bound
that was never going to tell me my objective was pointed slightly wrong.

The unpruned row stays in the results table permanently. It is the honest
baseline, and it is the reason the rest of the table is worth reading.

---

## 2 September — the extractor found a ring with 31,555 members

**What broke:** the first real run of the ring extractor on the week-2 graph.
The top seven results were exactly what I wanted — 51 to 226 accounts, density
12 to 24, tight and plausible. Then rank 8 came back with **10,593 members**,
rank 9 with 17,113, and rank 10 with **31,555**.

**What I believed:** that a maximum-density objective would naturally prefer
small tight groups, so a floor on ring size (`k_min`) was the only size
constraint I needed. I had written `k_min = 5` and given no thought at all to
the other end.

**What it actually is:** density is a *ratio*, and a large region of moderate
density can beat a small region of high density on that ratio. This graph has
a very thick core — its 50-core still contains 453,983 accounts — so once
peeling gets down into that core there is an enormous set sitting at density
around 10, and 31,555 accounts at density 9.9 scores better than 45 accounts
at density 9.6. The objective was working correctly. It was answering a
question I had not meant to ask.

The result is useless in exactly the way that matters: a "ring" of 31,555
accounts is not a case anyone can review, and shipping it as a detection would
mean recommending action against a substantial slice of the customer base.

**How I got out:** added an upper bound, `k_max = 500`, and only let a set be
recorded as the best if its size is inside `[k_min, k_max]`. This changes the
problem from densest-at-least-k to **densest-at-most-k**, which is also
NP-hard, and greedy peeling keeps a constant-factor bound on it. Rings now
come out at 41 to 466 members with densities from 7 to 24.

I picked 500 for a reason I can defend rather than one that maximises a
metric: a ring is a case a human reviews as a single unit, and beyond a few
hundred accounts it is a community, not a case. `k_max` is a config value and
I report results across a sweep of it, so the choice is visible.

`tests/test_peel_planted.py` now plants a large loose cluster next to a small
tight one and asserts the extractor returns the tight one.

**What I take from it:** I had read the warning about this exact failure —
"the extractor returns one giant component" — and had still only guarded one
side of the size constraint, because the objective *felt* like it preferred
small dense things. The seven good rings above it are what made this
dangerous: if I had looked at the top five and stopped, I would have shipped
an extractor that produces garbage from rank 8 onward and never known.

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

---

## 3 September — I measured two systems at two different budgets and read the sign backwards

**What broke:** the comparison at the heart of this project's business argument.
On YelpChi I ran the extractor twice: once with all three relations, and once
without `net_rur` — "posted by the same user" — which is the link only a
platform holds, because no single business can see that the reviewer who hit it
also hit forty others. I took the top 25 rings from each arm and compared them.
The merchant arm found 1,761 fraud accounts; the platform arm found 678. Read
straight, the relation only a platform can see was worth *minus* 1,083 fraud
accounts, and the aggregator's advantage — the thing this whole project argues
for — was actively harmful.

**What I believed:** that two runs of the same extractor, at the same operating
point, cut at the same top-25, were comparable. Same algorithm, same cut, same
labels; the only difference is the relation, so the difference in the answer is
what the relation is worth.

**Why that was wrong:** the top 25 *rings* is not a fixed amount of work. The
merchant arm's 25 rings held 1,856 accounts. The platform arm's held 681. I was
comparing an analyst team that reviews 1,856 accounts against a team that
reviews 681 and concluding the first team was better because it found more
fraud. It reviewed nearly three times as much. Of course it found more.

**What fixed it:** measuring at equal review capacity instead. `precision_at_budget`
walks the ranked rings, accumulates distinct accounts until it hits a budget,
and reports the fraud share at 250, 500 and 1,000 accounts reviewed — with both
arms pinned to the same score cut-off, which they had not been either. At 250
accounts the platform arm is ahead by +0.024 precision; at 500, by +0.038. The
sign flipped the moment the comparison was fair.

**What I take from it:** a per-item cut is not a unit of work when the items are
different sizes. Any comparison of two ranked lists whose entries cost different
amounts to act on has to fix the cost and let the count fall out, never the
other way round. I had the wrong version in a figure and a written paragraph
before I noticed, and what made me look was not a test — it was that the number
was too large. A relation cannot plausibly be worth minus a thousand accounts.

---

## 3 September — behavioural edges helped most where the graph was already ruined

**What broke:** nothing crashed. I added behavioural-twin edges — mutual
5-nearest-neighbour links in standardised feature space, drawn only between
accounts the scorer had already flagged, and weighted the same way every other
relation is: their measured fraud-fraud lift on training accounts (1.247) times
the median entity edge weight, giving 0.230. The point was to give fragmentation
something it cannot cut. An attacker who splits a ring into cells of three so
that no shared address or promotion joins them has not changed how those
accounts behave.

**What I believed:** that recovery would grow with the damage. The more entity
edges an attacker severs, the more the behavioural edges should matter, so the
curve should rise monotonically as the cells get smaller.

**What actually happened:**

| cells of | without twins | with twins | change |
|---|---|---|---|
| intact | 0.7292 | 0.7458 | +0.0166 |
| 20 | 0.5846 | 0.5823 | −0.0023 |
| 10 | 0.5444 | 0.5546 | +0.0102 |
| 5 | 0.5109 | 0.5114 | +0.0005 |
| 3 | 0.4539 | 0.4776 | +0.0237 |

The biggest recovery is at the worst damage, which is the direction I wanted.
But it is not monotonic, and at cells of 20 the twins cost a little.

**My hypothesis, labelled as one:** when the graph is cut into threes almost no
entity structure survives, so 97,016 twin edges are a large share of what is
left and their weaker evidence is still the best evidence available. At cells of
20 enough entity structure survives that the twin edges are largely redundant
with it, and adding a weaker relation on top of a sufficient one dilutes the
peeling objective slightly. That story fits the shape. I have not proved it, and
I am not going to pretend the middle of the curve is explained.

**The cost, stated:** the hostel test goes from 2 of 2,446 legitimate clusters
touched to 3. Behavioural edges do connect some genuinely similar ordinary
customers, which is exactly what you would expect them to do.

**What I take from it:** I nearly reported "behavioural edges recover
fragmentation damage" on the strength of the cells-of-3 number alone, because it
was the number I had gone looking for. All four cell sizes are in the report
because one point is not a curve, and the flat and negative points are the ones
that make the +0.0237 believable.

---

## 3 September — the most useful relation is the one that ties innocent people together

**What broke:** the collateral test, on the run with the best headline in the
project. IEEE-CIS gives 0.5079 ring precision at 18.14× its base rate — the
largest lift anywhere in this work. Then I ran the address-cluster test, which
is the apartment-building analogue of the hostel test: find billing addresses
with 15 or more cards on them where at least 80% of the labelled cards are good,
and count how many of those the rings touch. On the delivery data the equivalent
test touches 2 of 2,446 legitimate co-located groups. Here it touches 4 of 7.

**What I believed:** that a weighting rule which measured its way to a clean
collateral profile on one dataset would keep that profile on a new set of
relations, because the rule is the same and the rule is what protects innocent
groups.

**Why that was wrong:** the weighting is doing precisely what it should, and
that is the problem. `address_distance` is the most informative relation this
dataset offers — 5.42× fraud-fraud lift over 130,427 labelled edges — so it
earns the heaviest weight, 2.07, more than three times the payer e-mail domain.
It is also the single thing that legitimately ties together every card billed to
one apartment building. Those two facts are the same fact. The weighting cannot
separate them, because the weighting is what discovered the address was
informative in the first place. On the delivery data the strongest relations
were the device and the payment instrument, which co-locate far less innocently,
and I had quietly generalised from that.

**What I have not done:** seven is not a sample. Four of seven is four events,
not 57%, and I have not put a rate on it anywhere — the section in
`docs/results.md` says so in as many words. Establishing the real rate needs a
dataset with more buildings in it than this one has.

**What I take from it:** the hostel test result I am proud of is a property of
*which relations that dataset happens to have*, not a property of the method. A
relation set where the most predictive edge is also the most innocently shared
one would need an address-level exclusion list before I would run it against
real customers, and I would rather write that down than let 18× stand on its own.

---

## 3 September — `make reproduce` wrote a third of the report before the work existed

**What broke:** I cloned the repository into an empty directory, emptied
`data/processed`, and ran `make reproduce` the way a reader would. Twenty-six
stages, three and a half hours, exit code 0, every stage green. Then I diffed
the `docs/results.md` it produced against the one in the repository: 195 lines
short. Six entire sections missing — every section I had added in the last two
days of work. The README's generated table was missing its last six rows, and
one figure had silently lost a curve.

**What I believed:** that a green `make reproduce` meant the repository
regenerates itself, and that because `report` was written last in the list, it
ran last:

```make
reproduce-core: schema data graph ... generalise report
reproduce: reproduce-core merchant-view replay ring-scorer ... ieee-cis report
```

**Why that was wrong:** make builds any target at most once per invocation.
`report` was already a prerequisite of `reproduce-core`, and `reproduce-core`
is the *first* prerequisite of `reproduce` — so the report was generated in the
middle of the run, before the six stages named after it had produced anything
at all. Naming it again at the end was a silent no-op. No error, no warning;
make had satisfied that target and moved on. The stages themselves ran fine and
wrote their artefacts. Nothing read them.

**Why I did not notice for two days:** because I ran `python3 -m eval.report`
by hand after finishing each piece of work, so my own copy of the file was
always current, and the committed `docs/results.md` was correct. The file was
right. The command that is supposed to produce the file had never once produced
it. Every check I had been running used my history, not a reader's.

**What fixed it:** `report` is now a recipe step of `reproduce` rather than a
prerequisite, so it always runs after everything else. It runs twice in a full
pass, which costs about two minutes, and that is the right trade for a
guarantee that the report describes the run that just happened.

**Two tests, because one of them is not enough.** The first checks the symptom:
for every artefact on disk, the section that reports it must be present in
`docs/results.md`. The second checks the cause: `reproduce` must not name
`report` as a prerequisite. I put the old line back and confirmed the second
test fails, because a regression test that passes against the broken code is
not a test.

**What I take from it:** I had "clone it fresh and run it" written down as the
last thing to do, and I had been treating it as a formality — the box you tick
once the real work is finished. It found the worst bug in this project. It
found it because it was the only check that ran the command a reader runs,
rather than the commands I had got into the habit of running. Twenty-six green
stages and an exit code of 0 told me nothing whatsoever about whether the
output was correct.

The rest of the diff, once the ordering was fixed, was six lines: the `/check`
latencies and the per-night seconds in the replay. Those are wall-clock
measurements of the machine, they cannot reproduce byte for byte, and
`docs/results.md` now says so at the top rather than leaving a reader to
discover it in a diff.

---

## 3 September — the hand-written pages drifted, and the generated ones did not

**What broke:** a read-through of the public documents as a first-time reader,
checking every claim against the code and the artefacts rather than against my
memory of them. Four claims had gone stale. Two documents quoted the location
relation's fraud lift as 3.68× when the artefact says 3.7072. The design notes
described the GraphSAGE scorer training in 95 seconds on the GPU with a peak of
3.56 GB, when the recorded run is the CPU at 42 seconds, and measuring the
memory properly gave 5.42 GB. The threat model said the method is not built
for payment-instrument fraud, a day after I had run it on 590,540 card
transactions and reported the result. And the README never mentioned that
there is a console to look at.

**What I believed:** that the prose documents were stable because they
describe *design*, and design does not change when a run is repeated.

**Why that was wrong:** they describe design *with numbers in it*, and every
number typed into prose is a claim about a particular run that a later run can
supersede. The GraphSAGE paragraph was written when the model trained on
Metal; I moved it to the CPU for reproducibility and updated the code, the
docstring and the generated results, and not the page that argues the design.
The lift figure came from an earlier fit of the relation weights. Nothing
regenerated those sentences because nothing generates them.

**The pattern, which is the useful part:** every number that came out of
`eval/report.py` was correct, in every document, on every check. Every stale
number had been typed by a person. This project's rule that the README carries
no hand-typed numbers was written for exactly this reason, and it had been
applied to the README and the results and not to the three pages that explain
them.

**What fixed it:** the four claims, and a test that pins the numbers the
prose *has* to quote — the two relation lifts, the two edge shares, the
feature count — to the artefacts they come from, so the suite says which page
has drifted rather than a reader finding it. The wall-clock and memory figures
are not pinned, because they are the two kinds of number that legitimately
differ between runs, and the design page now says so where it quotes them.

**What I take from it:** a document I would not regenerate is a document I
should test, and I had been treating "written by hand" as a reason to trust it
rather than the opposite.
