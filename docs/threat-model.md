# Threat model

## What Orbweaver is built to catch

One party controlling many accounts to extract value that was meant for many
different people. Every order is individually legitimate; the abuse exists
only in the relationships between accounts.

| Abuse | What it looks like | Covered? |
|---|---|---|
| New-user offer farming | Many accounts, one operator, first-order discounts | **Yes** — directly, this is what PPA labels |
| Referral rings | A "refers" B, C, D…, all controlled by A | **Yes** — same graph signature |
| Cashback farming | Orders timed and shaped to milk cashback rules | **Yes** — labelled in PPA |
| Stocking up / reseller rings | Discounted stock bought across many accounts to resell | **Yes** — labelled in PPA |
| Seller-side collusion | A merchant places fake orders from controlled accounts | Same algorithm, different labels — not evaluated here |
| Delivery-partner collusion | Riders and fake customers farming incentives | Same algorithm, different labels — not evaluated here |
| Refund rings | Coordinated "item not received" claims | Same algorithm, different labels — not evaluated here |
| Money-mule networks | Proceeds hopping between accounts | Same algorithm, different labels — not evaluated here |

The last four share the structural signature and would need only different
labels and relations. I have not evaluated them, so I do not claim them.

## What it is not built to catch

- **A single fraudster acting alone.** No graph signature, nothing to peel.
  A per-transaction scorer is the right tool and Orbweaver sits downstream of
  one, not in place of it.
- **Account takeover.** The account belongs to a real customer with a real
  history; the fraud is in the session, not the relationships.
- **Payment-instrument fraud** such as stolen cards, *as a per-card problem*.
  The mechanism is different: one card used by one person who should not have
  it, rather than many accounts farming one offer. What does transfer is the
  case where shared entities link the accounts using those cards, and that is
  measurable rather than hypothetical — run unchanged on IEEE-CIS, where the
  relations are the device, e-mail domains, billing address and browser, the
  same extractor reaches 0.5079 ring precision at 18.14× that dataset's base
  rate. `docs/results.md` reports it with the caveats it needs, including that
  the account there is a card fingerprint rather than a person. So the boundary
  is the mechanism, not the payment domain.
- **Abuse with no shared entity at all.** If an operator uses a different
  address, device, promotion and coupon for every account and never overlaps,
  there is no edge to find. That is the honest boundary of the method, and it
  is the same boundary the adversarial evaluation probes.

## How an adversary would evade it

In roughly increasing order of cost to the attacker:

1. **Fragment the ring.** Split fifty accounts into ten cells of five that
   share nothing across cells. Density falls; below some cell size the group
   stops being distinguishable. Measuring where that threshold sits is the
   point of the fragmentation evaluation — the useful output is not "we catch
   everything" but "here is the cell size below which we do not." There is a
   partial answer to this one: accounts that behave alike still behave alike
   after every shared entity between them has been cut, and behavioural edges
   recover +0.0237 precision at cells of three. They recover nothing at cells
   of twenty, and they do not restore the curve, so fragmentation remains the
   attack that works.
2. **Dilute with camouflage.** Add edges to ordinary accounts through common
   entities so the group looks less cohesive. Rarity weighting is the direct
   answer: a shared entity that 3 million accounts have is worth almost
   nothing in the objective, so camouflage through common entities barely
   moves density. Camouflage through *rare* entities would work, but that
   means acquiring genuinely distinct addresses and instruments, which is the
   cost we want to impose.
3. **Slow down.** Spread the same behaviour over a longer window so less of it
   falls inside any one detection period. Effective, and it directly reduces
   the attacker's return per unit time.
4. **Use genuinely separate identities** — different address, device,
   instrument, promotion per account, never overlapping. This defeats the
   method completely. It also removes most of the economic advantage of
   running a ring, which is the point: the goal is to make the cheap version
   uneconomic, not to make fraud impossible.

## Costs of being wrong

**A false positive is a group, not a person.** Acting on a wrong ring of forty
accounts means forty real customers at once. This is why every detection
number in this project is reported next to its false-positive count and cost,
and why the output is a case file for review rather than an automated action.

**The most likely false positive is a shared address.** In India that is
routinely a hostel, a paying-guest place, an office or a joint family. The
evaluation includes a test built specifically on that population — co-located
groups whose labelled members are overwhelmingly normal — and reports both how
many the pipeline touches and what distinguishes those it touches from those
it leaves alone.

**The most likely false negative is a small, careful ring.** Below the size
floor, or spread across enough distinct entities, a group leaves no dense
structure. Lowering the floor to catch it raises the false-positive rate on
ordinary small groups such as families. That trade-off is a business decision
about review capacity and customer harm, not a modelling one, which is why the
operating point is reported as a curve rather than chosen here.
