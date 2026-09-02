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
- **Payment-instrument fraud** such as stolen cards, unless the same
  instrument links accounts — which is the relation this dataset does not have
  and only a payment aggregator could build.
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
   everything" but "here is the cell size below which we do not."
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
