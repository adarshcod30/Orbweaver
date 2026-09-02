# Scope and ethics

This is a detection system. It contains no attack tooling, and I have kept it
that way deliberately.

- **Detection only.** Nothing here helps anyone commit promotion abuse. The
  adversarial evaluation reproduces a published evaluation protocol — node
  duplication and group fragmentation on a public labelled dataset — to
  measure how the detector degrades when fraudsters adapt. It has no
  capability against any real system, and it operates only on data already
  released for research.

- **Public research data only.** PPA is a public dataset released with a
  peer-reviewed paper, with anonymised identifiers throughout. GADBench, if
  used, is likewise public. No real merchant, customer, or payment system is
  touched anywhere in this repository, and there are no credentials in it.

- **Outputs are case files, not verdicts.** A ring is a recommendation for a
  human to review, with the evidence attached so it can be disagreed with. I
  have not built, and would not build, an automated ban.

- **False-positive cost is reported next to every detection number.** Wrongly
  flagging a real customer is a harm, not a rounding error, and a precision
  figure quoted on its own hides it. A ring makes this sharper than
  per-account scoring does: acting on a wrong ring of forty accounts loses
  forty real customers at once.

- **Simulated data is labelled as simulated.** The aggregator-relation
  experiment adds a payment-instrument link that PPA does not contain. Every
  figure and table from it says "simulated relation — sensitivity analysis",
  and it is never quoted as a headline result.

- **Shared attributes are not guilt.** In India a shared delivery address is
  very often a hostel, a paying-guest accommodation, an office or a joint
  family. The evaluation includes a test for exactly this population, and
  reports how the pipeline treats it, because a detector that cannot tell a
  hostel from a fraud ring should not be deployed.
