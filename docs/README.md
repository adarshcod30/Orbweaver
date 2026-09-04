# Doc index

Nine files, no wrong door. Every one leads with an "In one minute" section
except this one and `results.md`, which is generated in full and already
short.

| File | Read this if… |
|---|---|
| [`../README.md`](../README.md) | You are starting cold. Everything below is reached from here. |
| [`why-this-data.md`](why-this-data.md) | You are asking "why this dataset, and is it trustworthy" — the question a reviewer asks first. |
| [`results.md`](results.md) | You want every number, table and figure this project produces, generated in full by `make reproduce`. |
| [`architecture.md`](architecture.md) | You want to know how a request actually flows through the five stages, with diagrams. |
| [`design-decisions.md`](design-decisions.md) | You want to know exactly where a model is trusted and where a proof is used instead — and why a second, provable method beats the learned one. |
| [`data.md`](data.md) | You want the raw PPA release measured file by file — what it actually contains against what its own paper claims. |
| [`threat-model.md`](threat-model.md) | You want to know what this catches, what it does not, and how an adversary would try to get past it. |
| [`../FAILURES.md`](../FAILURES.md) | You want the honest log — what broke, what I believed, why that was wrong, what fixed it. |
| [`../ETHICS.md`](../ETHICS.md) | You want the scope and ethics boundary in six lines. |
| [`case-files.html`](case-files.html) | You want the review queue itself, one card per ring, as a page rather than a document. |

Also in this directory: `figures/` (every chart `results.md` embeds, regenerated
by `make report`), `index.html` (the GitHub Pages landing page, generated),
and `.nojekyll` (so Pages serves this directory as static files rather than
running it through Jekyll first).

Back to [README](../README.md).
