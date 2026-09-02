"""Does ring context improve the per-account score, and can it be served fast?

Two questions, and the second is not decoration. If a per-transaction system
is going to consume this, it has to be able to ask about one account and get an
answer inside a request, not run a batch job. So the latency of `/check` is a
result here, measured the same way the precision is.

The primary comparison is parameter-free by design: two fixed ways of folding
context into the score, neither with anything to fit, so there is no way to
tune on the measurement. A one-parameter blend is reported after, fitted only
on the validation split, with its horizon stated.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pyarrow.parquet as pq

from orbweaver.config import Config, load_config
from orbweaver.features.ring_context import (
    CONTEXT_FEATURES, build_context, combine,
)

PRECISION_AT = (100, 500, 1000, 5000)


def precision_at_k(y: np.ndarray, s: np.ndarray, ks=PRECISION_AT) -> dict:
    order = np.argsort(-s)
    out = {}
    for k in ks:
        if k > y.size:
            continue
        out[str(k)] = round(float(y[order[:k]].mean()), 4)
    return out


def run(cfg: Config | None = None) -> dict:
    from sklearn.metrics import average_precision_score

    from eval.split import make_split

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    labels = split.labels
    n = labels.size

    ctx = build_context(cfg, n)
    feats = ctx["features"]

    scores = np.zeros(n)
    t = pq.read_table(proc / "scores_week2.parquet")
    scores[t["user_id"].to_numpy()] = t["score"].to_numpy()

    test = split.test
    y = split.y(test)
    base = {
        "auprc": round(float(average_precision_score(y, scores[test])), 4),
        "precision_at_k": precision_at_k(y, scores[test]),
    }

    arms = {"score alone": base}
    for cname in CONTEXT_FEATURES:
        c = feats[cname]
        if c[test].max() == 0:
            arms[f"score + {cname}"] = {"skipped": "no held-out account has this"}
            continue
        for how, blended in combine(scores, c).items():
            arms[f"{how} [{cname}]"] = {
                "auprc": round(float(average_precision_score(y, blended[test])), 4),
                "precision_at_k": precision_at_k(y, blended[test]),
                "delta_auprc": round(
                    float(average_precision_score(y, blended[test])) - base["auprc"], 4),
            }

    # The fitted blend, on the validation split only. Its context comes from
    # the same early rings, so for validation accounts the horizon is not the
    # clean one the held-out set enjoys - stated rather than hidden.
    val = split.val
    best_w, best_ap = 0.0, average_precision_score(split.y(val), scores[val])
    combo = feats["neighbours_in_previous_rings"]
    cmax = combo.max() or 1.0
    for w in np.linspace(0.0, 2.0, 21):
        s = scores * (1.0 + w * combo / cmax)
        ap = average_precision_score(split.y(val), s[val])
        if ap > best_ap:
            best_w, best_ap = float(w), float(ap)
    fitted = scores * (1.0 + best_w * combo / cmax)
    arms["fitted blend"] = {
        "weight": round(best_w, 3),
        "fitted_on": "validation split",
        "auprc": round(float(average_precision_score(y, fitted[test])), 4),
        "delta_auprc": round(
            float(average_precision_score(y, fitted[test])) - base["auprc"], 4),
        "caveat": ("The validation accounts' context comes from rings on the "
                   "same early window their own features are drawn from, so "
                   "their horizon is not the clean one the held-out accounts "
                   "have. The weight is the only thing fitted anywhere here."),
    }

    coverage = {
        c: {"accounts_nonzero": int((feats[c] > 0).sum()),
            "heldout_nonzero": int((feats[c][test] > 0).sum()),
            "heldout_share": round(float((feats[c][test] > 0).mean()), 4)}
        for c in CONTEXT_FEATURES
    }

    return {
        "horizon": ctx["horizon"],
        "accounts_in_previous_rings": ctx["accounts_in_previous_rings"],
        "coverage": coverage,
        "results": arms,
        "note": ("Context comes from rings on the early window and is joined to "
                 "late-window features, so every context day strictly precedes "
                 "every feature day. assert_horizon checks that against the "
                 "window manifests."),
    }


def measure_check_latency(cfg: Config, n_samples: int = 1000) -> dict:
    """How long one account's answer takes, once the indexes are loaded."""
    from orbweaver.console.check import CheckIndex

    idx = CheckIndex(cfg)
    rng = np.random.default_rng(cfg.seed)
    accounts = rng.integers(0, idx.n, size=n_samples)
    times = []
    for a in accounts:
        t0 = time.perf_counter()
        idx.check(int(a))
        times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "samples": int(n_samples),
        "p50_ms": round(float(np.percentile(times, 50)), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
        "p99_ms": round(float(np.percentile(times, 99)), 3),
        "max_ms": round(float(times.max()), 3),
        "note": "indexes are loaded once at startup; this is per-account lookup",
    }


def main() -> None:
    cfg = load_config()
    out = run(cfg)
    try:
        out["check_latency"] = measure_check_latency(cfg)
    except Exception as exc:                      # noqa: BLE001
        out["check_latency"] = {"error": f"{type(exc).__name__}: {exc}"}

    dest = cfg.abs_path(cfg.paths.processed) / "ring_context.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"accounts in a previous-window ring: {out['accounts_in_previous_rings']:,}")
    print(f"horizon: context {out['horizon']['context_days']} -> "
          f"features {out['horizon']['feature_days']}")
    print()
    print(f"{'arm':>52s} {'AUPRC':>8s} {'delta':>8s}")
    for name, r in out["results"].items():
        if "auprc" not in r:
            continue
        print(f"{name:>52s} {r['auprc']:>8} {str(r.get('delta_auprc', '')):>8}")
    lat = out.get("check_latency", {})
    if "p50_ms" in lat:
        print(f"\n/check latency over {lat['samples']:,} accounts: "
              f"p50 {lat['p50_ms']} ms, p95 {lat['p95_ms']} ms")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
