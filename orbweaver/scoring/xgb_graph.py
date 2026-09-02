"""Account-level suspicion scoring: XGBoost over transaction and graph features.

This is the only learned component in the pipeline. It produces a calibrated
score s(v) in [0, 1] per account, and nothing else. It does not decide who is
in a ring — the peeling objective in `orbweaver.rings.peel` does that, and it
is deterministic and inspectable. Keeping the boundary there is the point: I
can tell an analyst exactly why an account ended up in a ring without
appealing to a model.

I chose gradient boosting over a GNN for the first version because GADBench
(NeurIPS 2023) found XGBoost on graph-aggregated features beats most GNNs on
real anomaly-detection graphs, it trains in a couple of minutes on a laptop,
and the ring extraction downstream is scorer-agnostic, so a better scorer can
be swapped in later without touching the part that produces the output.

**Training uses the early window of week 2; scoring uses the late window.**
The accounts in `split.test` are absent from both training and calibration,
so their scores are the honest measurement. See `eval.split` for why the
evaluation lives inside week 2 rather than across the two order files.

**Calibration matters more than usual.** The ring objective adds
`lambda * s(v)` to a sum of edge weights, so the scores are being *summed
against a physical quantity*, not just ranked. An uncalibrated margin would
make lambda meaningless. Isotonic regression is fitted on the validation
accounts, which are themselves excluded from training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

from eval.split import Split, make_split
from orbweaver.config import Config, load_config
from orbweaver.data.windows import EARLY, LATE
from orbweaver.features.node_features import FEATURE_NAMES


@dataclass
class ScoringResult:
    scores: np.ndarray            # calibrated s(v) for every user id
    raw_scores: np.ndarray        # uncalibrated model output
    feature_importance: dict[str, float]
    n_train: int
    n_val: int
    best_iteration: int | None
    # The fitted objects themselves. A nightly run has to apply the model it
    # already had to whatever it can see today; refitting per night would be
    # a different, easier problem and would leak the future into the past.
    model: object | None = None
    calibrator: object | None = None


def load_features(week: int, cfg: Config, n_users: int,
                  tag: str | None = None) -> np.ndarray:
    """Feature matrix aligned to user id, padded for ids absent that week."""
    proc = cfg.abs_path(cfg.paths.processed)
    suffix = f"_{tag}" if tag else ""
    t = pq.read_table(proc / f"features_week{week}{suffix}.parquet")
    uid = t["user_id"].to_numpy()
    X = np.zeros((n_users, len(FEATURE_NAMES)), dtype=np.float32)
    cols = np.column_stack([t[name].to_numpy() for name in FEATURE_NAMES])
    keep = uid < n_users
    X[uid[keep]] = cols[keep]
    return X


def fit_scorer(cfg: Config | None = None, split: Split | None = None) -> ScoringResult:
    import xgboost as xgb

    cfg = cfg or load_config()
    split = split or make_split(cfg)
    n_users = split.labels.size

    X_early = load_features(2, cfg, n_users, EARLY)   # training window
    y_train = split.y(split.train)
    y_val = split.y(split.val)

    params = cfg.scoring.xgb.model_dump()
    n_estimators = params.pop("n_estimators")
    # The labelled pool is 22.4% fraud, far from the 2.1% population rate,
    # because unlabelled accounts are excluded. Weight to the labelled prior.
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    model = xgb.XGBClassifier(
        **params,
        n_estimators=n_estimators,
        scale_pos_weight=neg / max(pos, 1),
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=cfg.seed,
        early_stopping_rounds=40,
    )
    model.fit(X_early[split.train], y_train,
              eval_set=[(X_early[split.val], y_val)], verbose=False)

    # Isotonic calibration on the validation accounts, which are held out of
    # training. The ring objective sums scores against edge weights, so these
    # need to be probabilities rather than an arbitrary monotone margin.
    raw_val = model.predict_proba(X_early[split.val])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_val, y_val)

    # Score on the LATE window: behaviour that happened after everything the
    # model was trained on, which is the situation at decision time.
    X_late = load_features(2, cfg, n_users, LATE)
    raw = model.predict_proba(X_late)[:, 1]
    calibrated = iso.predict(raw).astype(np.float32)

    importance = dict(sorted(
        zip(FEATURE_NAMES, model.feature_importances_.astype(float)),
        key=lambda kv: kv[1], reverse=True))

    return ScoringResult(
        scores=calibrated, raw_scores=raw.astype(np.float32),
        feature_importance={k: round(v, 5) for k, v in importance.items()},
        n_train=int(split.train.size), n_val=int(split.val.size),
        best_iteration=getattr(model, "best_iteration", None),
        model=model, calibrator=iso,
    )


def score_and_save(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    split = make_split(cfg)
    result = fit_scorer(cfg, split)

    dest = proc / "scores_week2.parquet"
    pq.write_table(pa.table({
        "user_id": pa.array(np.arange(result.scores.size, dtype=np.int32)),
        "score": pa.array(result.scores),
        "raw_score": pa.array(result.raw_scores),
    }), dest, compression="zstd")

    (proc / "scores_week2_manifest.json").write_text(json.dumps({
        "n_train": result.n_train, "n_val": result.n_val,
        "best_iteration": result.best_iteration,
        "seed": cfg.seed,
        "feature_importance": result.feature_importance,
    }, indent=2))

    save_scorer(result, cfg)
    return dest


def save_scorer(result: ScoringResult, cfg: Config | None = None) -> Path:
    """Write the model and its calibrator so they can be reapplied later.

    Anything that replays the window needs the model that was already fitted,
    not a fresh one. XGBoost's own JSON format keeps this readable and avoids
    a pickle; the isotonic calibrator is a monotone step function, so its knots
    are enough to rebuild it exactly.
    """
    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    proc.mkdir(parents=True, exist_ok=True)
    result.model.save_model(proc / "scorer.json")
    iso = result.calibrator
    (proc / "calibrator.json").write_text(json.dumps({
        "x": np.asarray(iso.X_thresholds_).tolist(),
        "y": np.asarray(iso.y_thresholds_).tolist(),
        "y_min": float(iso.y_min), "y_max": float(iso.y_max),
    }))
    return proc / "scorer.json"


def load_scorer(cfg: Config | None = None):
    """The persisted model and a callable that applies its calibration."""
    import xgboost as xgb

    cfg = cfg or load_config()
    proc = cfg.abs_path(cfg.paths.processed)
    model = xgb.XGBClassifier()
    model.load_model(proc / "scorer.json")
    knots = json.loads((proc / "calibrator.json").read_text())
    x, y = np.asarray(knots["x"]), np.asarray(knots["y"])

    def calibrate(raw: np.ndarray) -> np.ndarray:
        return np.clip(np.interp(raw, x, y), knots["y_min"], knots["y_max"])

    return model, calibrate


def score_features(model, calibrate, X: np.ndarray) -> np.ndarray:
    """Calibrated scores for a feature matrix, using an already-fitted model."""
    return calibrate(model.predict_proba(X)[:, 1]).astype(np.float32)
