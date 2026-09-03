"""Typed configuration, loaded once from config/default.yaml.

Every threshold, seed and path comes from here. Nothing is hard-coded at a
call site, so a `make reproduce` run is a function of this one file plus the
raw data, and the seeds make it repeatable.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


class Paths(BaseModel):
    raw: Path
    processed: Path
    figures: Path
    results: Path


class DataCfg(BaseModel):
    week1_last_day: str
    week2_first_day: str
    buildable_relations: list[str]
    all_relations: list[str]


class GraphCfg(BaseModel):
    n_max: int
    n_max_sweep: list[int]
    rarity_base: float
    min_edge_weight: float
    # Whether burst-corrected time weighting is on. False everywhere the
    # standard pipeline runs; no code path in build_graph.py reads this field
    # at all, so flipping it changes nothing until something is wired to
    # check it. See orbweaver/data/lockstep.py.
    time_weighting: bool = False


class SubsampleCfg(BaseModel):
    seed_nodes: int
    closure_hops: int
    region_relation: str


class XGBCfg(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    n_jobs: int
    tree_method: str


class ScoringCfg(BaseModel):
    xgb: XGBCfg
    val_fraction: float
    calibration: str
    heldout_fraction: float


class RingsCfg(BaseModel):
    lambda_sweep: list[float]
    lambda_default: float
    k_min: int
    k_max: int
    top_k: int
    g_min: float
    prune_tau: float
    prune_tau_headline: float
    lambda_headline: float


class CostCfg(BaseModel):
    assumed_avg_promo_value_inr: float
    assumed_avg_order_value_inr: float
    assumed_reviewer_cost_per_minute_inr: float = 10.0
    review_minutes_fixed: float = 3.0
    review_minutes_per_member: float = 0.25


class Config(BaseModel):
    seed: int
    paths: Paths
    data: DataCfg
    graph: GraphCfg
    subsample: SubsampleCfg
    scoring: ScoringCfg
    rings: RingsCfg
    cost: CostCfg

    def abs_path(self, p: Path | str) -> Path:
        """Resolve a configured path against the repo root."""
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p


@lru_cache(maxsize=None)
def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    return Config(**yaml.safe_load(Path(path).read_text()))
