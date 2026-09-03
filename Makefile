# Every number in README.md and docs/results.md is produced by `make reproduce`.
# None of them are typed in by hand.

PY := python3
.DEFAULT_GOAL := help
.PHONY: help setup download schema data graph windows weights features subsample \
        windows-weighted score sage rings rings-deep hostel views adversarial overlay generalise \
        download-gadbench download-ieee-cis ieee-cis merchant-view replay \
        ring-scorer ring-context twins \
        report reproduce reproduce-core test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

setup:  ## install python dependencies
	$(PY) -m pip install -r requirements.txt

download:  ## fetch PPA from OSF (~4.0 GB, resumable, md5-verified)
	./scripts/download_ppa.sh

schema:  ## measure the raw files into data/ppa_schema_facts.json
	$(PY) scripts/inspect_ppa.py

data:  ## raw CSV -> canonical parquet, week cut applied on order_time
	$(PY) -m orbweaver.data.load_ppa

graph:  ## multi-relation account graph for both weeks
	$(PY) -m orbweaver.data.build_graph

features:  ## per-account transaction and graph features
	$(PY) -m orbweaver.features.node_features

weights:  ## fit per-relation edge weights on training accounts only
	$(PY) -m orbweaver.data.relation_weights

windows:  ## split week 2 into early/late and build a graph and features for each
	$(PY) -u -m orbweaver.data.windows

# The relation weights are fitted FROM the early-window graph, and that graph
# is then rebuilt WITH them applied. That is circular, so it is resolved by
# building once with neutral weights, fitting, and rebuilding. Both steps live
# in one target because make would otherwise run `windows` only once.
windows-weighted:  ## bootstrap the windows, fit relation weights, rebuild
	$(PY) -u -m orbweaver.data.windows
	$(PY) -u -m orbweaver.data.relation_weights
	$(PY) -u -m orbweaver.data.windows

subsample:  ## entity-anchored development subsample
	$(PY) -m orbweaver.data.subsample

score:  ## train the account scorer and report detection numbers
	$(PY) -u -m eval.score_report

rings:  ## extract rings across the tau/lambda grid and measure them
	$(PY) -u -m eval.run_rings

rings-deep:  ## a deeper pass at the chosen operating point, for precision@K
	ORBWEAVER_TOP_K=200 ORBWEAVER_TAU=0.5 ORBWEAVER_LAMBDA=5.0 \
	  ORBWEAVER_OUT=ring_report_deep.json $(PY) -u -m eval.run_rings

hostel:  ## check the pipeline against legitimate co-located groups
	$(PY) -u -m orbweaver.rings.hostel_test

adversarial:  ## fragmentation curve and multi-round duplication
	$(PY) -u -m orbweaver.adversarial.fragment
	$(PY) -u -m orbweaver.adversarial.duplicate

overlay:  ## simulated payment-instrument relation, sensitivity analysis
	$(PY) -u -m orbweaver.aggregator.instrument_overlay

views:  ## my 5-relation graph against the authors' 8-relation graph
	$(PY) -u -m eval.compare_views

download-gadbench:  ## Amazon and YelpChi from the CARE-GNN release (~44 MB)
	./scripts/download_gadbench.sh

generalise:  ## run the whole pipeline on Amazon and YelpChi
	$(PY) -u -m eval.generalise

merchant-view:  ## what one business sees against what the platform sees
	$(PY) -u -m eval.merchant_view

replay:  ## replay the scoring window one night at a time
	$(PY) -u -m eval.replay

ring-scorer:  ## learn a confidence per ring and compare it to density
	$(PY) -u -m eval.ring_confidence

ring-context:  ## feed ring context back into the account score, and time /check
	$(PY) -u -m eval.ring_context_eval

twins:  ## behaviour edges an attacker cannot cut by fragmenting
	$(PY) -u -m orbweaver.adversarial.twins

download-ieee-cis:  ## IEEE-CIS from Kaggle (~1.2 GB; needs accepted rules)
	./scripts/download_ieee_cis.sh

ieee-cis:  ## run the pipeline on a payment processor's transactions
	$(PY) -u -m eval.ieee_cis_run

sage:  ## optional GraphSAGE scorer, reported beside the default one
	$(PY) -u -m orbweaver.scoring.sage

console:  ## review queue at http://127.0.0.1:8000 (needs fastapi, uvicorn)
	$(PY) -m orbweaver.console.app

report:  ## regenerate docs/results.md, the figures and the case-file page
	$(PY) -u -m eval.report
	$(PY) -u -m eval.case_report

test:  ## schema, temporal-split and planted-ring tests
	$(PY) -m pytest tests/ -q

check:  ## tests plus the pre-push check on committed prose
	$(PY) -m pytest tests/ -q
	./scripts/voice_check.sh

# Every stage from the raw files to the numbers in the documentation.
reproduce-core: schema data graph windows-weighted test score sage rings rings-deep hostel views adversarial overlay generalise report  ## the core pipeline on its own, about 50 min

# `report` is a recipe step here, not a prerequisite. It is already a
# prerequisite of reproduce-core, and make builds any target at most once per
# run, so naming it again in this list is a silent no-op: the report gets
# written before the six stages above it have produced anything, and the
# sections that depend on them are quietly missing. Running it twice costs a
# couple of minutes and is worth it.
reproduce: reproduce-core merchant-view replay ring-scorer ring-context twins ieee-cis  ## everything, end to end
	$(PY) -u -m eval.report
	@echo
	@echo "reproduce complete. See docs/results.md"

clean:  ## remove processed data; raw is kept because re-downloading is slow
	rm -rf data/processed
