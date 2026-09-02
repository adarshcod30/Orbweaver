# Every number in README.md and docs/results.md is produced by `make reproduce`.
# None of them are typed in by hand.

PY := python3
.DEFAULT_GOAL := help
.PHONY: help setup download schema data graph windows weights features subsample \
        score rings hostel views adversarial overlay report reproduce test clean

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
	$(PY) -m orbweaver.data.windows

subsample:  ## entity-anchored development subsample
	$(PY) -m orbweaver.data.subsample

score:  ## train the account scorer and report detection numbers
	$(PY) -u -m eval.score_report

rings:  ## extract rings across the tau/lambda grid and measure them
	$(PY) -u -m eval.run_rings

hostel:  ## check the pipeline against legitimate co-located groups
	$(PY) -u -m orbweaver.rings.hostel_test

adversarial:  ## fragmentation curve and multi-round duplication
	$(PY) -u -m orbweaver.adversarial.fragment
	$(PY) -u -m orbweaver.adversarial.duplicate

overlay:  ## simulated payment-instrument relation, sensitivity analysis
	$(PY) -u -m orbweaver.aggregator.instrument_overlay

views:  ## my 5-relation graph against the authors' 8-relation graph
	$(PY) -u -m eval.compare_views

sage:  ## optional GraphSAGE scorer, reported beside the default one
	$(PY) -u -m orbweaver.scoring.sage

report:  ## regenerate docs/results.md, the figures and the case-file page
	$(PY) -u -m eval.report
	$(PY) -u -m eval.case_report

test:  ## schema, temporal-split and planted-ring tests
	$(PY) -m pytest tests/ -q

# The full path from raw files to the numbers in the documentation.
# `weights` is fitted before `windows` because the graph applies it.
reproduce: data graph weights windows test score rings hostel views adversarial overlay report  ## everything, end to end
	@echo
	@echo "reproduce complete. See docs/results.md"

clean:  ## remove processed data; raw is kept because re-downloading is slow
	rm -rf data/processed
