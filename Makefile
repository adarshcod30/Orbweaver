# Every number in README.md is produced by `make reproduce`.
# None of them are typed in by hand.

PY := python3
.DEFAULT_GOAL := help
.PHONY: help setup download data graph subsample score rings eval adv console report reproduce test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## install python dependencies
	$(PY) -m pip install -r requirements.txt

download:  ## fetch the PPA release from OSF (~4.0 GB, resumable, md5-verified)
	./scripts/download_ppa.sh

schema:  ## regenerate data/ppa_schema_facts.json from the raw files
	$(PY) scripts/inspect_ppa.py

data:  ## raw CSV -> canonical parquet (temporal split applied here)
	$(PY) -m orbweaver.data.load_ppa

test:  ## schema + temporal-split-leak tests. must pass before any metric
	$(PY) -m pytest tests/ -q

clean:  ## remove processed data (raw is kept; re-downloading is slow)
	rm -rf data/processed

reproduce: data test  ## regenerate every number in the README
	@echo "reproduce: stages beyond 'data' land as they are built"
