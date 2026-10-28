.PHONY: install sample test train

install:
	python -m pip install -e ".[dev]"

sample:
	python scripts/generate_synthetic_data.py

test:
	pytest -q

train:
	python -m seva_model.train --data data/guest_visits.csv
