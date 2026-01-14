VENV=.venv
PY=.venv/bin/python
PIP=.venv/bin/pip

.PHONY: setup lint test pytest train predict docker-build docker-run clean report-e0 report-e1

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

lint:
	$(VENV)/bin/flake8

test:
	$(PY) -m unittest discover -s tests -v

pytest:
	$(PY) -m pytest -q

train:
	$(PY) main.py --train

predict:
	$(PY) main.py --predict

docker-build:
	docker build -t premier-league-predictor:latest .

docker-run:
	docker run --rm -e FOOTBALL_API_KEY=$$FOOTBALL_API_KEY premier-league-predictor:latest --help

report-e0:
	$(PY) scripts/dataset_report.py E0
	$(PY) scripts/baseline_wdl.py E0

report-e1:
	$(PY) scripts/dataset_report.py E1
	$(PY) scripts/baseline_wdl.py E1

clean:
	rm -rf $(VENV)
