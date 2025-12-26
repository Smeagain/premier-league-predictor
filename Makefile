VENV=.venv
PY=.venv/bin/python
PIP=.venv/bin/pip

.PHONY: setup lint test train predict docker-build docker-run clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

lint:
	$(VENV)/bin/flake8

test:
	$(PY) -m unittest discover -s tests -v

train:
	$(PY) main.py --train

predict:
	$(PY) main.py --predict

docker-build:
	docker build -t premier-league-predictor:latest .

docker-run:
	docker run --rm -e FOOTBALL_API_KEY=$$FOOTBALL_API_KEY premier-league-predictor:latest --help

clean:
	rm -rf $(VENV)
