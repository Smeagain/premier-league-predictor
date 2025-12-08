INSTRUCTIONS

- Make sure .env contains FOOTBALL_API_KEY
- make setup  # creates venv and installs deps
- make test   # run unit tests
- make train  # train model
- make predict # list upcoming fixtures

Run the REST prediction server (after make setup):

- Start server locally: .venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
- Docker (build server image): docker build -f Dockerfile.server -t premier-league-predictor-server:latest .
- Docker (run): docker run --rm -e FOOTBALL_API_KEY=$$FOOTBALL_API_KEY -p 8000:8000 premier-league-predictor-server:latest

For development, use .venv/bin/python and .venv/bin/pip
