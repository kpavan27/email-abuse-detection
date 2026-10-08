.PHONY: install data run test lint serve docker

install:
	pip install -e ".[dev]"

data:
	abuse-detect prepare

run:
	abuse-detect run

test:
	pytest

lint:
	ruff check src tests && ruff format --check src tests

serve:
	uvicorn abuse_detect.serve:app --host 0.0.0.0 --port 8000

docker:
	docker build -t abuse-detect .
