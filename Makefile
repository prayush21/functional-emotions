VENV := .venv/bin

.PHONY: help test lint run cloud-run fetch register

help:
	@echo "Available targets:"
	@echo "  make test                 Run the test suite (pytest)"
	@echo "  make lint                 Run ruff check on the whole repo"
	@echo "  make run P=51             Run a local experiment (fe-prototype\$$P);"
	@echo "                            override config with CONFIG=..., pass"
	@echo "                            STAGE=... for prototype1/25"
	@echo "  make cloud-run P=51       Launch a Modal cloud run for prototype P;"
	@echo "                            override CONFIG=... and GPU=... (default T4)"
	@echo "  make fetch P=51           Fetch a completed cloud run for prototype P"
	@echo "  make register P=51       Fetch and register a completed cloud run"

test:
	$(VENV)/pytest

lint:
	$(VENV)/ruff check .

# prototype1 and prototype25 require --stage; pass STAGE=<stage> to supply it.
run:
	$(VENV)/fe-prototype$(P) --config $(or $(CONFIG),configs/prototype$(P).yaml) $(if $(STAGE),--stage $(STAGE),)

cloud-run:
	modal run cloud/modal_run.py --prototype $(P) --config $(or $(CONFIG),configs/prototype$(P).yaml) --gpu $(or $(GPU),T4)

fetch:
	python cloud/fetch_run.py --prototype $(P)

register:
	python cloud/fetch_run.py --prototype $(P) --register
