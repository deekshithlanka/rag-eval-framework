PY ?= python
export PYTHONPATH := src

.PHONY: install test smoke calibrate run combine report all clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

# Offline end-to-end check. No API key, no network.
smoke:
	$(PY) -m rageval.calibrate --mock
	$(PY) -m rageval.run --mock
	$(PY) -m rageval.report

calibrate:
	$(PY) -m rageval.calibrate

run:
	$(PY) -m rageval.run

combine:
	$(PY) -m rageval.run --combine

report:
	$(PY) -m rageval.report

all: calibrate run combine report

clean:
	rm -rf results .cache reports/calibration.* reports/eval_report.md
