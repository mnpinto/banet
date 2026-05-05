SRC = $(wildcard banet/*.py)

all: setup

syslibs:
	sudo apt-get install -y libhdf4-dev libproj-dev proj-data proj-bin libgeos-dev

setup: syslibs
	python -m venv .venv
	.venv/bin/pip install -e .

test:
	python -m pytest --tb=short -q

release: pypi

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python -m build

clean:
	rm -rf dist