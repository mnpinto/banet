SRC = $(wildcard banet/*.py)

all: test

test:
	python -m pytest --tb=short -q

release: pypi

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist