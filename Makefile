.PHONY: clean build lint docs install

clean: # clean all build, python, and testing files
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .tox/
	rm -fr .coverage
	rm -fr coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache

build: # run tests
	pytest

lint: # run formatting and linting
	black -l 88 .
	isort -l 88 .
	ruff .
	mypy --disallow-untyped-calls --disallow-untyped-defs --disallow-incomplete-defs .

docs: # generates documentation
	make -C docs html

install: # installs dependencies
ifdef DEV
	conda env update --file environment.yml --prune
	pdm sync
else
	conda env update --file environment.yml --prune
	pip install -r requirements.txt
	pip install -e .
endif
