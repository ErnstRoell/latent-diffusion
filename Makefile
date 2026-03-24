.PHONY: tests notebooks

tests:
	uv run coverage run --source=src -m pytest
	uv run coverage report -m
	uv run coverage html

