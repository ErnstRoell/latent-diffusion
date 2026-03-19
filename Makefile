.PHONY: tests notebooks

tests:
	uv run coverage run -m pytest 
	uv run coverage html
