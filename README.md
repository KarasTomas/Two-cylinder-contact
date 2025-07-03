# __package_name__

__package_description__

## Development

### Environment Setup

```bash
uv venv
uv sync
.venv\Scripts\activate
```

[official installation guide for `uv`](https://github.com/astral-sh/uv#installation)

## Development Tools

some short text here

### Pre-commit

```bash
# Update pre-commit hooks
pre-commit autoupdate 
# Install pre-commit hooks 
pre-commit install
# Run pre-commit hooks on all files without committing
pre-commit run --all-files
```

### Pytest

```bash
# Run all tests
pytest
# Run tests in a specific file
pytest path/to/test_file.py
# Run a specific test function
pytest path/to/test_file.py::test_function
```

### Ruff

```bash
# Check for linting errors and formatting issues
ruff check .
# Format code
ruff format .
# Fix linting errors automatically (use with caution)
ruff check . --fix
```

### Mypy

```bash
# Run type checking
mypy .
```
