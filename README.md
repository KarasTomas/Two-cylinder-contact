# Two Cylinder Contact

This project provides tools for automating the analysis of gear contact mechanics. By simplifying the complex geometry of gear teeth, it models the interaction as the contact between two cylinders with equivalent radii and involute profiles. This abstraction enables efficient computation and analysis of contact parameters such as pressure distribution, contact stresses, and deformation. The approach is particularly useful for preliminary design, optimization, and educational purposes, offering a balance between accuracy and computational simplicity.

The package also includes Python 2.7 scripts compatible with Abaqus CAE 2019. These scripts automate model generation, job execution, and results extraction, streamlining the workflow for finite element analysis within Abaqus.

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
