# Animax Tests

This directory contains tests for the Animax package.

## Running Tests

To run all tests:

```bash
pytest
```

To run a specific test file:

```bash
pytest tests/test_anima.py
```

To run a specific test:

```bash
pytest tests/test_anima.py::test_anima_creation_with_default_name
```

To run tests with verbose output:

```bash
pytest -v
``` 