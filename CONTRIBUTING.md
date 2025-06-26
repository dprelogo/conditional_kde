# Contributing to Conditional KDE

We love your input! We want to make contributing to Conditional KDE as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code follows the style guidelines.
6. Issue that pull request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/conditional_kde.git
cd conditional_kde

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=conditional_kde --cov-report=html

# Run specific test file
pytest tests/test_gaussian.py

# Run tests in multiple Python versions
tox
```

## Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **isort** for import sorting
- **mypy** for type checking

Pre-commit hooks will automatically format your code. You can also run manually:

```bash
# Format code
black conditional_kde tests

# Check linting
flake8 conditional_kde tests

# Sort imports
isort conditional_kde tests

# Type checking
mypy conditional_kde
```

## Pull Request Process

1. Update the README.rst with details of changes to the interface, if applicable.
2. Update the docs with any new functionality.
3. The PR will be merged once you have the sign-off of at least one maintainer.

## Commit Messages

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, missing semicolons, etc)
- `refactor:` Code changes that neither fix bugs nor add features
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Releasing

Releases are automated via GitHub Actions:

1. Update version using `bump2version`:
   ```bash
   bump2version patch  # or minor, major
   ```
2. Push tags: `git push --tags`
3. GitHub Actions will automatically publish to PyPI

## Any contributions you make will be under the MIT License

When you submit code changes, your submissions are understood to be under the same [MIT License](LICENSE) that covers the project.
