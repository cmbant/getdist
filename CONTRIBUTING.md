# Contributing to GetDist

Thank you for your interest in contributing to GetDist! This guide will help you set up your development environment and understand our coding standards.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- [uv](https://docs.astral.sh/uv/) (recommended) or pip for package management

### Installation for Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cmbant/getdist.git
   cd getdist
   ```

2. **Install in development mode with development dependencies:**

   Using uv (recommended):
   ```bash
   uv pip install -e ".[dev]"
   ```

   Using pip:
   ```bash
   pip install -e ".[dev]"
   ```

   This installs GetDist in editable mode along with development tools including:
   - `pre-commit` for code quality checks
   - `pytest` for testing

3. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Optional Dependencies

For specific development tasks, you may need additional dependencies:

- **GUI development:** `uv pip install -e ".[GUI]"` (adds PySide6)
- **Streamlit GUI:** `uv pip install -e ".[StreamlitGUI]"` (adds Streamlit)
- **Documentation:** `uv pip install -e ".[docs]"` (adds Sphinx and related tools)
- **All dependencies:** `uv pip install -e ".[dev,GUI,StreamlitGUI,docs]"`

## Code Standards

### Code Style

GetDist uses [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting. The configuration is defined in `pyproject.toml`:

- **Line length:** 120 characters
- **Quote style:** Double quotes
- **Target Python version:** 3.9+
- **Import sorting:** Enabled (isort-compatible)

### Pre-commit Hooks

Pre-commit hooks automatically run code quality checks before each commit:

- **Ruff linting** with auto-fix enabled
- **Ruff formatting** for consistent code style

The hooks are configured in `.pre-commit-config.yaml` and will:
- Automatically fix common issues
- Block commits if unfixable issues are found
- Ensure consistent code formatting across the project

### Running Code Quality Checks

You can manually run the code quality checks at any time:

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run ruff
pre-commit run ruff-format
```

## Testing

### Running Tests

GetDist includes unit tests to ensure code quality and functionality:

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest getdist/tests/getdist_test.py

# Run the basic unit test (legacy method)
python -m unittest getdist.tests.getdist_test
```

### Writing Tests

When adding new features or fixing bugs:

1. Add appropriate tests in the `getdist/tests/` directory
2. Ensure tests pass locally before submitting
3. Aim for good test coverage of new code

## Development Workflow

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code standards above

3. **Test your changes:**
   ```bash
   # Run tests
   python -m pytest

   # Run code quality checks
   pre-commit run --all-files
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

   The pre-commit hooks will run automatically and may modify files or block the commit if issues are found.

5. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Messages

Use clear, descriptive commit messages:
- Start with a brief summary (50 characters or less)
- Use the imperative mood ("Add feature" not "Added feature")
- Include more details in the body if necessary

### Pull Request Guidelines

- Ensure all tests pass
- Include tests for new functionality
- Update documentation if needed
- Keep changes focused and atomic
- Respond to code review feedback promptly


## Architecture Overview

GetDist has several main components:

- **Command line/Python API:** Core functionality for sample analysis
- **Qt GUI:** Desktop application (`getdist/gui/mainwindow.py`) using PySide6
- **Streamlit GUI:** Web-based interface (`getdist/gui/streamlit_app.py`)
- **Plotting library:** Publication-ready plotting tools
- **Sample analysis:** MCMC chain analysis and statistics

## Getting Help

- Check the [documentation](https://getdist.readthedocs.io/)
- Look at existing [issues](https://github.com/cmbant/getdist/issues)
- Ask questions in new issues with the "question" label
