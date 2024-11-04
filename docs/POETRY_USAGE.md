# Poetry Usage Guide

This guide provides instructions on how to use [Poetry](https://python-poetry.org/) to manage dependencies, install packages, and prepare your project for both development and production environments.

## Table of Contents

- [Overview](#overview)
- [Installing Poetry](#installing-poetry)
- [Using Poetry in Development](#using-poetry-in-development)
  - [Installing Dependencies](#installing-dependencies)
  - [Updating Dependencies](#updating-dependencies)
  - [Adding and Removing Dependencies](#adding-and-removing-dependencies)
  - [Synchronizing Dependencies](#synchronizing-dependencies)
- [Using Poetry in Production](#using-poetry-in-production)
  - [Locking Dependencies](#locking-dependencies)
  - [Installing from `poetry.lock`](#installing-from-poetrylock)
- [Poetry Commands Summary](#poetry-commands-summary)

---

## Overview

Poetry is a dependency manager and build tool for Python projects. It simplifies managing dependencies, creating virtual environments, and ensuring version consistency between development and production environments. Poetry relies on two files:

- **`pyproject.toml`**: Defines the dependencies and configuration.
- **`poetry.lock`**: Locks dependencies to specific versions to ensure consistency.

---

## Installing Poetry(macOS only)

To install Poetry, use the following command:

```bash
brew install poetry
```

Refer to the [Poetry documentation](https://python-poetry.org/docs/#installation) for more options and OS-specific installation instructions.

---

## Using Poetry in Development

### Installing Dependencies

In development, install dependencies specified in `pyproject.toml`:

1. Navigate to the project directory:

   ```bash
   cd path/to/project
   ```

2. Run:
   ```bash
   poetry install
   ```

This command creates a virtual environment, installs all dependencies, and ensures they are compatible with the Python version specified.

### Updating Dependencies

During development, you can update dependencies by editing `pyproject.toml` directly and then running:

```bash
poetry install
```

This will apply any changes and update the environment without manually adding each dependency.

### Adding and Removing Dependencies

- **Add a New Dependency**:

  ```bash
  poetry add <package-name>
  ```

  Example:

  ```bash
  poetry add requests
  ```

- **Add a Development Dependency** (only used for development/testing):

  ```bash
  poetry add --group dev <package-name>
  ```

  Example:

  ```bash
  poetry add --group dev pytest
  ```

- **Remove a Dependency**:
  ```bash
  poetry remove <package-name>
  ```

### Synchronizing Dependencies

If the `pyproject.toml` or `poetry.lock` files are updated (e.g., after pulling changes), run:

```bash
poetry install
```

This keeps your environment synchronized with any updates made to the dependency files.

---

## Using Poetry in Production

### Locking Dependencies

To lock dependencies for production use, run:

```bash
poetry lock
```

This creates or updates `poetry.lock`, which pins each dependency to a specific version. This lock file should be used to maintain consistency in production.

### Installing from `poetry.lock`

In production, use `poetry.lock` to ensure exact dependency versions:

1. Install only the required (non-development) dependencies:
   ```bash
   poetry install --no-dev
   ```

This ensures that dependencies are installed exactly as defined in `poetry.lock`.

---

## Poetry Commands Summary

| Command                        | Description                                                   |
| ------------------------------ | ------------------------------------------------------------- |
| `poetry install`               | Installs dependencies from `pyproject.toml` or `poetry.lock`. |
| `poetry add <package-name>`    | Adds a new dependency and updates `pyproject.toml`.           |
| `poetry add --group dev <pkg>` | Adds a development-only dependency.                           |
| `poetry remove <package-name>` | Removes a dependency and updates `pyproject.toml`.            |
| `poetry update`                | Updates all dependencies to their latest compatible versions. |
| `poetry lock`                  | Locks dependencies to specific versions for production.       |
| `poetry shell`                 | Activates the Poetry-managed virtual environment.             |

---

## Additional Resources

- **Poetry Documentation**: [https://python-poetry.org/docs/](https://python-poetry.org/docs/)
- **GitHub Repository**: [https://github.com/python-poetry/poetry](https://github.com/python-poetry/poetry)

For further help, please refer to the [Poetry documentation](https://python-poetry.org/docs/).
