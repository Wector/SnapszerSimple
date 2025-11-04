# JAX Snapszer

This repository contains a JAX implementation of the Hungarian Snapszer card game.

## Setup

It is recommended to use a virtual environment to manage the dependencies. This project uses `uv` for environment and package management.

### Create a virtual environment with `uv`

```bash
uv venv
```

### Activate the virtual environment

**On macOS and Linux:**

```bash
source .venv/bin/activate
```

**On Windows:**

```bash
.venv\\Scripts\\activate
```

### Install dependencies

```bash
uv pip install -r requirements.txt
```

## Testing

A parity test is included to verify the correctness of the JAX implementation by comparing it against a base Python version.

### Run the parity test

```bash
python3 parity_test.py
```
