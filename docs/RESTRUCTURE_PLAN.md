# Project Restructure Plan

## Current Issues
1. All files in root directory (messy)
2. No clear separation between source, tests, and benchmarks
3. Test files have inconsistent naming
4. No package structure for imports
5. Documentation scattered

## Proposed Structure

```
SnapszerSimple/
├── src/
│   └── snapszer/                    # Main package
│       ├── __init__.py              # Package exports
│       ├── base.py                  # Base Python implementation
│       ├── jax_impl.py              # JAX implementation (parity-tested)
│       ├── jax_optimized.py         # Optimized JAX for training
│       └── utils/
│           ├── __init__.py
│           └── mt19937.py           # MT19937 utilities (if needed separately)
│
├── tests/                           # Unit/integration tests
│   ├── __init__.py
│   ├── conftest.py                  # Pytest configuration
│   ├── test_parity.py               # Parity tests (main correctness test)
│   ├── test_base.py                 # Tests for base implementation
│   ├── test_jax.py                  # Tests for JAX implementation
│   ├── test_mt19937.py              # MT19937 RNG tests
│   └── debug/                       # Debug-specific tests
│       ├── __init__.py
│       ├── test_seed_55.py          # Specific seed debugging
│       ├── test_mt_state.py
│       └── test_mt_extract.py
│
├── benchmarks/                      # Performance benchmarks
│   ├── __init__.py
│   ├── bench_gpu_setup.py           # GPU verification
│   ├── bench_batch_speed.py         # Batch performance
│   ├── bench_optimization.py        # Original vs optimized comparison
│   ├── bench_profile.py             # Detailed profiling
│   └── utils.py                     # Benchmark utilities
│
├── examples/                        # Example usage scripts
│   ├── basic_game.py                # Play a simple game
│   ├── batch_simulation.py          # Run batched games
│   └── visualize_game.py            # Visualize game state
│
├── training/                        # Training scripts (future)
│   ├── __init__.py
│   ├── sf_ocr.py                    # SF-OCR algorithm
│   ├── config.py                    # Training configuration
│   └── train.py                     # Main training script
│
├── docs/                            # Documentation
│   ├── README.md                    # Main documentation
│   ├── OPTIMIZATION.md              # Optimization details
│   ├── API.md                       # API documentation
│   └── TRAINING.md                  # Training guide
│
├── scripts/                         # Utility scripts
│   ├── run_parity_tests.sh          # Quick parity check
│   ├── run_benchmarks.sh            # Run all benchmarks
│   └── setup_gpu.sh                 # GPU setup verification
│
├── pyproject.toml                   # Project configuration
├── README.md                        # Project overview
├── .gitignore
└── LICENSE

```

## Benefits

### 1. **Clear Separation of Concerns**
- Source code in `src/snapszer/`
- Tests in `tests/`
- Benchmarks in `benchmarks/`
- Future training code in `training/`

### 2. **Better Imports**
```python
# Instead of:
import snapszer_jax as jax_impl

# Clean imports:
from snapszer import jax_impl
from snapszer import jax_optimized
from snapszer.base import SnapszerState
```

### 3. **Easy Installation**
```bash
# Install as editable package
pip install -e .

# Or for users:
pip install snapszer
```

### 4. **Better Testing**
```bash
# Run all tests
pytest tests/

# Run only parity tests
pytest tests/test_parity.py

# Run benchmarks
pytest benchmarks/ --benchmark-only
```

### 5. **Clear Documentation**
- API docs in `docs/API.md`
- Optimization details in `docs/OPTIMIZATION.md`
- Training guide in `docs/TRAINING.md`

## Migration Steps

1. Create directory structure
2. Move source files to `src/snapszer/`
3. Update imports in all files
4. Move tests to `tests/`
5. Move benchmarks to `benchmarks/`
6. Create `__init__.py` files with proper exports
7. Update `pyproject.toml` for package installation
8. Create README.md with usage examples
9. Add .gitignore for Python projects
10. Run tests to verify everything still works

## File Mapping

### Source Files
- `snapszer_base.py` → `src/snapszer/base.py`
- `snapszer_jax.py` → `src/snapszer/jax_impl.py`
- `snapszer_jax_optimized.py` → `src/snapszer/jax_optimized.py`

### Test Files
- `test_parity.py` → `tests/test_parity.py`
- `test_mt19937.py` → `tests/test_mt19937.py`
- `test_seed_55.py` → `tests/debug/test_seed_55.py`
- `test_mt_state.py` → `tests/debug/test_mt_state.py`
- `test_mt_extract.py` → `tests/debug/test_mt_extract.py`

### Benchmark Files
- `test_gpu_setup.py` → `benchmarks/bench_gpu_setup.py`
- `test_batch_speed.py` → `benchmarks/bench_batch_speed.py`
- `test_quick_comparison.py` → `benchmarks/bench_optimization.py`
- `test_optimization_speedup.py` → `benchmarks/bench_optimization_full.py`
- `profile_jax.py` → `benchmarks/bench_profile.py`

### Documentation
- `OPTIMIZATION_SUMMARY.md` → `docs/OPTIMIZATION.md`

## Updated pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "snapszer"
version = "0.1.0"
description = "Hungarian Snapszer card game with JAX implementation for Nash Equilibrium training"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.20.0",
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
]

[project.optional-dependencies]
gpu = [
    "jax[cuda12]>=0.4.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-benchmark>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"

[tool.black]
line-length = 100
target-version = ['py310']

[tool.ruff]
line-length = 100
target-version = "py310"
```

## Advantages for Future Work

### For Nash Equilibrium Training:
```python
from snapszer.jax_optimized import new_game, apply_action
from training.sf_ocr import SFOCRTrainer

# Clean, organized imports
trainer = SFOCRTrainer(config)
trainer.train()
```

### For Development:
```bash
# Install in dev mode
pip install -e ".[dev,gpu]"

# Run tests
pytest tests/ -v

# Run benchmarks
python benchmarks/bench_batch_speed.py

# Check code style
black src/ tests/
ruff check src/ tests/
```

### For Distribution:
```bash
# Build package
python -m build

# Upload to PyPI (if desired)
twine upload dist/*
```

## Next Steps

Would you like me to:
1. Execute this restructure automatically?
2. Create the structure incrementally with your review?
3. Focus on specific parts first (e.g., just source → package)?
