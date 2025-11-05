# Snapszer

Hungarian Snapszer (Snapser) card game implementation with GPU-accelerated JAX for Nash Equilibrium training.

## Overview

This project provides:
- **Pure Python implementation** (`snapszer.base`) - Reference implementation
- **JAX implementation** (`snapszer.jax_impl`) - GPU-accelerated, parity-tested with base
- **Optimized JAX** (`snapszer.jax_optimized`) - ~3-40x faster for training

## Features

- ✅ Complete Hungarian Snapszer rules implementation
- ✅ GPU acceleration via JAX
- ✅ Comprehensive parity tests (1000+ seeds tested)
- ✅ High-performance batched execution (~125k games/sec on GPU)
- ✅ Ready for Nash Equilibrium training with SF-OCR

## Installation

### Basic Installation
```bash
pip install -e .
```

### With GPU Support
```bash
pip install -e ".[gpu]"
```

### Development Installation
```bash
pip install -e ".[dev,gpu]"
```

## Quick Start

### Play a Simple Game

```python
from snapszer import jax_optimized
import jax.random as random

# Initialize game
key = random.PRNGKey(42)
state = jax_optimized.new_game(key)

# Game loop
while not state.terminal:
    # Get legal actions
    legal_mask = jax_optimized.legal_actions_mask(state)

    # Pick first legal action (for demo)
    action = jax.numpy.argmax(legal_mask)

    # Apply action
    state = jax_optimized.apply_action(state, action)

# Get game results
returns = jax_optimized.returns(state)
print(f"Game finished! Returns: {returns}")
```

### Batched Execution (Fast!)

```python
import jax
from snapszer import jax_optimized

@jax.jit
def run_batch(keys):
    # Initialize 1000 games in parallel
    states = jax.vmap(jax_optimized.new_game)(keys)

    # Play one step for all games
    def step(state):
        mask = jax_optimized.legal_actions_mask(state)
        action = jax.numpy.argmax(mask)
        return jax_optimized.apply_action(state, action)

    return jax.vmap(step)(states)

# Run 1000 games in parallel
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 1000)
states = run_batch(keys)

print(f"Batch executed! {states.terminal.sum()} games finished")
```

## Performance

### Benchmarks (NVIDIA GPU)

| Operation | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| new_game | 25ms | 0.6ms | **41.8x** |
| legal_actions_mask | 0.22ms | 0.06ms | **3.6x** |
| Full game | 67ms | 31ms | **2.15x** |
| Batch (1000 games) | 0.028ms/game | 0.0015ms/game | **18.8x** |

**Throughput**: ~125,000 games/second with batch_size=10,000

See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) for details.

## Project Structure

```
snapszer/
├── src/snapszer/          # Main package
│   ├── base.py            # Pure Python implementation
│   ├── jax_impl.py        # JAX (parity-tested)
│   └── jax_optimized.py   # Optimized JAX (for training)
│
├── tests/                 # Parity and unit tests
├── benchmarks/            # Performance benchmarks
├── examples/              # Usage examples
├── training/              # Training scripts (future: SF-OCR)
└── docs/                  # Documentation
```

## Testing

### Run All Tests
```bash
pytest tests/
```

### Run Parity Tests Only
```bash
pytest tests/test_parity.py -v
```

### Run Specific Test
```bash
pytest tests/test_parity.py::test_1000_seeds -v
```

## Benchmarks

### GPU Setup Verification
```bash
python benchmarks/bench_gpu_setup.py
```

### Batch Speed Test
```bash
python benchmarks/bench_batch_speed.py
```

### Optimization Comparison
```bash
python benchmarks/bench_optimization.py
```

## Game Rules

Hungarian Snapszer is a 2-player trick-taking card game played with 20 cards (A, 10, K, Q, J in 4 suits).

**Key rules:**
- **Marriages**: K+Q of same suit scores 20 (40 for trump)
- **Trump exchange**: Jack of trump can be exchanged with face-up trump card
- **Talon closing**: Leading player can close the talon for strict follow rules
- **Strict rules**: When talon closed/empty, must follow suit and beat if possible
- **Scoring**: First to 66 points wins (1-3 game points based on opponent's score)

See [snapszer.base.SnapszerState](src/snapszer/base.py) for complete implementation.

## Development

### Code Style
```bash
black src/ tests/ benchmarks/
ruff check src/ tests/ benchmarks/
```

### Run Profiling
```bash
python benchmarks/bench_profile.py
```

## Implementation Comparison

### Use `jax_impl` for:
- ✅ Parity testing with base implementation
- ✅ Correctness verification
- ✅ When deterministic RNG is required

### Use `jax_optimized` for:
- ✅ Training (Nash Equilibrium, SF-OCR)
- ✅ Large-scale simulations
- ✅ Maximum performance

**Key differences:**
- `jax_impl`: MT19937 RNG, sorted hands, exact parity with base
- `jax_optimized`: JAX native RNG, unsorted hands, ~3-40x faster

## Future Work

- [ ] Implement SF-OCR algorithm for Nash Equilibrium
- [ ] Add policy gradient training
- [ ] Create interactive web UI
- [ ] Add bot evaluation framework
- [ ] Publish pre-trained Nash policy

## License

MIT

## Contributing

Contributions welcome! Please ensure:
1. All tests pass: `pytest tests/`
2. Code is formatted: `black src/ tests/`
3. Linting passes: `ruff check src/ tests/`

## Citation

If you use this implementation in research, please cite:

```bibtex
@software{snapszer2025,
  title = {Snapszer: GPU-Accelerated Hungarian Snapszer for Nash Equilibrium Training},
  author = {Wector},
  year = {2025},
  url = {https://github.com/yourusername/SnapszerSimple}
}
```
