"""Quick test for GPU CFR - small batches for fast compilation."""

import time
import jax

from training.neural_cfr_pure_gpu import PureGPUCFRTrainer, PureGPUCFRConfig

print("=" * 60)
print("TESTING GPU CFR WITH SMALL BATCHES")
print("=" * 60)

# Small configuration for quick testing
config = PureGPUCFRConfig(
    num_iterations=3,
    games_per_batch=256,  # Small batch for fast compilation
    train_batches_per_iter=10,  # Few batches
    hidden_sizes=(256, 128),  # Smaller network
    learning_rate=1e-3,
)

print(f"Iterations:           {config.num_iterations}")
print(f"Games per batch:      {config.games_per_batch}")
print(f"Train batches/iter:   {config.train_batches_per_iter}")
print(f"Network:              {config.hidden_sizes}")
print("=" * 60)
print()

# Initialize trainer
rng_key = jax.random.PRNGKey(42)

print("Initializing (JIT compilation on first run)...")
trainer = PureGPUCFRTrainer(config, rng_key)

print(f"Device: {jax.devices()[0]}")
print()

start_time = time.time()

# Train
print("Training 3 iterations...")
for i in range(1, 4):
    iter_start = time.time()
    trainer.train(1)
    iter_time = time.time() - iter_start

    if i == 1:
        print(f"  Iteration 1: {iter_time:.1f}s (includes JIT compilation)")
    else:
        print(f"  Iteration {i}: {iter_time:.2f}s")

total_time = time.time() - start_time

print()
print("=" * 60)
print("SUCCESS!")
print("=" * 60)
print(f"Total time:   {total_time:.1f}s")
print(f"Training ran on GPU with outcome-based CFR learning!")
print()
print("Next steps:")
print("  - Run train_gpu_maxed.py for full GPU utilization")
print("  - Monitor GPU with: nvidia-smi")
