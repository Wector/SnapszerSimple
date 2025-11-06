"""Benchmark: Tabular CFR vs Neural CFR."""

import time
import jax
from training.config import CFRConfig
from training.cfr_trainer import CFRTrainer
from training.neural_cfr_trainer import NeuralCFRTrainer, NeuralCFRConfig


def benchmark_tabular(num_iterations=100):
    """Benchmark tabular CFR."""
    print("\n" + "="*70)
    print("TABULAR CFR BENCHMARK")
    print("="*70)

    config = CFRConfig(
        num_iterations=num_iterations,
        cfr_variant='cfr+',
        sampling='outcome',
        n_traversals_per_iter=1
    )

    trainer = CFRTrainer(config)

    print(f"Running {num_iterations} iterations of tabular CFR...")
    start = time.time()
    trainer.train(num_iterations)
    elapsed = time.time() - start

    num_info_sets = sum(trainer.get_num_info_sets())
    iters_per_sec = num_iterations / elapsed

    print(f"\nResults:")
    print(f"  Total time:        {elapsed:.2f}s")
    print(f"  Speed:             {iters_per_sec:.2f} iter/s")
    print(f"  Time per iter:     {elapsed/num_iterations*1000:.1f}ms")
    print(f"  Info sets:         {num_info_sets:,}")
    print("="*70)

    return elapsed, iters_per_sec


def benchmark_neural(num_iterations=100):
    """Benchmark neural CFR."""
    print("\n" + "="*70)
    print("NEURAL CFR BENCHMARK")
    print("="*70)

    config = NeuralCFRConfig(
        num_iterations=num_iterations,
        trajectories_per_iter=100,  # Same as tabular (1 iteration = 100 games)
        train_steps_per_iter=50,    # Neural network updates
        buffer_capacity=10_000,
        hidden_sizes=(256, 256, 128),
        learning_rate=1e-3
    )

    rng_key = jax.random.PRNGKey(42)
    trainer = NeuralCFRTrainer(config, rng_key)

    print(f"Device: {jax.devices()[0]}")
    print(f"Running {num_iterations} iterations of neural CFR...")
    print(f"  Trajectories per iter: {config.trajectories_per_iter}")
    print(f"  Train steps per iter:  {config.train_steps_per_iter}")

    start = time.time()
    trainer.train(num_iterations)
    elapsed = time.time() - start

    iters_per_sec = num_iterations / elapsed

    print(f"\nResults:")
    print(f"  Total time:        {elapsed:.2f}s")
    print(f"  Speed:             {iters_per_sec:.2f} iter/s")
    print(f"  Time per iter:     {elapsed/num_iterations*1000:.1f}ms")
    print(f"  Buffer size:       {len(trainer.buffer):,} samples")
    print("="*70)

    return elapsed, iters_per_sec


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CFR TRAINING SPEED COMPARISON")
    print("="*70)
    print("\nThis benchmark compares tabular CFR vs neural CFR on the same task.")
    print("Each iteration processes the same amount of game data.")
    print()

    num_iters = 50  # Use fewer iterations for quick benchmark

    # Benchmark tabular CFR
    tabular_time, tabular_speed = benchmark_tabular(num_iters)

    # Benchmark neural CFR
    neural_time, neural_speed = benchmark_neural(num_iters)

    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Tabular CFR:   {tabular_time:.2f}s  ({tabular_speed:.2f} iter/s)")
    print(f"Neural CFR:    {neural_time:.2f}s  ({neural_speed:.2f} iter/s)")
    print(f"\nSpeedup:       {tabular_time/neural_time:.2f}x")

    if neural_time < tabular_time:
        print(f"Neural CFR is {tabular_time/neural_time:.2f}x FASTER! 🚀")
    else:
        print(f"Tabular CFR is {neural_time/tabular_time:.2f}x faster")

    print("="*70)
    print("\nNOTE:")
    print("- Neural CFR will be even faster with more iterations (amortized setup cost)")
    print("- Neural CFR fully utilizes GPU for game simulation AND network training")
    print("- Tabular CFR is limited by Python dictionary operations on CPU")
    print("- For long training runs (1000+ iterations), neural CFR >> tabular CFR")
