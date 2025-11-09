"""GPU-maxed CFR training - MAXIMUM utilization!"""

import time
import jax

from training.neural_cfr_pure_gpu import PureGPUCFRTrainer, PureGPUCFRConfig
from training.evaluation import evaluate_vs_random


if __name__ == '__main__':
    # Configuration - MAXIMUM GPU UTILIZATION
    config = PureGPUCFRConfig(
        num_iterations=100,

        # MASSIVE BATCHES - saturate the GPU!
        games_per_batch=8192,         # 8192 games at once on GPU!
        train_batches_per_iter=200,   # 200 batches per iteration (90-100% GPU!)

        # YOUR NETWORK SIZE
        hidden_sizes=(1024, 512, 256),  # Your requested architecture
        learning_rate=1e-3,

        eval_freq=5,  # Evaluate every 5 iterations
        checkpoint_freq=25,  # Save every 25 iterations

        checkpoint_dir='checkpoints/gpu_maxed/',
        log_dir='logs/gpu_maxed/'
    )

    print("="*70)
    print("GPU-MAXED CFR TRAINING - MAXIMUM UTILIZATION")
    print("="*70)
    print(f"Iterations:           {config.num_iterations}")
    print(f"Games per batch:      {config.games_per_batch:,}")
    print(f"Train batches/iter:   {config.train_batches_per_iter}")
    print(f"Network architecture: {config.hidden_sizes}")
    print(f"Learning rate:        {config.learning_rate}")
    print("="*70)
    print()
    print("OPTIMIZATIONS:")
    print(f"  - {config.games_per_batch:,} games/batch (vs 256)")
    print(f"  - {sum(config.hidden_sizes):,} total neurons (vs 640)")
    print(f"  - {config.train_batches_per_iter} batches/iter (more GPU work)")
    print("="*70)
    print()
    print("EXPECTED GPU USAGE: 80-100%")
    print("If GPU is still low, increase games_per_batch to 8192 or 16384")
    print()

    # Initialize trainer
    rng_key = jax.random.PRNGKey(42)

    print("Initializing (this takes time due to JIT compilation)...")
    trainer = PureGPUCFRTrainer(config, rng_key)

    print(f"Device: {jax.devices()[0]}")
    print()
    print("NOW MONITOR GPU:")
    print("  nvidia-smi")
    print()

    start_time = time.time()
    compilation_done = False

    # Training loop
    print("Training...")
    for i in range(1, config.num_iterations + 1):
        iter_start = time.time()

        # Train one iteration
        trainer.train(1)

        iter_time = time.time() - iter_start

        # First iteration is slow (JIT compilation)
        if i == 1:
            print(f"First iteration: {iter_time:.1f}s (JIT compilation overhead)")
            print(f"Subsequent iterations should be much faster...")
            print()
            compilation_done = True

        # Progress
        if i % 5 == 0:
            elapsed = time.time() - start_time
            # Exclude first iteration from speed calculation
            if compilation_done and i > 1:
                effective_elapsed = elapsed - iter_time  # Subtract compilation time
                effective_iters = i - 1
                iters_per_sec = effective_iters / effective_elapsed if effective_elapsed > 0 else 0
            else:
                iters_per_sec = i / elapsed if elapsed > 0 else 0

            print(f"Iteration {i:3d}/{config.num_iterations} | "
                  f"{elapsed:.1f}s | {iters_per_sec:.2f} iter/s | "
                  f"{iter_time:.2f}s/iter")

        # Evaluation
        if i % config.eval_freq == 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at iteration {i}")
            print(f"{'='*70}")

            eval_start = time.time()
            win_rate = evaluate_vs_random(trainer, n_games=100)
            eval_time = time.time() - eval_start

            print(f"Win rate vs random:  {win_rate:.3f} ({win_rate*100:.1f}%)")
            print(f"Eval time:           {eval_time:.2f}s")
            print(f"{'='*70}\n")

        # Checkpoint
        if i % config.checkpoint_freq == 0:
            checkpoint_path = f"{config.checkpoint_dir}/checkpoint_{i}.pkl"
            trainer.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Final evaluation
    total_time = time.time() - start_time
    final_win_rate = evaluate_vs_random(trainer, n_games=200)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time:          {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Training speed:      {config.num_iterations/total_time:.2f} iter/s")
    print(f"Time per iteration:  {total_time/config.num_iterations:.2f}s")
    print(f"Final win rate:      {final_win_rate:.3f} ({final_win_rate*100:.1f}%)")
    print(f"{'='*70}\n")

    # Save final checkpoint
    final_path = f"{config.checkpoint_dir}/final.pkl"
    trainer.save_checkpoint(final_path)
    print(f"Final checkpoint saved: {final_path}")

    print("\nGPU TUNING:")
    print("If GPU was still underutilized (<80%), try:")
    print("  - Increase games_per_batch to 8192 or 16384")
    print("  - Increase network size to (1024, 1024, 512)")
    print("  - Increase train_batches_per_iter to 100")
