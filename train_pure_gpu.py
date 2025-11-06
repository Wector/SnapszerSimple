"""Pure GPU CFR training - Minimal CPU usage!"""

import time
import jax

from training.neural_cfr_pure_gpu import PureGPUCFRTrainer, PureGPUCFRConfig
from training.evaluation import evaluate_vs_random


if __name__ == '__main__':
    # Configuration
    config = PureGPUCFRConfig(
        num_iterations=100,
        games_per_batch=512,          # 512 games in parallel
        train_batches_per_iter=20,    # 20 training steps per iteration

        hidden_sizes=(256, 256, 128),
        learning_rate=1e-3,

        eval_freq=10,
        checkpoint_freq=50,

        checkpoint_dir='checkpoints/pure_gpu/',
        log_dir='logs/pure_gpu/'
    )

    print("="*70)
    print("PURE GPU CFR TRAINING - ZERO CPU BOTTLENECK")
    print("="*70)
    print(f"Iterations:           {config.num_iterations}")
    print(f"Games per batch:      {config.games_per_batch}")
    print(f"Train batches/iter:   {config.train_batches_per_iter}")
    print(f"Network architecture: {config.hidden_sizes}")
    print(f"Learning rate:        {config.learning_rate}")
    print("="*70)
    print()
    print("KEY DIFFERENCE:")
    print("- NO experience buffer (no CPU memory)")
    print("- NO JAX→NumPy conversions (no CPU transfers)")
    print("- Everything stays on GPU!")
    print("="*70)
    print()

    # Initialize trainer
    rng_key = jax.random.PRNGKey(42)
    trainer = PureGPUCFRTrainer(config, rng_key)

    print("Initializing neural network...")
    print(f"GPU: {jax.devices()[0].platform == 'gpu'}")
    print(f"Device: {jax.devices()[0]}")
    print()
    print("MONITOR GPU/CPU USAGE:")
    print("  nvidia-smi  # GPU should be 90-100%")
    print("  htop        # CPU should be minimal")
    print()

    start_time = time.time()

    # Training loop
    print("Training...")
    for i in range(1, config.num_iterations + 1):
        iter_start = time.time()

        # Train one iteration (all on GPU!)
        trainer.train(1)

        iter_time = time.time() - iter_start

        # Progress
        if i % 10 == 0:
            elapsed = time.time() - start_time
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

    print("\nPERFORMANCE:")
    print("If you monitored GPU/CPU during training:")
    print("  ✓ GPU: 90-100% (all compute happening here)")
    print("  ✓ CPU: <10% (only print statements)")
    print("\nThis is MAXIMUM efficiency! 🚀")
