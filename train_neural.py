"""Neural CFR training script using Single Deep CFR."""

import time
import jax

from training.neural_cfr_trainer import NeuralCFRTrainer, NeuralCFRConfig
from training.evaluation import evaluate_vs_random


if __name__ == '__main__':
    # Configuration
    config = NeuralCFRConfig(
        num_iterations=100,             # Start with 100 iterations
        trajectories_per_iter=100,      # 100 games per iteration
        train_steps_per_iter=50,        # 50 network updates per iteration

        hidden_sizes=(256, 256, 128),   # Network architecture
        learning_rate=1e-3,

        buffer_capacity=50_000,         # 50k training samples

        cfr_variant='cfr+',

        eval_freq=10,
        checkpoint_freq=50,

        checkpoint_dir='checkpoints/neural/',
        log_dir='logs/neural/'
    )

    print("="*70)
    print("NEURAL CFR TRAINING (Single Deep CFR)")
    print("="*70)
    print(f"Iterations:           {config.num_iterations}")
    print(f"Trajectories/iter:    {config.trajectories_per_iter}")
    print(f"Train steps/iter:     {config.train_steps_per_iter}")
    print(f"Network architecture: {config.hidden_sizes}")
    print(f"Learning rate:        {config.learning_rate}")
    print(f"Buffer capacity:      {config.buffer_capacity:,}")
    print(f"CFR variant:          {config.cfr_variant}")
    print("="*70)
    print()

    # Initialize trainer
    rng_key = jax.random.PRNGKey(42)
    trainer = NeuralCFRTrainer(config, rng_key)

    print("Initializing neural network...")
    print(f"GPU available: {jax.devices()[0].platform == 'gpu'}")
    print(f"Device: {jax.devices()[0]}")
    print()

    start_time = time.time()

    # Training loop with evaluation
    for i in range(1, config.num_iterations + 1):
        iter_start = time.time()

        # Train one iteration
        trainer.train(1)

        iter_time = time.time() - iter_start

        # Progress
        if i % 10 == 0:
            elapsed = time.time() - start_time
            iters_per_sec = i / elapsed if elapsed > 0 else 0

            print(f"Iteration {i:3d}/{config.num_iterations} | "
                  f"{elapsed:.1f}s | {iters_per_sec:.2f} iter/s | "
                  f"{iter_time:.2f}s/iter | "
                  f"Buffer: {len(trainer.buffer):,}")

        # Evaluation
        if i % config.eval_freq == 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at iteration {i}")
            print(f"{'='*70}")

            eval_start = time.time()
            win_rate = evaluate_vs_random(trainer, n_games=100)
            eval_time = time.time() - eval_start

            print(f"Win rate vs random:  {win_rate:.3f} ({win_rate*100:.1f}%)")
            print(f"Buffer size:         {len(trainer.buffer):,} samples")
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
    print(f"Total samples:       {len(trainer.buffer):,}")
    print(f"{'='*70}\n")

    # Save final checkpoint
    final_path = f"{config.checkpoint_dir}/final.pkl"
    trainer.save_checkpoint(final_path)
    print(f"Final checkpoint saved: {final_path}")

    print("\nNEXT STEPS:")
    print("- Increase num_iterations to 1000+ for better convergence")
    print("- Adjust trajectories_per_iter and train_steps_per_iter")
    print("- Monitor GPU utilization with: nvidia-smi")
    print("- Compare speed to tabular CFR (should be 10-100x faster!)")
