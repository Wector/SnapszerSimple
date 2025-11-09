"""Train GPU-Accelerated Neural CFR with visualization."""

import time
import jax
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from training.neural_cfr_pure_gpu import PureGPUCFRTrainer, PureGPUCFRConfig
from training.evaluation import evaluate_vs_random


def plot_progress(metrics, save_path='logs/gpu_cfr/progress.png'):
    """Plot training progress."""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss over time
    if len(metrics['avg_loss']) > 0:
        ax1.plot(metrics['iteration'], metrics['avg_loss'], 'b-', linewidth=2)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Average Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)

    # Win rate vs random
    if len(metrics['win_rate_vs_random']) > 0:
        ax2.plot(metrics['eval_iterations'], metrics['win_rate_vs_random'], 'g-', linewidth=2, marker='o')
        ax2.axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Win Rate vs Random Opponent')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(True)

    # Buffer size
    if len(metrics['buffer_size']) > 0:
        ax3.plot(metrics['iteration'], metrics['buffer_size'], 'm-', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Buffer Size')
        ax3.set_title('Reservoir Buffer Size')
        ax3.grid(True)

    # Info
    ax4.text(0.5, 0.5, f'Current Iteration: {metrics["iteration"][-1] if metrics["iteration"] else 0}',
             ha='center', va='center', fontsize=20)
    ax4.set_title('Info')
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"  Progress plot saved: {save_path}")


if __name__ == '__main__':
    # Configuration - FULL GPU UTILIZATION with your settings
    config = PureGPUCFRConfig(
        num_iterations=100,

        # MASSIVE GPU batches - everything on GPU!
        games_per_batch=8192,  # 8192 games simultaneously on GPU
        train_batches_per_iter=200,  # 200 batches keeps GPU at 100%

        # Your network size
        hidden_sizes=(1024, 512, 256),
        learning_rate=1e-3,

        # Evaluation
        eval_freq=5,
        checkpoint_freq=25,

        checkpoint_dir='checkpoints/gpu_cfr/',
        log_dir='logs/gpu_cfr/'
    )

    print("="*70)
    print("GPU-ACCELERATED NEURAL CFR")
    print("="*70)
    print(f"Iterations:           {config.num_iterations}")
    print(f"CFR games/iter:       {config.cfr_games_per_iter} (GPU-accelerated!)")
    print(f"Train steps/iter:     {config.train_steps_per_iter}")
    print(f"Batch size:           {config.batch_size}")
    print(f"Network:              {config.hidden_sizes}")
    print(f"Learning rate:        {config.learning_rate}")
    print(f"Reservoir size:       {config.reservoir_size:,}")
    print("="*70)
    print()
    print("HOW IT WORKS:")
    print("1. Generate games on GPU (batched, JIT-compiled)")
    print("2. Compute outcome-based advantages (GPU)")
    print("3. Train network to predict strategies (GPU)")
    print("4. Repeat → fast convergence!")
    print()
    print("KEY ADVANTAGE: CFR generation now runs on GPU!")
    print("Expected speedup: 10-30x faster than CPU CFR")
    print("="*70)
    print()

    # Initialize learner
    rng_key = jax.random.PRNGKey(42)

    print("Initializing (JIT compilation will happen on first iteration)...")
    learner = GPUCFRLearner(config, rng_key)

    print(f"Device: {jax.devices()[0]}")
    print()

    start_time = time.time()
    first_iter_time = None

    # Training loop
    for i in range(1, config.num_iterations + 1):
        iter_start = time.time()

        # Train one iteration
        learner.train(1)

        iter_time = time.time() - iter_start

        # Track first iteration time (includes compilation)
        if i == 1:
            first_iter_time = iter_time
            print(f"\nFirst iteration: {iter_time:.1f}s (includes JIT compilation)")
            print("Subsequent iterations should be much faster...\n")

        # Evaluation
        if i % config.eval_freq == 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at iteration {i}")
            print(f"{'='*70}")

            eval_start = time.time()
            win_rate = evaluate_vs_random(learner, n_games=100)
            eval_time = time.time() - eval_start

            # Store metrics
            learner.metrics['eval_iterations'].append(i)
            learner.metrics['win_rate_vs_random'].append(win_rate)

            print(f"Win rate vs random:  {win_rate:.3f} ({win_rate*100:.1f}%)")
            print(f"Buffer size:         {learner.buffer.size:,} states")
            print(f"Avg loss:            {learner.metrics['avg_loss'][-1]:.4f}")
            print(f"Eval time:           {eval_time:.2f}s")
            print(f"{'='*70}\n")

            # Plot progress
            plot_progress(learner.metrics)

        # Checkpoint
        if i % config.checkpoint_freq == 0:
            checkpoint_path = f"{config.checkpoint_dir}/checkpoint_{i}.pkl"
            learner.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Progress summary
        if i % 10 == 0:
            elapsed = time.time() - start_time
            # Exclude compilation time from speed calculation
            if first_iter_time and i > 1:
                effective_time = elapsed - first_iter_time
                effective_iters = i - 1
                iters_per_sec = effective_iters / effective_time if effective_time > 0 else 0
            else:
                iters_per_sec = i / elapsed if elapsed > 0 else 0

            print(f"\n--- Iteration {i}/{config.num_iterations} ---")
            print(f"Time: {elapsed:.1f}s | Speed: {iters_per_sec:.2f} iter/s")

    # Final evaluation and plot
    total_time = time.time() - start_time
    final_win_rate = evaluate_vs_random(learner, n_games=200)
    learner.metrics['eval_iterations'].append(config.num_iterations)
    learner.metrics['win_rate_vs_random'].append(final_win_rate)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time:          {total_time:.1f}s ({total_time/60:.1f} min)")

    # Calculate speed excluding compilation
    if first_iter_time:
        effective_time = total_time - first_iter_time
        effective_iters = config.num_iterations - 1
        actual_speed = effective_iters / effective_time if effective_time > 0 else 0
        print(f"Training speed:      {actual_speed:.2f} iter/s (excluding compilation)")
    else:
        print(f"Training speed:      {config.num_iterations/total_time:.2f} iter/s")

    print(f"Final win rate:      {final_win_rate:.3f} ({final_win_rate*100:.1f}%)")
    print(f"Total states learned: {learner.buffer.size:,}")
    print(f"{'='*70}\n")

    # Final plot
    plot_progress(learner.metrics)

    # Save final checkpoint
    final_path = f"{config.checkpoint_dir}/final.pkl"
    learner.save_checkpoint(final_path)
    print(f"Final checkpoint saved: {final_path}")

    print(f"\nProgress plots: logs/gpu_cfr/progress.png")
    print(f"Checkpoints: {config.checkpoint_dir}")

    print("\nGPU CFR is now running! Monitor GPU usage:")
    print("  watch -n 1 nvidia-smi")
