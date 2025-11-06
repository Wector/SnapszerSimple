"""Main CFR training script for Hungarian Snapszer."""

import os
import time
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from training.config import CFRConfig
from training.cfr_trainer import CFRTrainer
from training.evaluation import (
    compute_exploitability,
    evaluate_vs_random,
    evaluate_self_play,
    evaluate_strategy_entropy
)


class TrainingMetrics:
    """Track training metrics over iterations."""

    def __init__(self):
        self.iterations = []
        self.exploitability = []
        self.win_rate_vs_random = []
        self.self_play_mean = []
        self.self_play_std = []
        self.strategy_entropy = []
        self.num_info_sets = []
        self.wall_time = []

    def add(self, iteration: int, metrics: dict, wall_time: float):
        """Add metrics for current iteration."""
        self.iterations.append(iteration)
        self.exploitability.append(metrics.get('exploitability', 0.0))
        self.win_rate_vs_random.append(metrics.get('win_rate_vs_random', 0.0))
        self.self_play_mean.append(metrics.get('self_play_mean', 0.0))
        self.self_play_std.append(metrics.get('self_play_std', 0.0))
        self.strategy_entropy.append(metrics.get('strategy_entropy', 0.0))
        self.num_info_sets.append(metrics.get('num_info_sets', 0))
        self.wall_time.append(wall_time)

    def save(self, filepath: str):
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            'iterations': self.iterations,
            'exploitability': self.exploitability,
            'win_rate_vs_random': self.win_rate_vs_random,
            'self_play_mean': self.self_play_mean,
            'self_play_std': self.self_play_std,
            'strategy_entropy': self.strategy_entropy,
            'num_info_sets': self.num_info_sets,
            'wall_time': self.wall_time
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.iterations = data['iterations']
        self.exploitability = data['exploitability']
        self.win_rate_vs_random = data['win_rate_vs_random']
        self.self_play_mean = data['self_play_mean']
        self.self_play_std = data['self_play_std']
        self.strategy_entropy = data['strategy_entropy']
        self.num_info_sets = data['num_info_sets']
        self.wall_time = data['wall_time']


def plot_training_metrics(metrics: TrainingMetrics, save_path: str):
    """Create comprehensive training plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CFR Training Metrics', fontsize=16)

    # 1. Exploitability (most important - convergence metric)
    ax = axes[0, 0]
    ax.plot(metrics.iterations, metrics.exploitability, 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Exploitability')
    ax.set_title('Exploitability (Lower = Better Nash Approximation)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. Win rate vs random
    ax = axes[0, 1]
    ax.plot(metrics.iterations, metrics.win_rate_vs_random, 'g-', linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate vs Random Opponent')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Self-play balance
    ax = axes[0, 2]
    ax.plot(metrics.iterations, metrics.self_play_mean, 'purple', linewidth=2)
    ax.fill_between(
        metrics.iterations,
        np.array(metrics.self_play_mean) - np.array(metrics.self_play_std),
        np.array(metrics.self_play_mean) + np.array(metrics.self_play_std),
        alpha=0.3
    )
    ax.axhline(y=0, color='r', linestyle='--', label='Perfect balance')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Game Points Diff (P0 - P1)')
    ax.set_title('Self-Play Balance (Should converge to 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Strategy entropy
    ax = axes[1, 0]
    ax.plot(metrics.iterations, metrics.strategy_entropy, 'orange', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Entropy')
    ax.set_title('Strategy Entropy (Mixed vs Deterministic)')
    ax.grid(True, alpha=0.3)

    # 5. Information set growth
    ax = axes[1, 1]
    ax.plot(metrics.iterations, metrics.num_info_sets, 'brown', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Info Sets')
    ax.set_title('Information Set Discovery')
    ax.grid(True, alpha=0.3)

    # 6. Wall time
    ax = axes[1, 2]
    if len(metrics.wall_time) > 1:
        # Compute iterations per second
        iterations_per_sec = []
        for i in range(1, len(metrics.iterations)):
            iters = metrics.iterations[i] - metrics.iterations[i-1]
            time_delta = metrics.wall_time[i] - metrics.wall_time[i-1]
            if time_delta > 0:
                iterations_per_sec.append(iters / time_delta)
            else:
                iterations_per_sec.append(0)

        ax.plot(metrics.iterations[1:], iterations_per_sec, 'cyan', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Iterations/Second')
        ax.set_title('Training Throughput')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {save_path}")


def evaluate_trainer(trainer: CFRTrainer, iteration: int) -> dict:
    """Run all evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluation at iteration {iteration}")
    print(f"{'='*60}")

    metrics = {}

    # Exploitability (main convergence metric)
    print("Computing exploitability...", end=' ', flush=True)
    start = time.time()
    exploitability = compute_exploitability(trainer, n_games=50)
    metrics['exploitability'] = exploitability
    print(f"✓ ({time.time() - start:.1f}s)")

    # Win rate vs random
    print("Evaluating vs random...", end=' ', flush=True)
    start = time.time()
    win_rate = evaluate_vs_random(trainer, n_games=100)
    metrics['win_rate_vs_random'] = win_rate
    print(f"✓ ({time.time() - start:.1f}s)")

    # Self-play balance
    print("Self-play evaluation...", end=' ', flush=True)
    start = time.time()
    self_play_mean, self_play_std = evaluate_self_play(trainer, n_games=100)
    metrics['self_play_mean'] = self_play_mean
    metrics['self_play_std'] = self_play_std
    print(f"✓ ({time.time() - start:.1f}s)")

    # Strategy entropy
    print("Computing strategy entropy...", end=' ', flush=True)
    start = time.time()
    entropy = evaluate_strategy_entropy(trainer)
    metrics['strategy_entropy'] = entropy
    print(f"✓ ({time.time() - start:.1f}s)")

    # Info set counts
    num_info_sets_0, num_info_sets_1 = trainer.get_num_info_sets()
    metrics['num_info_sets'] = num_info_sets_0 + num_info_sets_1

    # Print summary
    print(f"\nResults:")
    print(f"  Exploitability:      {exploitability:.6f}")
    print(f"  Win rate vs random:  {win_rate:.3f} ({win_rate*100:.1f}%)")
    print(f"  Self-play balance:   {self_play_mean:.3f} ± {self_play_std:.3f}")
    print(f"  Strategy entropy:    {entropy:.3f}")
    print(f"  Info sets (P0/P1):   {num_info_sets_0:,} / {num_info_sets_1:,}")
    print(f"  Total info sets:     {metrics['num_info_sets']:,}")
    print(f"{'='*60}\n")

    return metrics


def train_cfr(config: CFRConfig, resume_from: str = None):
    """
    Main CFR training loop.

    Args:
        config: Training configuration
        resume_from: Optional checkpoint path to resume from
    """
    # Create output directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Initialize trainer
    trainer = CFRTrainer(config)
    metrics = TrainingMetrics()

    start_iteration = 0

    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)
        start_iteration = trainer.iteration

        # Try to load metrics
        metrics_path = resume_from.replace('.pkl', '_metrics.json')
        if os.path.exists(metrics_path):
            metrics.load(metrics_path)
            print(f"Loaded metrics from {metrics_path}")

    print("\n" + "="*60)
    print("CFR Training Configuration")
    print("="*60)
    print(f"Algorithm:           {config.cfr_variant.upper()}")
    print(f"Sampling:            {config.sampling}")
    print(f"Total iterations:    {config.num_iterations:,}")
    print(f"Eval frequency:      {config.eval_freq}")
    print(f"Checkpoint freq:     {config.checkpoint_freq}")
    print(f"Starting iteration:  {start_iteration}")
    print("="*60 + "\n")

    # Training loop
    training_start_time = time.time()
    last_eval_time = training_start_time

    for iteration in range(start_iteration + 1, config.num_iterations + 1):
        # Run one iteration of CFR
        trainer.train(1)

        # Periodic evaluation
        if iteration % config.eval_freq == 0 or iteration == 1:
            current_time = time.time()

            # Evaluate
            eval_metrics = evaluate_trainer(trainer, iteration)
            metrics.add(iteration, eval_metrics, current_time - training_start_time)

            # Compute time estimates
            elapsed = current_time - training_start_time
            iterations_done = iteration - start_iteration
            if iterations_done > 0:
                time_per_iter = elapsed / iterations_done
                remaining_iters = config.num_iterations - iteration
                eta_seconds = time_per_iter * remaining_iters
                eta_minutes = eta_seconds / 60

                print(f"Timing: {elapsed:.1f}s elapsed, "
                      f"{time_per_iter*1000:.2f}ms/iter, "
                      f"ETA: {eta_minutes:.1f}min\n")

        # Progress indicator for non-eval iterations
        elif iteration % 100 == 0:
            num_info_sets = sum(trainer.get_num_info_sets())
            print(f"Iteration {iteration:,}/{config.num_iterations:,} "
                  f"({100*iteration/config.num_iterations:.1f}%) - "
                  f"{num_info_sets:,} info sets discovered")

        # Save checkpoint
        if iteration % config.checkpoint_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"cfr_iter_{iteration}_{timestamp}.pkl"
            )
            trainer.save_checkpoint(checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")

            # Save metrics alongside checkpoint
            metrics_path = checkpoint_path.replace('.pkl', '_metrics.json')
            metrics.save(metrics_path)
            print(f"Metrics saved: {metrics_path}\n")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60 + "\n")

    final_metrics = evaluate_trainer(trainer, config.num_iterations)
    current_time = time.time()
    metrics.add(config.num_iterations, final_metrics, current_time - training_start_time)

    # Save final checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_checkpoint = os.path.join(
        config.checkpoint_dir,
        f"cfr_final_{timestamp}.pkl"
    )
    trainer.save_checkpoint(final_checkpoint)
    print(f"\nFinal checkpoint saved: {final_checkpoint}")

    # Save final metrics
    final_metrics_path = os.path.join(
        config.log_dir,
        f"training_metrics_{timestamp}.json"
    )
    metrics.save(final_metrics_path)
    print(f"Final metrics saved: {final_metrics_path}")

    # Generate plots
    plot_path = os.path.join(
        config.log_dir,
        f"training_plots_{timestamp}.png"
    )
    plot_training_metrics(metrics, plot_path)

    # Training summary
    total_time = time.time() - training_start_time
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total time:          {total_time/60:.1f} minutes")
    print(f"Iterations:          {config.num_iterations:,}")
    print(f"Final exploitability: {final_metrics['exploitability']:.6f}")
    print(f"Final win rate:      {final_metrics['win_rate_vs_random']:.3f}")
    print(f"Total info sets:     {final_metrics['num_info_sets']:,}")
    print(f"{'='*60}\n")

    return trainer, metrics


def main():
    """Main entry point."""
    # Default configuration
    config = CFRConfig(
        num_iterations=10_000,
        cfr_variant='cfr+',
        sampling='external',
        n_traversals_per_iter=1,
        eval_freq=100,
        checkpoint_freq=1000,
        checkpoint_dir='checkpoints/',
        log_dir='logs/'
    )

    # Train
    trainer, metrics = train_cfr(config)

    print("Training finished successfully!")
    print(f"Checkpoints saved in: {config.checkpoint_dir}")
    print(f"Logs and plots in: {config.log_dir}")


if __name__ == '__main__':
    main()
