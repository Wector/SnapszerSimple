"""Quick CFR training test with fewer iterations."""

from training.config import CFRConfig
from training.train import train_cfr

if __name__ == '__main__':
    # Quick test configuration
    config = CFRConfig(
        num_iterations=1000,        # Reduced from 10,000 for quick test
        cfr_variant='cfr+',
        sampling='external',
        n_traversals_per_iter=1,
        eval_freq=100,              # Evaluate every 100 iterations
        checkpoint_freq=500,        # Save checkpoint every 500 iterations
        checkpoint_dir='checkpoints/',
        log_dir='logs/'
    )

    print("Starting quick CFR training test...")
    print(f"Configuration: {config.num_iterations} iterations, CFR+, external sampling")
    print("This should take 1-3 minutes...\n")

    # Train
    trainer, metrics = train_cfr(config)

    print("\n" + "="*60)
    print("Quick training test completed!")
    print("="*60)
    print(f"Final exploitability: {metrics.exploitability[-1]:.6f}")
    print(f"Final win rate vs random: {metrics.win_rate_vs_random[-1]:.3f}")
    print(f"Total info sets discovered: {metrics.num_info_sets[-1]:,}")
    print(f"\nCheckpoints saved in: {config.checkpoint_dir}")
    print(f"Plots saved in: {config.log_dir}")
