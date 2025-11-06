"""Simple CFR training without expensive exploitability calculations."""

import time
from training.config import CFRConfig
from training.cfr_trainer import CFRTrainer
from training.evaluation import evaluate_vs_random

if __name__ == '__main__':
    # Simple configuration
    config = CFRConfig(
        num_iterations=1000,
        cfr_variant='cfr+',
        sampling='external',
        n_traversals_per_iter=1,
        eval_freq=200,  # Less frequent evaluation
        checkpoint_freq=500,
        checkpoint_dir='checkpoints/',
        log_dir='logs/'
    )

    print("Starting simple CFR training...")
    print(f"Config: {config.num_iterations} iterations, CFR+, external sampling")
    print("Skipping expensive exploitability calculation for speed\n")

    # Initialize trainer
    trainer = CFRTrainer(config)

    start_time = time.time()

    # Training loop
    for i in range(1, config.num_iterations + 1):
        trainer.train(1)

        # Progress indicator
        if i % 100 == 0:
            elapsed = time.time() - start_time
            num_info_sets = sum(trainer.get_num_info_sets())
            iters_per_sec = i / elapsed if elapsed > 0 else 0

            print(f"Iteration {i:4d}/{config.num_iterations} | "
                  f"{elapsed:.1f}s | {iters_per_sec:.1f} iter/s | "
                  f"{num_info_sets:,} info sets")

        # Quick evaluation (win rate vs random only - fast!)
        if i % config.eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at iteration {i}")
            print(f"{'='*60}")

            eval_start = time.time()
            win_rate = evaluate_vs_random(trainer, n_games=100)
            eval_time = time.time() - eval_start

            num_info_sets_0, num_info_sets_1 = trainer.get_num_info_sets()

            print(f"Win rate vs random:  {win_rate:.3f} ({win_rate*100:.1f}%)")
            print(f"Info sets (P0/P1):   {num_info_sets_0:,} / {num_info_sets_1:,}")
            print(f"Total info sets:     {num_info_sets_0 + num_info_sets_1:,}")
            print(f"Eval time:           {eval_time:.2f}s")
            print(f"{'='*60}\n")

    # Final stats
    total_time = time.time() - start_time
    final_win_rate = evaluate_vs_random(trainer, n_games=200)
    num_info_sets = sum(trainer.get_num_info_sets())

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Total time:           {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Training speed:       {config.num_iterations/total_time:.1f} iter/s")
    print(f"Final win rate:       {final_win_rate:.3f} ({final_win_rate*100:.1f}%)")
    print(f"Total info sets:      {num_info_sets:,}")
    print(f"{'='*60}\n")

    # Save checkpoint
    checkpoint_path = f"{config.checkpoint_dir}/final_simple.pkl"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
