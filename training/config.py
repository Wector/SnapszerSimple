"""Configuration for CFR training."""

from dataclasses import dataclass


@dataclass
class CFRConfig:
    """Configuration for tabular CFR training."""

    # Training parameters
    num_iterations: int = 10_000
    cfr_variant: str = 'cfr+'  # 'vanilla' or 'cfr+'
    sampling: str = 'outcome'  # 'vanilla', 'external', or 'outcome'

    # Sampling parameters
    n_traversals_per_iter: int = 1  # Traversals per player per iteration

    # Evaluation
    eval_freq: int = 100  # Evaluate every N iterations
    checkpoint_freq: int = 1000  # Save checkpoint every N iterations

    # Paths
    checkpoint_dir: str = 'checkpoints/'
    log_dir: str = 'logs/'

    def __post_init__(self):
        """Validate configuration."""
        assert self.cfr_variant in ['vanilla', 'cfr+'], \
            f"cfr_variant must be 'vanilla' or 'cfr+', got {self.cfr_variant}"
        assert self.sampling in ['vanilla', 'external', 'outcome'], \
            f"sampling must be 'vanilla', 'external', or 'outcome', got {self.sampling}"
        assert self.num_iterations > 0, "num_iterations must be positive"
        assert self.eval_freq > 0, "eval_freq must be positive"
        assert self.checkpoint_freq > 0, "checkpoint_freq must be positive"
