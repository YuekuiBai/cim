"""训练模块"""

from .train import (
    TrainingConfig,
    NonlinearityAwareTrainer,
    finetune_with_nonlinearity,
    train_from_scratch,
    compare_training_strategies,
    analyze_convergence,
    plot_training_comparison
)

__all__ = [
    'TrainingConfig',
    'NonlinearityAwareTrainer',
    'finetune_with_nonlinearity',
    'train_from_scratch',
    'compare_training_strategies',
    'analyze_convergence',
    'plot_training_comparison'
]
