#!/usr/bin/env python3
"""Minimal training script for TimeSeriesForecaster."""

import argparse
import jax
from pathlib import Path
from src.config import load_config, cfg_to_training_config
from src.data.containers import ShardedDataset, DataLoader
from src.tsf import TimeSeriesForecaster

def main():
    parser = argparse.ArgumentParser(description="Train TimeSeriesForecaster")
    parser.add_argument(
        "--config",
        type=str,
        default="conf/training.yaml",
        help="Path to training config YAML file"
    )
    args = parser.parse_args()

    # Load config from YAML
    print(f"Loading config from {args.config}...")
    cfg = load_config(args.config)

    # Convert to TrainingConfig dataclass
    training_config = cfg_to_training_config(cfg)

    # Setup data
    print("Setting up data loader...")
    data_cfg = cfg.data
    dataset = ShardedDataset(
        root_path=Path(data_cfg.root_path),
        time_features_path=Path(data_cfg.timefs_rootpth) if data_cfg.timefs_rootpth is not None else None
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=training_config.batch_size,
        full_length=data_cfg.full_length,
        seed=data_cfg.seed
    )

    # Initialize forecaster
    print("Initializing model...")
    print(f"  num_heads={cfg.num_heads}, n_layers={cfg.n_layers}")
    print(f"  embedding_dim={cfg.embedding_dim}, head_embedding_dim={cfg.head_embedding_dim}")
    forecaster = TimeSeriesForecaster(config=training_config, seed=data_cfg.seed)

    # Training loop
    num_epochs = training_config.num_epochs
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Checkpoints will be saved to {training_config.checkpoint_path} after each epoch")
    forecaster.train(loader, num_epochs=num_epochs)
    print("\nTraining complete!")

if __name__ == "__main__":
    main()
