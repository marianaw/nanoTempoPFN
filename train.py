#!/usr/bin/env python3
"""Minimal training script for TimeSeriesForecaster."""

import argparse
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
    parser.add_argument(
        "--save-path",
        type=str,
        default="model_checkpoint.pkl",
        help="Path to save trained model"
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
        batches_per_shard=data_cfg.batches_per_shard
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=training_config.batch_size,
        full_length=data_cfg.full_length,
        future_length=data_cfg.future_length,
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
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        losses = forecaster.train(loader)
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch + 1} - Average loss: {avg_loss:.6f}")

    # Save model
    print(f"\nSaving model to {args.save_path}...")
    forecaster.save(args.save_path)
    print("Training complete!")

if __name__ == "__main__":
    main()
