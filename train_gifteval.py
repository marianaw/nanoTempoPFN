#!/usr/bin/env python3
"""Minimal training script for GIFT-Eval datasets."""

import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from gluonts.dataset.repository import get_dataset

from src.config import load_config, cfg_to_training_config
from src.tsf import TimeSeriesForecaster
from src.data.containers import NpBatchTSContainer
from src.data.time_features import compute_batch_time_features
from src.data.frequency import parse_frequency

from tqdm import tqdm


class GiftEvalDataLoader:
    """DataLoader for GIFT-Eval that yields NpBatchTSContainer."""

    def __init__(self, data, freq, prediction_length, batch_size=32,
                 context_lengths=(128, 512, 1024), max_windows_per_series=100,
                 time_dim=6, seed=42):
        self.data = list(data)
        self.freq = freq
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.context_lengths = context_lengths
        self.max_windows_per_series = max_windows_per_series
        self.time_dim = time_dim

        # Parse frequency once
        self.frequency = parse_frequency(freq)

        # Set seed
        random.seed(seed)
        np.random.seed(seed)

        # Create windows
        self.windows = self._create_windows()
        random.shuffle(self.windows)

    def _create_windows(self):
        """Generate windows with random context lengths."""
        windows = []

        for ts in self.data:
            target = np.array(ts['target'])

            for _ in range(self.max_windows_per_series):
                # Randomly sample context length
                context_length = random.choice(self.context_lengths)
                window_size = context_length + self.prediction_length

                if len(target) < window_size:
                    continue

                # Random start position
                max_start = len(target) - window_size
                start_idx = random.randint(0, max_start)
                end_idx = start_idx + window_size

                context = target[start_idx:start_idx + context_length]
                pred_target = target[start_idx + context_length:end_idx]

                windows.append({
                    'context': context,
                    'target': pred_target,
                    'context_length': context_length,
                    'start': ts['start'],
                })

        return windows

    def __len__(self):
        return len(self.windows) // self.batch_size

    def __iter__(self):
        """Yield batches as NpBatchTSContainer."""
        random.shuffle(self.windows)

        num_batches = len(self.windows) // self.batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = batch_start + self.batch_size
            batch_windows = self.windows[batch_start:batch_end]

            # Prepare batch components
            contexts = []
            targets = []
            starts = []
            frequencies = []
            history_tfs = []
            future_tfs = []

            # Find max context length in batch
            max_context_len = max(w['context_length'] for w in batch_windows)

            for window in batch_windows:
                context_length = window['context_length']
                context = window['context']

                # Pad context to max length if needed
                if len(context) < max_context_len:
                    pad_len = max_context_len - len(context)
                    context = np.pad(context, (pad_len, 0), mode='edge')

                contexts.append(context)
                targets.append(window['target'])
                starts.append(np.datetime64(window['start']))
                frequencies.append(self.frequency)

                # Compute time features
                history_tf, future_tf = compute_batch_time_features(
                    start=[np.datetime64(window['start'])],
                    history_length=max_context_len,
                    future_length=self.prediction_length,
                    batch_size=1,
                    frequency=[self.frequency],
                    K_max=self.time_dim,
                    include_extra=False,
                )
                history_tfs.append(history_tf[0])
                future_tfs.append(future_tf[0])

            # Stack into batch arrays
            history = np.stack(contexts)[:, :, None, None]  # (batch, seq, 1, 1)
            future = np.stack(targets)[:, :, None, None]    # (batch, pred, 1, 1)
            history_tf = np.stack(history_tfs)
            future_tf = np.stack(future_tfs)

            # Create NpBatchTSContainer
            batch = NpBatchTSContainer(
                history=history,
                future=future,
                start=starts,
                frequency=frequencies,
                history_time_features=history_tf,
                future_time_features=future_tf,
            )

            yield batch


def main():
    parser = argparse.ArgumentParser(description="Train on GIFT-Eval")
    parser.add_argument("--config", default="conf/training.yaml", help="Config path")
    parser.add_argument("--dataset", default="electricity", help="Dataset name")
    parser.add_argument("--output-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max-windows", type=int, default=100, help="Max windows per series")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset)
    train_data = dataset.train

    # Get metadata
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    print(f"Frequency: {freq}, Prediction length: {prediction_length}")

    # Load model
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    training_config = cfg_to_training_config(cfg)
    forecaster = TimeSeriesForecaster(config=training_config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n=== Training for {args.epochs} epochs ===")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Create data loader (fresh each epoch for shuffling)
        loader = GiftEvalDataLoader(
            data=train_data,
            freq=freq,
            prediction_length=prediction_length,
            batch_size=args.batch_size,
            context_lengths=(128, 512, 1024),
            max_windows_per_series=args.max_windows,
            time_dim=training_config.time_dim,
            seed=args.seed + epoch,  # Different seed per epoch
        )
        print(f"  Created loader with {len(loader)} batches")

        # Custom training loop with progress bar
        epoch_losses = []
        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            # Training step
            loss = forecaster.train_step(batch)
            epoch_losses.append(loss)

            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss:.4f}'})

        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = output_dir / f"model_epoch_{epoch + 1}.pkl"
        forecaster.save(str(checkpoint_path))
        print(f"  Saved checkpoint: {checkpoint_path}")

    print("\n=== Training Complete ===")
    final_path = output_dir / "model_final.pkl"
    forecaster.save(str(final_path))
    print(f"Final model saved: {final_path}")


if __name__ == "__main__":
    main()
