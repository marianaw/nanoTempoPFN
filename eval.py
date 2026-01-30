#!/usr/bin/env python3
"""Minimal windowed evaluation on GIFT-Eval datasets."""

import argparse
import numpy as np
import jax.numpy as jnp
from gluonts.dataset.repository import get_dataset

from src.config import load_config, cfg_to_training_config
from src.tsf import TimeSeriesForecaster
from src.data.time_features import compute_batch_time_features
from src.data.frequency import parse_frequency


class TestDataset:
    """Split time series into non-overlapping windows."""

    def __init__(self, data, context_length, prediction_length, max_windows=20):
        self.data = list(data)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.max_windows = max_windows
        self.window_size = context_length + prediction_length

    def create_windows(self):
        """Generate (context, target, metadata) tuples."""
        windows = []

        for ts in self.data:
            target = np.array(ts['target'])

            # Calculate how many windows we can fit
            max_possible = (len(target) - self.window_size) // self.prediction_length + 1
            num_windows = min(max_possible, self.max_windows)

            # Create windows working backwards from end
            for i in range(num_windows):
                end_idx = len(target) - i * self.prediction_length
                start_idx = end_idx - self.window_size

                if start_idx < 0:
                    break

                context = target[start_idx:start_idx + self.context_length]
                pred_target = target[start_idx + self.context_length:end_idx]

                windows.append({
                    'context': context,
                    'target': pred_target,
                    'start': ts['start'],
                    'item_id': ts.get('item_id', '')
                })

        return windows


def quantile_loss(y_true, y_pred, quantile):
    """Compute quantile loss."""
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))


def main():
    parser = argparse.ArgumentParser(description="Evaluate on GIFT-Eval")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="conf/training.yaml", help="Config path")
    parser.add_argument("--dataset", default="electricity", help="Dataset name")
    parser.add_argument("--context-length", type=int, default=512, help="Context length")
    parser.add_argument("--max-windows", type=int, default=20, help="Max windows per series")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset)
    test_data = dataset.test

    # Get metadata
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    print(f"Frequency: {freq}, Prediction length: {prediction_length}")

    # Load model
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    training_config = cfg_to_training_config(cfg)

    forecaster = TimeSeriesForecaster(config=training_config)
    forecaster.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Create test windows
    test_dataset = TestDataset(
        data=test_data,
        context_length=args.context_length,
        prediction_length=prediction_length,
        max_windows=args.max_windows
    )
    windows = test_dataset.create_windows()
    print(f"Created {len(windows)} windows")

    # Parse frequency once
    frequency = parse_frequency(freq)
    quantiles = forecaster.quantiles

    # Run predictions
    print("Running predictions...")
    predictions = []
    targets = []

    for i, window in enumerate(windows):
        if i % 100 == 0:
            print(f"  {i}/{len(windows)}")

        # Prepare context
        context = jnp.array(window['context'][None, ..., None, None])

        # Compute time features
        history_tf, future_tf = compute_batch_time_features(
            start=[np.datetime64(window['start'])],
            history_length=args.context_length,
            future_length=prediction_length,
            batch_size=1,
            frequency=[frequency],
            K_max=training_config.time_dim,
            include_extra=False,
        )

        # Predict
        preds = forecaster.predict(context, history_tf, future_tf)
        preds = np.array(preds[0, :, 0, :])  # (pred_len, num_quantiles)

        predictions.append(preds)
        targets.append(window['target'])

    # Stack results
    all_preds = np.stack(predictions)  # (num_windows, pred_len, num_quantiles)
    all_targets = np.stack(targets)    # (num_windows, pred_len)

    print(f"\nPredictions shape: {all_preds.shape}")
    print(f"Targets shape: {all_targets.shape}")

    # Compute metrics
    print("\n=== Computing Metrics ===")

    # MSE (median)
    median_idx = list(quantiles).index(0.5)
    median_preds = all_preds[:, :, median_idx]
    mse_median = np.mean((median_preds - all_targets) ** 2)

    # MSE (mean)
    mean_preds = np.mean(all_preds, axis=2)
    mse_mean = np.mean((mean_preds - all_targets) ** 2)

    # Quantile losses
    quantile_losses = {}
    for i, q in enumerate(quantiles):
        q_preds = all_preds[:, :, i]
        loss = quantile_loss(all_targets, q_preds, q)
        quantile_losses[q] = loss

    avg_ql = np.mean(list(quantile_losses.values()))

    # Print results
    print("\n=== Results ===")
    print(f"MSE (median): {mse_median:.4f}")
    print(f"MSE (mean): {mse_mean:.4f}")
    print(f"Avg Quantile Loss: {avg_ql:.4f}")
    print("\nPer-Quantile Losses:")
    for q, loss in quantile_losses.items():
        print(f"  Q{q}: {loss:.4f}")


if __name__ == "__main__":
    main()
