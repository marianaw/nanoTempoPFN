#!/usr/bin/env python3
"""Minimal GIFT-Eval evaluation script for TimeSeriesForecaster.

Usage:
    python eval.py --checkpoint model.pkl --dataset electricity --prediction-length 96
"""

import argparse
import numpy as np
import jax.numpy as jnp
from gluonts.model.forecast import QuantileForecast
from gluonts.evaluation import make_evaluation_predictions, Evaluator

from src.config import load_config, cfg_to_training_config
from src.tsf import TimeSeriesForecaster
from src.data.time_features import compute_batch_time_features
from src.data.frequency import parse_frequency


class GluonTSPredictor:
    """Minimal wrapper for GluonTS compatibility."""

    def __init__(self, forecaster, prediction_length, context_length=512, time_dim=6):
        self.forecaster = forecaster
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.time_dim = time_dim

    def predict(self, dataset, num_samples=None):
        """Generate quantile forecasts."""
        for data in dataset:
            target = np.array(data['target'])
            start = data['start']
            freq = str(start.freq) if hasattr(start, 'freq') else 'D'

            # Use last context_length points
            history = target[-self.context_length:]
            history = jnp.array(history[None, ..., None, None])  # (1, seq, 1, 1)

            # Compute time features
            frequency = parse_frequency(freq)
            hist_len = history.shape[1]

            history_tf, future_tf = compute_batch_time_features(
                start=[np.datetime64(start)],
                history_length=hist_len,
                future_length=self.prediction_length,
                batch_size=1,
                frequency=[frequency],
                K_max=self.time_dim,
                include_extra=False,
            )

            # Predict all quantiles
            preds = self.forecaster.predict(history, history_tf, future_tf)
            preds = np.array(preds[0, :, 0, :])  # (pred_len, num_quantiles)

            yield QuantileForecast(
                forecast_arrays=preds.T,  # (num_quantiles, pred_len)
                start_date=start + len(target),
                forecast_keys=[str(q) for q in self.forecaster.quantiles],
                item_id=data.get('item_id', ''),
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate TimeSeriesForecaster on GIFT-Eval")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="conf/training.yaml", help="Config YAML path")
    parser.add_argument("--dataset", required=True, help="Dataset name or path")
    parser.add_argument("--prediction-length", type=int, default=96, help="Forecast horizon")
    parser.add_argument("--context-length", type=int, default=512, help="History length")
    args = parser.parse_args()

    # Initialize model
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    training_config = cfg_to_training_config(cfg)

    forecaster = TimeSeriesForecaster(config=training_config)
    forecaster.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Create predictor
    predictor = GluonTSPredictor(
        forecaster=forecaster,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        time_dim=training_config.time_dim,
    )

    # Load dataset (supports GIFT-Eval or GluonTS datasets)
    try:
        from gift_eval import Dataset
        dataset = Dataset(args.dataset)
        test_data = dataset.test_data
        print(f"Loaded GIFT-Eval dataset: {args.dataset}")
    except (ImportError, Exception):
        from gluonts.dataset.repository import get_dataset
        dataset = get_dataset(args.dataset)
        test_data = dataset.test
        print(f"Loaded GluonTS dataset: {args.dataset}")

    # Generate predictions
    print("Generating predictions...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,
        predictor=predictor,
        num_samples=None,
    )
    forecasts = list(forecast_it)
    targets = list(ts_it)

    # Evaluate
    print("Computing metrics...")
    evaluator = Evaluator()
    metrics, _ = evaluator(targets, forecasts)

    print("\n=== Results ===")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")


if __name__ == "__main__":
    main()
