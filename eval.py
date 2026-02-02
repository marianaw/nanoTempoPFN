#!/usr/bin/env python3
"""Minimal windowed evaluation on GIFT-Eval datasets using GluonTS evaluate_model."""

import argparse
import numpy as np
import jax.numpy as jnp
from typing import Iterator
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.split import split
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import QuantileForecast
from gluonts.model.evaluation import evaluate_model
from gluonts.ev.metrics import MAE, MSE, MASE
from gluonts.time_feature import get_seasonality

from src.config import load_config, cfg_to_training_config
from src.tsf import TimeSeriesForecaster
from src.data.time_features import compute_batch_time_features
from src.data.frequency import parse_frequency


class TempoPredictorWrapper(Predictor):
    """Wrapper to make TimeSeriesForecaster compatible with GluonTS Predictor interface."""

    def __init__(self, forecaster, prediction_length, context_length, freq, time_dim):
        # super().__init__(prediction_length, freq)
        self.forecaster = forecaster
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.freq = freq
        self.time_dim = time_dim
        self.frequency = parse_frequency(freq)
        self.quantiles = forecaster.quantiles

    def predict(self, dataset, num_samples=None) -> Iterator[QuantileForecast]:
        """Generate forecasts for the test data."""
        for entry in dataset:
            # Extract target and metadata
            target = np.array(entry['target'])
            start = entry['start']
            item_id = entry.get('item_id', '')

            # Use last context_length points as context
            context = target[-self.context_length:]
            context_jnp = jnp.array(context[None, ..., None, None])

            # Compute time features - start from the beginning of context
            context_start_offset = len(target) - self.context_length
            history_tf, future_tf = compute_batch_time_features(
                # start=[np.datetime64(start) + np.timedelta64(context_start_offset, self.frequency.timedelta_unit)],
                start=[np.datetime64(start)],
                history_length=self.context_length,
                future_length=self.prediction_length,
                batch_size=1,
                frequency=[self.frequency],
                K_max=self.time_dim,
                include_extra=False,
            )

            # Predict
            preds = self.forecaster.predict(context_jnp, history_tf, future_tf)
            preds = np.array(preds[0, :, 0, :])  # (pred_len, num_quantiles)

            # Create forecast start (after full target)
            forecast_start = start + len(target)

            # Build forecast array: (num_quantiles+1, pred_len)
            mean_pred = preds.mean(axis=1, keepdims=True).T
            forecast_array = np.concatenate([preds.T, mean_pred], axis=0)

            # Create QuantileForecast
            yield QuantileForecast(
                forecast_arrays=forecast_array,
                start_date=forecast_start,
                forecast_keys=[str(q.item()) for q in self.quantiles] + ['mean'],
                item_id=item_id
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate on GIFT-Eval")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="conf/training.yaml", help="Config path")
    parser.add_argument("--dataset", default="electricity", help="Dataset name")
    parser.add_argument("--context-length", type=int, default=512, help="Context length")
    parser.add_argument("--windows", type=int, default=10, help="Number of windows per series")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset)

    # Get metadata
    freq = dataset.metadata.freq
    prediction_length = dataset.metadata.prediction_length
    seasonality = get_seasonality(freq)
    print(f"Frequency: {freq}, Prediction length: {prediction_length}, Seasonality: {seasonality}")

    # Load model
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    training_config = cfg_to_training_config(cfg)

    forecaster = TimeSeriesForecaster(config=training_config)
    forecaster.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Create test split with windowing
    # Split creates train/test, then generate_instances creates multiple windows
    _, test_template = split(dataset.test, offset=-prediction_length * args.windows)
    test_data = test_template.generate_instances(
        prediction_length=prediction_length,
        windows=args.windows,
        distance=prediction_length,
    )

    print(f"Created test split with {args.windows} windows per series")

    # Create predictor wrapper
    predictor = TempoPredictorWrapper(
        forecaster=forecaster,
        prediction_length=prediction_length,
        context_length=args.context_length,
        freq=freq,
        time_dim=training_config.time_dim
    )

    # Evaluate using GluonTS evaluate_model
    print("\n=== Evaluating ===")
    metrics_df = evaluate_model(
        model=predictor,
        test_data=test_data,
        metrics=[MAE(), MSE(), MASE()],
        seasonality=seasonality,
        axis=None
    )

    # Print results
    print("\n=== Results ===")
    print(metrics_df.to_string())


if __name__ == "__main__":
    main()
