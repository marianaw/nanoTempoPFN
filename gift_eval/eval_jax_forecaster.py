""" This is a frankensteinish script to run evaluations on gift-eval consisting of copy-paste code from
other repos, copy-pastes from my own code, and a bit of AI-generated stuff."""

import sys

sys.path.append('/Users/mariana/Documents/projects/ts/gift-eval/src')
sys.path.append('/Users/mariana/Documents/projects/ts/nanoTempoPFN')

import os
os.environ['GIFT_EVAL'] = '/Users/mariana/Documents/projects/ts/gift-eval-data'

# from gift_eval.data import Dataset
import numpy as np
import jax.numpy as jnp
import pyarrow.compute as pc
import datasets
import math 
import json
from argparse import ArgumentParser

from src.tsf import TimeSeriesForecaster
from src.config import load_config, cfg_to_training_config
from src.data.time_features import compute_batch_time_features
from src.data.frequency import parse_frequency
from src.data.containers import NpBatchTSContainer

from gluonts.itertools import Map
from gluonts.time_feature import get_seasonality, norm_freq_str
from gluonts.dataset import DataEntry
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import Predictor
from gluonts.model.evaluation import evaluate_model
from gluonts.ev.metrics import (  # using the same metrics as the tempoPFN evaluation
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.transform import Transformation
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.dataset.common import ProcessDataEntry

from enum import Enum
from typing import Iterator
from pathlib import Path
from collections.abc import Iterable
from functools import cached_property
from toolz import compose
from pandas.tseries.frequencies import to_offset
from sympy import Q

import logging 
import warnings
from linear_operator.utils.cholesky import NumericalWarning

# --- Setup Logging as in tempoPFN ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Ensure this module's logger actually outputs INFO-level logs even if
# logging was configured earlier by imported libraries (which can make
# basicConfig a no-op or set higher levels).
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False


# Filter out specific gluonts warnings
class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter: str) -> None:
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record: logging.LogRecord) -> bool:
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(WarningFilter("The mean prediction is not stored in the forecast data"))

# Filter out numerical warnings
warnings.filterwarnings("ignore", category=NumericalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# we borrow this from TempoPFN's repo.
# Please check their submission notebook at https://github.com/automl/TempoPFN/blob/main/examples/gift_eval/gift_eval_submission.ipynb 

# Constants and classes
TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
    "h": 48,
    "Y": 6,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "h": 48,
    "T": 48,
    "S": 60,
    "s": 60,
    "min": 48,
}

TFB_PRED_LENGTH_MAP = {
    "A": 6,
    "Y": 6,
    "H": 48,
    "h": 48,
    "Q": 8,
    "D": 14,
    "M": 18,
    "W": 13,
    "U": 8,
    "T": 8,
    "min": 8,
    "us": 8,
}

METRICS = (
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    )


class Term(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        elif self == Term.LONG:
            return 15


def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry


class MultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(self, data_it: Iterable[DataEntry], is_train: bool = False) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry


class Dataset:
    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        to_univariate: bool = False,
        storage_path: str = None,
        max_windows: int | None = None,
    ):
        storage_path = Path(storage_path)
        self.hf_dataset = datasets.load_from_disk(str(storage_path / name)).with_format("numpy")
        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)
        if to_univariate:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(self.gluonts_dataset)

        self.term = Term(term)
        self.name = name
        self.max_windows = max_windows if max_windows is not None else MAX_WINDOW

    @cached_property
    def prediction_length(self) -> int:
        freq = norm_freq_str(to_offset(self.freq).name)
        if freq.endswith("E"):
            freq = freq[:-1]
        pred_len = M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        return target.shape[0] if len((target := self.hf_dataset[0]["target"]).shape) > 1 else 1

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        if "past_feat_dynamic_real" not in self.hf_dataset[0]:
            return 0
        elif len((past_feat_dynamic_real := self.hf_dataset[0]["past_feat_dynamic_real"]).shape) > 1:
            return past_feat_dynamic_real.shape[0]
        else:
            return 1

    @cached_property
    def windows(self) -> int:
        if "m4" in self.name:
            return 1
        w = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, w), self.max_windows)

    @cached_property
    def _min_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(pc.list_flatten(pc.list_slice(self.hf_dataset.data.column("target"), 0, 1)))
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return min(lengths.to_numpy())

    @cached_property
    def sum_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(pc.list_flatten(self.hf_dataset.data.column("target")))
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return sum(lengths.to_numpy())

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(self.gluonts_dataset, offset=-self.prediction_length * (self.windows + 1))
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(self.gluonts_dataset, offset=-self.prediction_length * self.windows)
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        _, test_template = split(self.gluonts_dataset, offset=-self.prediction_length * self.windows)
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data


class TempoPredictorWrapper(Predictor):
    """Wrapper to make TimeSeriesForecaster compatible with GluonTS Predictor interface.
    """

    def __init__(self, forecaster: TimeSeriesForecaster,
                 prediction_length: int,
                 context_length: int,
                 time_dim: int = 6,
                 batch_size: int = 64):
        self.forecaster = forecaster
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.time_dim = time_dim
        self.quantiles = forecaster.quantiles
        self.batch_size = batch_size

    def _convert_to_npbatchc(self, items: list[dict]) -> NpBatchTSContainer:
        batch_size = len(items)
        history_values_list = []
        start_dates = []
        frequencies = []

        for entry in items:
            target = entry["target"]

            if target.ndim == 1:
                # target is (seq_len,), then (seq_len, dim) where dim =1
                target = target.reshape(-1, 1)
            else:
                # (n_channels, seq_len) -> (seq_len, n_channels)
                target = target.T

            if target.ndim == 2:
                # we add a "dim" dimension at the end because our model expects that.
                target = np.expand_dims(target, -1)
            # target at this point is of shape (seq_len, n_channels, dim)

            if self.context_length is not None and len(target) > self.context_length:
                target = target[-self.context_length:]

            history_values_list.append(target)
            start_dates.append(entry["start"].to_timestamp().to_datetime64())
            frequencies.append(parse_frequency(entry["freq"]))

        # debug_msg = [str(h.shape[0]) for h in history_values_list[:4]]
        # min_shape = min([h.shape[0] for h in history_values_list])
        # print(f'\n\n =-================= \n {', '.join(debug_msg)} and min h_len is {min_shape} \n -=-------------------- \n\n')

        # (batch_size, seq_len, n_channels, dim)
        history_values = np.stack(history_values_list, axis=0)
        n_channels = history_values.shape[2]

        future_values = np.zeros(
            (batch_size, self.prediction_length, n_channels),
            dtype=np.float32
        )

        history_tf, future_tf = compute_batch_time_features(
            start=start_dates,
            history_length=self.context_length,
            future_length=self.prediction_length,
            batch_size=batch_size,
            frequency=frequencies,
            K_max=self.time_dim,
            include_extra=True,
        )

        return NpBatchTSContainer(
            history=history_values,
            future=future_values,
            start=start_dates,
            frequency=frequencies,
            history_time_features=history_tf,
            future_time_features=future_tf
        )

    def _handle_missing_data(self, x):
        return x

    def _predict_batch(self, batch: NpBatchTSContainer) -> np.ndarray:
        x = jnp.array(batch.history)
        t_hist = jnp.array(batch.history_time_features)
        t_future = jnp.array(batch.future_time_features)
        x = self._handle_missing_data(x)
        x_scaled, (m, iqr) = self.forecaster.robust_scaler.scale(x)
        preds = self.forecaster.model.apply(
            self.forecaster.model_state.params, x_scaled, t_hist, t_future, training=False)
        preds = self.forecaster.robust_scaler.inverse_scale(preds, medians=m, iqrs=iqr)
        return np.asarray(preds)

    def _to_quantile_forecast(self, preds, h_len, test_batch) -> list[QuantileForecast]:

        forecasts = []

        # pred is of shape (batch_size, f_len, n_channels, n_quantiles)
        for i in range(len(test_batch)):
            pred = np.array(preds[i])  # (f_len, n_channels, n_quantiles)
            pred = pred.transpose(2, 0, 1)  # (n_quantiles, f_len, n_channels)
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)  # if n_channels =1 then we make pred (n_quantiles, f_len)

            test_item = test_batch[i]
            start = test_item['start']
            item_id = test_item['item_id']
            forecast_start = start + h_len
            forecast = QuantileForecast(
                forecast_arrays=pred,
                start_date=forecast_start,
                forecast_keys=[str(q.item())
                               for q in self.quantiles],
                item_id=item_id
            )
            forecasts.append(forecast)
        return forecasts

    def predict(self, dataset) -> Iterator[QuantileForecast]:
        """Generate forecasts for the test data."""

        def get_h_len(item):
            target = item['target']
            if target.ndim == 1:
                # target is (seq_len,), then (seq_len, dim) where dim =1
                seq_len = target.shape[0]
            else:
                # (n_channels, seq_len) -> (seq_len, n_channels)
                seq_len = target.shape[1]
            h_len = min(seq_len, self.context_length)
            return h_len

        dataset_list = list(dataset)

        # Process in batches and yield forecasts
        for i in range(0, len(dataset_list), self.batch_size):
            items = dataset_list[i: i+self.batch_size]
            batch = self._convert_to_npbatchc(items)
            preds = self._predict_batch(batch)
            h_len = get_h_len(items[0])
            batch_forecasts = self._to_quantile_forecast(preds, h_len, items)
            
            # Yield each forecast individually
            for forecast in batch_forecasts:
                yield forecast


def main(dataset_name, storage_path, config_path, output_dir, context_length=3072, seed=42):
    dataset = Dataset(name=dataset_name, storage_path=storage_path)

    # Load config
    cfg = load_config(config_path)
    training_config = cfg_to_training_config(cfg)

    # Create forecaster
    forecaster = TimeSeriesForecaster(config=training_config, seed=seed)
    logger.info(f'Loaded model from {cfg.training.checkpoint_path}')

    # Create predictor
    predictor = TempoPredictorWrapper(
        forecaster=forecaster,
        prediction_length=dataset.prediction_length,
        context_length=context_length,  # Your model's max context
        time_dim=training_config.time_dim
    )

    seasonality = get_seasonality(dataset.freq)

    # Evaluate
    logger.info("Evaluating...")
    metrics_df = evaluate_model(
        model=predictor,
        test_data=dataset.test_data,
        metrics=METRICS,
        seasonality=seasonality,
        axis=None
    )

    logger.info("\n=== Evaluation Results ===")
    logger.info(metrics_df.to_string())

    # Convert metrics to dict
    results = {
        'dataset': dataset_name,
        'model': 'JAX-TimeSeriesForecaster',
        'weights': cfg.training.checkpoint_path,  #storing some info for reproducibility
        'config': config_path,
        'metrics': metrics_df.iloc[0].to_dict()
    }

    if not os.path.exists(output_dir):
         os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f'{dataset_name.replace('/', '_')}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}\n")
    logger.info("""⠀⠀⠀⠀⢱⣆⠀⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠀⠈⣿⣷⡀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠀⢸⣿⣿⣷⣧⠀⠀⠀
                ⠀⠀⠀⠀⡀⢠⣿⡟⣿⣿⣿⡇⠀⠀
                ⠀⠀⠀⠀⣳⣼⣿⡏⢸⣿⣿⣿⢀⠀
                ⠀⠀⠀⣰⣿⣿⡿⠁⢸⣿⣿⡟⣼⡆
                ⢰⢀⣾⣿⣿⠟⠀⠀⣾⢿⣿⣿⣿⣿
                ⢸⣿⣿⣿⡏⠀⠀⠀⠃⠸⣿⣿⣿⡿
                ⢳⣿⣿⣿⠀⠀⠀⠀⠀⠀⢹⣿⡿⡁
                ⠀⠹⣿⣿⡄⠀⠀⠀⠀⠀⢠⣿⡞⠁
                ⠀⠀⠈⠛⢿⣄⠀⠀⠀⣠⠞⠋⠀⠀
                ⠀⠀⠀⠀⠀⠀⠉⠀⠀⠀⠀⠀⠀⠀""")


if __name__ == '__main__':
    parser = ArgumentParser('Evaluation of gift-eval')
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="electricity/H",
        help="Dataset to evaluate",
    )
    parser.add_argument(
        '--storage_path',
        default='/Users/mariana/Documents/projects/ts/gift-eval-data',
        type=str,
        help='Path to gift-eval datasets, needs downloading.'
    )
    parser.add_argument(
        '--config_path',
        default='../conf/training.yaml',
        type=str,
        help='Path to config.'
    )
    parser.add_argument(
        '--output_dir',
        default='results/',
        type=str,
        help='Path to gift-eval datasets, needs downloading.'
    )
    
    args = parser.parse_args()

    main(args.dataset_name, args.storage_path, args.config_path, args.output_dir)
