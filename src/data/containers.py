from dataclasses import dataclass
from pathlib import Path
import random
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np
import jax.numpy as jnp
import chex
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq

from .frequency import Frequency, parse_frequency


def load_arrow_file(path):
    """
    Load a .arrow, .feather, or .parquet file using PyArrow.
    Returns: pyarrow.Table
    """
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext == ".arrow":
        with pa.ipc.open_file(str(file_path)) as reader:
            return reader.read_all()
    elif ext == ".feather":
        return feather.read_table(str(file_path))
    elif ext == ".parquet":
        return pq.read_table(str(file_path))
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


class BatchContainerLoader(ABC):
    """Abstract base class for loaders that yield NpBatchTSContainer.

    Loaders should handle reshuffling/regeneration automatically when
    __iter__ is called multiple times (e.g., across epochs).
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator['NpBatchTSContainer']:
        """Yield batches as NpBatchTSContainer."""
        pass


@dataclass
class NpBatchTSContainer:

    history: np.ndarray
    future: np.ndarray
    start: list[np.datetime64]
    frequency: list[Frequency]
    history_mask: np.ndarray | None = None
    future_mask: np.ndarray | None = None
    history_time_features: np.ndarray | None = None
    future_time_features: np.ndarray | None = None

    @property
    def batch_size(self) -> int:
        return self.history.shape[0]

    @property
    def history_length(self) -> int:
        return self.history.shape[1]

    @property
    def future_length(self) -> int:
        return self.future.shape[1]

    @property
    def num_channels(self) -> int:
        return self.history.shape[2]


class ShardedDataset:

    def __init__(self,
                 root_path: Path = Path("data"),
                 time_features_path: Path | None = None
                 ):
        self.root_path = root_path
        self.time_features_path = time_features_path
        self.gen_types = ['gp', 'kernel', 'sinewave',
                          'sawtooth', 'step', 'spike', 'anomaly', 'ou_process']
        self.files_list = self._get_files_list()
        self.current_file_index = 0

    def _get_files_list(self) -> list[Path]:
        files_list = []
        for gen_type in self.gen_types:
            path = self.root_path / gen_type
            files = list(path.glob('*.arrow'))
            files_list.extend(files)
        return files_list

    def load_in_memory(self) -> tuple[pa.Table, pa.Table | None]:
        """Load data and optionally time features. Returns (data_table, tf_table)."""
        try:
            file = self.files_list[self.current_file_index]
            self.current_file_index += 1

            data_table = load_arrow_file(file)

            # Load time features if path is provided
            tf_table = None
            if self.time_features_path is not None:
                tf_file = self.time_features_path / file.parent.name / file.name
                if tf_file.exists():
                    tf_table = load_arrow_file(tf_file)

            return data_table, tf_table
        except IndexError:
            raise StopIteration("No more files to load")

    def shuffle_files_list_and_reset_index(self):
        random.shuffle(self.files_list)
        self.current_file_index = 0


GIFT_EVAL_FORECAST_LENGTHS = {
    48: 5,
    720: 38,
    480: 38,
    30: 3,
    300: 16,
    8: 2,
    120: 3,
    450: 8,
    80: 8,
    12: 2,
    900: 10,
    180: 3,
    600: 10,
    60: 3,
    210: 3,
    195: 3,
    140: 3,
    130: 3,
    14: 1,
    18: 1,
    13: 1,
    6: 1,
}


def _sample_future_length():
    return random.choices(list(GIFT_EVAL_FORECAST_LENGTHS.keys()),
                          weights=list(GIFT_EVAL_FORECAST_LENGTHS.values()))[0]


class DataLoader:

    def __init__(self, dataset: ShardedDataset, batch_size: int = 32,
                 full_length: int = 2048, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.full_length = full_length

        # Set the seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        self.df = pd.DataFrame()  # Empty until __iter__ is called
        self.tf_df = pd.DataFrame()  # Time features dataframe
        self._total_records = None  # Lazy computation of total records

    def _compute_total_records(self) -> int:
        """Compute total records across all sharded files."""
        total_records = 0
        for file_path in self.dataset.files_list:
            table = load_arrow_file(file_path)
            total_records += len(table)
            del table  # Free memory immediately after counting
        return total_records

    def __len__(self):
        """Return number of batches per epoch."""
        if self._total_records is None:
            self._total_records = self._compute_total_records()
        return self._total_records // self.batch_size

    def __iter__(self):
        """Called at the start of each `for batch in loader:` loop."""
        self.dataset.shuffle_files_list_and_reset_index()
        self._load_dataset_from_memory()
        return self

    def _load_dataset_from_memory(self):
        # Load data and time features from dataset
        data_table, tf_table = self.dataset.load_in_memory()

        # Convert data to pandas
        df = data_table.to_pandas()
        df['values'] = df['values'].map(lambda x: np.stack(x))
        self.df = df

        # Convert time features to pandas if available
        if tf_table is not None:
            tf_df = tf_table.to_pandas()
            tf_df['time_features'] = tf_df.time_features.map(lambda x: np.stack(x))
            self.tf_df = tf_df
        else:
            self.tf_df = pd.DataFrame()

    def __next__(self) -> NpBatchTSContainer:
        # Not enough samples left, try loading next file
        while len(self.df) < self.batch_size:
            self._load_dataset_from_memory()  # Raises StopIteration when no more files

        batch_df = self.df.sample(n=self.batch_size)

        # Sample time features with same indices
        batch_tf_df = None
        if not self.tf_df.empty:
            batch_tf_df = self.tf_df.loc[batch_df.index]
            self.tf_df = self.tf_df.drop(batch_df.index)

        # Drop sampled rows to avoid repetition
        self.df = self.df.drop(batch_df.index)

        return self._to_np_container(batch_df, batch_tf_df)

    def _to_np_container(self, batch_df: pd.DataFrame, batch_tf_df: pd.DataFrame | None = None) -> NpBatchTSContainer:
        # Stack values into array: (batch_size, seq_len, 1)
        values = np.stack(batch_df["values"].values)
        values = np.transpose(values, (0, 2, 1))

        if values.ndim == 2:
            # (batch_size, seq_len, 1) if univariate. We add the channel dimension.
            values = values[:, :, None]

        # (batch_size, seq_len, n_channels, 1) if univariate. We explicitely
        values = values[:, :, None]
        # add a "univariate" feature dimension so that we can apply the embeddings without breaking shapes.

        # Sample future length and split
        future_length = _sample_future_length()
        history_length = self.full_length - future_length
        history = values[:, :history_length, :]
        future = values[:, history_length:history_length + future_length, :]

        start = batch_df["start"].tolist()
        frequency = [parse_frequency(f)
                     for f in batch_df["frequency"].tolist()]

        # Split time features using the same history/future lengths
        history_tf = None
        future_tf = None
        if batch_tf_df is not None:
            history_tf_list = []
            future_tf_list = []
            for _, row in batch_tf_df.iterrows():
                time_features = np.array(row['time_features'])
                history_tf_list.append(time_features[:history_length])
                future_tf_list.append(time_features[history_length:history_length + future_length])

            history_tf = np.stack(history_tf_list)
            future_tf = np.stack(future_tf_list)

        return NpBatchTSContainer(
            history=history,
            future=future,
            start=start,
            frequency=frequency,
            history_time_features=history_tf,
            future_time_features=future_tf,
        )
