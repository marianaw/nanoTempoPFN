from dataclasses import dataclass
from pathlib import Path
import random

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
                 batches_per_shard: int = 2
                 ):
        self.root_path = root_path
        self.gen_types = ['gp', 'kernel', 'sinewave',
                          'sawtooth', 'step', 'spike', 'anomaly', 'ou_process']
        self.batches_per_shard = batches_per_shard
        self.files_list = self._get_files_list()
        self.current_file_index = 0

    def _get_files_list(self) -> list[Path]:
        files_list = []
        for gen_type in self.gen_types:
            path = self.root_path / gen_type
            files = list(path.glob('*.arrow'))
            files_list.extend(files)
        return files_list

    def load_in_memory(self) -> pd.DataFrame:
        try:
            file = self.files_list[self.current_file_index]
            self.current_file_index += 1
            return load_arrow_file(file)
        except IndexError:
            raise StopIteration("No more files to load")

    def shuffle_files_list_and_reset_index(self):
        random.shuffle(self.files_list)
        self.current_file_index = 0


class DataLoader:

    def __init__(self, dataset: ShardedDataset, batch_size: int = 32,
                 full_length: int = 2048, future_length: int = 512, seed: int = 42,
                 time_features_path: Path | None = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.history_length = full_length - future_length
        self.future_length = future_length
        self.time_features_path = time_features_path

        # Set the seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        self.df = pd.DataFrame()  # Empty until __iter__ is called
        self.tf_df = pd.DataFrame()  # Time features dataframe

    def __len__(self):
        """Return approximate number of batches per epoch."""
        return len(self.dataset.files_list) * self.dataset.batches_per_shard

    def __iter__(self):
        """Called at the start of each `for batch in loader:` loop."""
        self.dataset.shuffle_files_list_and_reset_index()
        self._load_dataset_from_memory()
        return self

    def _load_dataset_from_memory(self):
        # Load next file, convert to pandas
        table = self.dataset.load_in_memory()  # Raises StopIteration when exhausted
        df = table.to_pandas()
        df['values'] = df['values'].map(lambda x: np.stack(x))
        self.df = df

        # Load time features if path is specified
        if self.time_features_path is not None:
            current_file = self.dataset.files_list[self.dataset.current_file_index - 1]
            tf_file = self.time_features_path / current_file.parent.name / current_file.name
            if tf_file.exists():
                tf_table = load_arrow_file(tf_file)
                self.tf_df = tf_table.to_pandas()
            else:
                self.tf_df = pd.DataFrame()

    def __next__(self) -> NpBatchTSContainer:
        # Not enough samples left, try loading next file
        while len(self.df) < self.batch_size:
            self._load_dataset_from_memory()  # Raises StopIteration when no more files

        batch_df = self.df.sample(n=self.batch_size)
        # Drop sampled rows to avoid repetition
        self.df = self.df.drop(batch_df.index)

        return self._to_np_container(batch_df)

    def _to_np_container(self, batch_df: pd.DataFrame) -> NpBatchTSContainer:
        # Stack values into array: (batch_size, seq_len, 1)
        values = np.stack(batch_df["values"].values)
        values = np.transpose(values, (0, 2, 1))

        if values.ndim == 2:
            values = values[:, :, None]  # (batch_size, seq_len, 1) if univariate. We add the channel dimension.

        values = values[:, :, None]  # (batch_size, seq_len, n_channels, 1) if univariate. We explicitely
            # add a "univariate" feature dimension so that we can apply the embeddings without breaking shapes.

        history = values[:, :self.history_length, :]
        future = values[:, self.history_length:self.history_length + self.future_length, :]

        start = batch_df["start"].tolist()
        frequency = [parse_frequency(f) for f in batch_df["frequency"].tolist()]

        # Extract time features if available
        history_tf = None
        future_tf = None
        if not self.tf_df.empty:
            tf_batch = self.tf_df[self.tf_df['start'].isin(batch_df['start']) &
                                   self.tf_df['frequency'].isin(batch_df['frequency'])]
            if len(tf_batch) == len(batch_df):
                history_tf_list = []
                future_tf_list = []
                for _, row in tf_batch.iterrows():
                    h_shape = tuple(row['history_shape'])
                    f_shape = tuple(row['future_shape'])
                    history_tf_list.append(np.frombuffer(row['history_features'], dtype=np.float64).reshape(h_shape))
                    future_tf_list.append(np.frombuffer(row['future_features'], dtype=np.float64).reshape(f_shape))
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
