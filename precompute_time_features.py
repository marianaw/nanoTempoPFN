#!/usr/bin/env python3
"""Precompute time features and save to disk."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm

from src.data.time_features import TimeFeatureGenerator, _pad_or_truncate_features
from src.data.frequency import parse_frequency, validate_frequency_safety
from src.data.constants import BASE_START_DATE, BASE_END_DATE


def precompute_for_file(
    input_file: Path,
    output_file: Path,
    history_length: int,
    future_length: int,
    K_max: int,
    batch_size: int = 1,
):
    """Precompute time features for a single file."""
    # Load data
    table = pa.ipc.open_file(input_file).read_all()
    df = table.to_pandas()

    feature_generator = TimeFeatureGenerator()
    total_length = history_length + future_length
    records = []

    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    with open(output_file, "wb") as f:
        schema = pa.schema([
            ("series_id", pa.int64()),
            ("future_features", pa.list_(pa.list_(pa.float64()))),  # 2D array → list of lists
            ("history_features", pa.list_(pa.list_(pa.float64())))
        ])

        with pa.ipc.new_file(f, schema) as writer:        
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {input_file.name}", leave=False):
                series_id = row.series_id
                frequency = parse_frequency(row['frequency'])
                freq_str = frequency.to_pandas_freq(for_date_range=True)
                period_freq_str = frequency.to_pandas_freq(for_date_range=False)

                start_ts = pd.Timestamp(row['start'])
                if not validate_frequency_safety(start_ts, total_length, frequency):
                    start_ts = BASE_START_DATE

                # Create ranges
                history_range = pd.date_range(start=start_ts, periods=history_length, freq=freq_str)

                if history_range[-1] > BASE_END_DATE:
                    safe_start = BASE_END_DATE - pd.tseries.frequencies.to_offset(freq_str) * total_length
                    if safe_start < BASE_START_DATE:
                        safe_start = BASE_START_DATE
                    history_range = pd.date_range(start=safe_start, periods=history_length, freq=freq_str)

                future_start = history_range[-1] + pd.tseries.frequencies.to_offset(freq_str)
                future_range = pd.date_range(start=future_start, periods=future_length, freq=freq_str)

                # Compute features
                history_period_idx = history_range.to_period(period_freq_str)
                future_period_idx = future_range.to_period(period_freq_str)

                history_features = feature_generator.compute_features(history_period_idx, history_range, freq_str)
                future_features = feature_generator.compute_features(future_period_idx, future_range, freq_str)

                history_features = _pad_or_truncate_features(history_features, K_max)
                future_features = _pad_or_truncate_features(future_features, K_max)

                record = {'series_id': series_id, 'future_features': future_features.tolist(), 'history_features': history_features.tolist()}
                records.append(record)

                # Write batch when accumulated enough records
                if len(records) >= batch_size:
                    batch_df = pd.DataFrame(records)
                    batch_table = pa.Table.from_pandas(batch_df)
                    writer.write_table(batch_table)
                    records = []

            # Write remaining records
            if records:
                batch_df = pd.DataFrame(records)
                batch_table = pa.Table.from_pandas(batch_df)

                if writer is None:
                    f = pa.OSFile(str(output_file), 'wb')
                    writer = pa.ipc.new_file(f, batch_table.schema)

                writer.write_table(batch_table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', type=str, default='data', help='Root data directory')
    parser.add_argument('--output-path', type=str, default='data_timefeatures', help='Output directory')
    parser.add_argument('--history-length', type=int, default=1536, help='History length')
    parser.add_argument('--future-length', type=int, default=512, help='Future length')
    parser.add_argument('--K-max', type=int, default=6, help='Max time features')
    args = parser.parse_args()

    root_path = Path(args.root_path)
    output_path = Path(args.output_path)

    gen_types = ['gp', 'kernel', 'sinewave', 'sawtooth', 'step', 'spike', 'anomaly', 'ou_process']

    for gen_type in gen_types:
        input_dir = root_path / gen_type
        if not input_dir.exists():
            continue

        arrow_files = list(input_dir.glob('*.arrow'))
        print(f"\nProcessing {len(arrow_files)} files from {gen_type}...")

        for arrow_file in tqdm(arrow_files, desc=gen_type):
            output_file = output_path / gen_type / arrow_file.name
            precompute_for_file(
                arrow_file,
                output_file,
                args.history_length,
                args.future_length,
                args.K_max,
            )

    print(f"\nDone! Time features saved to {output_path}")


if __name__ == '__main__':
    main()
