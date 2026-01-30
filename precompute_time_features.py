#!/usr/bin/env python3
"""Precompute time features and save to disk."""

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
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
    full_length: int,
    K_max: int,
    batch_size: int = 64,
):
    """Precompute time features for a single file."""
    # Load data
    table = pa.ipc.open_file(input_file).read_all()
    df = table.to_pandas()

    feature_generator = TimeFeatureGenerator()
    records = []

    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    with open(output_file, "wb") as f:
        schema = pa.schema([
            ("series_id", pa.int64()),
            ("start", pa.timestamp('ns')),
            ("frequency", pa.string()),
            ("time_features", pa.list_(pa.list_(pa.float64())))  # Full-length features
        ])

        with pa.ipc.new_file(f, schema) as writer:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {input_file.name}", leave=False):
                series_id = row.series_id
                frequency = parse_frequency(row['frequency'])
                freq_str = frequency.to_pandas_freq(for_date_range=True)
                period_freq_str = frequency.to_pandas_freq(for_date_range=False)

                start_ts = pd.Timestamp(row['start'])
                if not validate_frequency_safety(start_ts, full_length, frequency):
                    start_ts = BASE_START_DATE

                # Create full-length range
                date_range = pd.date_range(start=start_ts, periods=full_length, freq=freq_str)

                if date_range[-1] > BASE_END_DATE:
                    safe_start = BASE_END_DATE - pd.tseries.frequencies.to_offset(freq_str) * full_length
                    if safe_start < BASE_START_DATE:
                        safe_start = BASE_START_DATE
                    date_range = pd.date_range(start=safe_start, periods=full_length, freq=freq_str)

                # Compute features for full length
                period_idx = date_range.to_period(period_freq_str)
                time_features = feature_generator.compute_features(period_idx, date_range, freq_str)
                time_features = _pad_or_truncate_features(time_features, K_max)

                record = {
                    'series_id': series_id,
                    'start': row['start'],
                    'frequency': row['frequency'],
                    'time_features': time_features.tolist()
                }
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


def process_gen_type(gen_type, root_path, output_path, full_length, K_max):
    """Process all files for a single generator type."""
    input_dir = root_path / gen_type
    if not input_dir.exists():
        return

    arrow_files = list(input_dir.glob('*.arrow'))
    print(f"\nProcessing {len(arrow_files)} files from {gen_type}...")

    for arrow_file in tqdm(arrow_files, desc=gen_type):
        output_file = output_path / gen_type / arrow_file.name
        precompute_for_file(
            arrow_file,
            output_file,
            full_length,
            K_max,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', type=str, default='data', help='Root data directory')
    parser.add_argument('--output-path', type=str, default='data_timefeatures', help='Output directory')
    parser.add_argument('--full-length', type=int, default=2048, help='Full sequence length')
    parser.add_argument('--K-max', type=int, default=6, help='Max time features')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()

    root_path = Path(args.root_path)
    output_path = Path(args.output_path)

    gen_types = ['gp', 'kernel', 'sinewave', 'sawtooth', 'step', 'spike', 'anomaly', 'ou_process']

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_gen_type,
                gen_type,
                root_path,
                output_path,
                args.full_length,
                args.K_max,
            )
            for gen_type in gen_types
        ]
        for future in futures:
            future.result()

    print(f"\nDone! Time features saved to {output_path}")


if __name__ == '__main__':
    main()
