#!/usr/bin/env python3
"""Migrate existing split time features (history/future) to full-length features."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm


def migrate_file(input_file: Path, output_file: Path):
    """Merge history and future features into full-length features."""
    print(f"Loading {input_file}...")
    with pa.ipc.open_file(str(input_file)) as reader:
        table = reader.read_all()
    df = table.to_pandas()

    print("Concatenating features...")
    time_features_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        history = np.array(row['history_features'])
        future = np.array(row['future_features'])
        time_features = np.concatenate([history, future], axis=0)
        time_features_list.append(time_features.tolist())

    # Create output with only series_id and time_features
    output_df = pd.DataFrame({
        'series_id': df['series_id'],
        'time_features': time_features_list
    })

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_file}...")

    schema = pa.schema([
        ("series_id", pa.int64()),
        ("time_features", pa.list_(pa.list_(pa.float64())))
    ])

    output_table = pa.Table.from_pandas(output_df, schema=schema)
    with pa.ipc.new_file(str(output_file), schema) as writer:
        writer.write_table(output_table)

    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input file with split features')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output file for merged features')
    args = parser.parse_args()

    migrate_file(Path(args.input_file), Path(args.output_file))


if __name__ == '__main__':
    main()
