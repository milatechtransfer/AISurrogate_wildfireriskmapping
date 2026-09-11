import argparse
import os

import pandas as pd

from data_preparation.utils import get_processed_hex_ids


def get_selected_hexel_metadata(data_dir: str, hex_ids: list) -> pd.DataFrame:
    """
    Combine all the selected dfs for the train/val/eval sets
    """
    df_list = [pd.read_csv(os.path.join(data_dir, f"meta_hex_{hex_id}.csv")) for hex_id in hex_ids]
    df = pd.concat(df_list, ignore_index=True)
    return df


def get_data_splits(data_dir: str, val_hex_id: list, test_hex_id: list):
    """Split the data into train, val and test sets using the metadata csvs created"""
    # Get the train ids
    all_hex_ids = get_processed_hex_ids(data_dir)
    train_hex_ids = list(set(all_hex_ids) - set(val_hex_id) - set(test_hex_id))

    train_df = get_selected_hexel_metadata(data_dir, hex_ids=train_hex_ids)
    val_df = get_selected_hexel_metadata(data_dir, hex_ids=val_hex_id)
    eval_df = get_selected_hexel_metadata(data_dir, hex_ids=test_hex_id)

    print("Total number of samples in train, val, eval are:", len(train_df), len(val_df), len(eval_df))

    train_df.to_csv(os.path.join(data_dir, "train_indices.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val_indices.csv"), index=False)
    eval_df.to_csv(os.path.join(data_dir, "test_indices.csv"), index=False)


def get_test_split_only(data_dir: str, test_hex_id: list | None = None) -> None:
    """Create only the test CSV, with empty train and val CSVs.

    Use this when all available data is test data (e.g. a new deployment region
    with no training hexels). If ``test_hex_id`` is None, all hexels in
    ``data_dir`` are used as test data. An empty train_indices.csv and
    val_indices.csv are still written so that downstream code that expects all
    three files does not fail.
    """
    if test_hex_id is None:
        test_hex_id = list(get_processed_hex_ids(data_dir))
        print(f"No --test_hex_id provided; using all {len(test_hex_id)} hexels as test.")

    eval_df = get_selected_hexel_metadata(data_dir, hex_ids=test_hex_id)
    print(f"Total number of test samples: {len(eval_df)}")

    eval_df.to_csv(os.path.join(data_dir, "test_indices.csv"), index=False)

    # Write empty stubs so downstream tooling finds all three split files.
    empty = eval_df.iloc[0:0]  # zero rows, same columns
    empty.to_csv(os.path.join(data_dir, "train_indices.csv"), index=False)
    empty.to_csv(os.path.join(data_dir, "val_indices.csv"), index=False)
    print("Written empty train_indices.csv and val_indices.csv.")


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test data splits for wildfire hexels.")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the preprocessed data samples.",
    )

    parser.add_argument(
        "--val_hex_id",
        type=str,
        nargs="+",
        default=None,
        help="List of hex IDs to use for the validation set: ids separated by space. Not required with --test_only.",
    )

    parser.add_argument(
        "--test_hex_id",
        type=str,
        nargs="+",
        default=None,
        help="List of hex IDs to use for the test set: ids separated by space. "
        "With --test_only, omit to use all hexels in data_dir as test.",
    )

    parser.add_argument(
        "--test_only",
        action="store_true",
        help="Create only the test split CSV. Use when all data is test data. "
        "Empty train_indices.csv and val_indices.csv are written as stubs.",
    )

    args = parser.parse_args()

    if args.test_only:
        get_test_split_only(
            data_dir=args.data_dir,
            test_hex_id=args.test_hex_id,  # None means "use all"
        )
    else:
        if args.test_hex_id is None:
            parser.error("--test_hex_id is required unless --test_only is set.")
        if args.val_hex_id is None:
            parser.error("--val_hex_id is required unless --test_only is set.")
        get_data_splits(
            data_dir=args.data_dir,
            val_hex_id=args.val_hex_id,
            test_hex_id=args.test_hex_id,
        )


if __name__ == "__main__":
    main()
