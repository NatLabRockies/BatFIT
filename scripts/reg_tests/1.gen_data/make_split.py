"""Assemble the generated dataset and write the 3-way train/test/val split."""

import argparse
import os

from batfit.utils.data_utils import assemble_all_data, split_dataset_from_np


def main():
    parser = argparse.ArgumentParser(
        description="Assemble sols.pkl and write the train/test/val split"
    )
    parser.add_argument("-folder_save", required=True)
    parser.add_argument("-n_points", type=int, default=256)
    parser.add_argument("-cyc_mode", default="discharge")
    parser.add_argument("-target_mode", default="phi")
    parser.add_argument("-test_split", type=float, default=0.1)
    parser.add_argument("-val_split", type=float, default=0.1)
    parser.add_argument("-seed", type=int, default=42)
    args = parser.parse_args()

    X_data, Y_data = assemble_all_data(
        args.folder_save,
        n_points=args.n_points,
        combined_pickle_file="sols.pkl",
        target_mode=args.target_mode,
        save_data=True,
        cyc_mode=args.cyc_mode,
        save_path=args.folder_save,
    )
    # Battery-level 3-way split, cached to data_split.npz for all downstream
    # steps to reuse (no scaling here — scalers are fit per model type).
    split_dataset_from_np(
        X_data,
        Y_data,
        test_split=args.test_split,
        val_split=args.val_split,
        save_path=args.folder_save,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
