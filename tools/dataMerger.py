import json
import pandas as pd
import argparse
import sys


def main():
    """Main function."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--ptx-csv', type=str, default='data/ptx.csv')
    parser.add_argument('--profile-csv', type=str)
    parser.add_argument('--output', type=str, default='data/data.csv')

    args = parser.parse_args()
    print(args)

    ptx_stats = pd.read_csv(args.ptx_csv, header=0)
    profile_stats = pd.read_csv(args.profile_csv, header=0)

    # TODO: need to optimize, apply pd.join or pd.merge
    data = pd.DataFrame()
    for idx, rec in profile_stats.iterrows():
        
        new_rec = {}

        benchmark = rec.benchmark
        kernel = rec.kernel
        ptx_info = ptx_stats[(ptx_stats.benchmark == benchmark) & (ptx_stats['kernel'].str.contains(kernel))].iloc[0]

        profile_dict = rec.to_dict()
        ptx_dict = ptx_info.to_dict()
        del ptx_dict['benchmark']
        del ptx_dict['kernel']

        new_rec.update(profile_dict)
        new_rec.update(ptx_dict)

        data = data.append(new_rec, ignore_index=True)

    print(data.head())
    data.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
