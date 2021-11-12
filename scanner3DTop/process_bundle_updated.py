#!/usr/bin/env python3
"""
Author : eg
Date   : 2021-10-30
Purpose: Rock the Casbah
"""

import argparse
import json
import sys
import os
import subprocess
import time

# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('positional',
                        metavar='str',
                        help='A positional argument')

    parser.add_argument('-f',
                        '--file',
                        help='Name of process shell script.',
                        metavar='file',
                        type=str)

    return parser.parse_args()


# --------------------------------------------------
def read_bundle_json(json_filename):
    with open(json_filename, "r") as outfile:
        bundle_json = json.load(outfile)
        return bundle_json


# --------------------------------------------------
def process_one_set(data_set, file_name):
    assert(isinstance(data_set, dict))
    my_env = os.environ.copy()
    my_env.update(data_set) # add data_set into the environment
    try:
        proc = subprocess.check_output(["/bin/bash", file_name], env=my_env)
    except subprocess.SubprocessError:
        print("Error when running for data_set: ", data_set)
        sys.exit(proc.returncode)


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()

    try:
        bundle = read_bundle_json(args.positional)
        for data_set in bundle["DATA_SETS"]:
            process_one_set(data_set, args.file)
    except:
        raise



# --------------------------------------------------
if __name__ == '__main__':
    main()
