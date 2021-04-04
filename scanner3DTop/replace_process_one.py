#!/usr/bin/env python3
"""
Author : emmanuelgonzalez
Date   : 2021-01-17
Purpose: Rock the Casbah
"""

import argparse
import os
import sys


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('positional',
                        metavar='str',
                        help='A positional argument')

    args = parser.parse_args()

    if '/' not in args.positional[-1]:

        args.positional = args.positional + '/'

    return args


# --------------------------------------------------
def replace_file_one(line2replace, date):
    f = open("process_one_set.sh", 'r')
    fdata = f.read()
    f.close()

    nline = fdata.replace(line2replace, date)
    f = open("process_one_set.sh", 'w')
    f.write(nline)
    f.close()

    #print("Change complete")


# --------------------------------------------------
def replace_file_two(line2replace, date):
    f = open("process_one_set2.sh", 'r')
    fdata = f.read()
    f.close()

    nline = fdata.replace(line2replace, date)
    f = open("process_one_set2.sh", 'w')
    f.write(nline)
    f.close()

    #print("Change complete")


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()

    #print(args.positional)

    with open('process_one_set.sh', 'r') as f:
        line_list = f.readlines()

        for line in line_list:

            if 'HPC_PATH=' in line:
                replacing = line.split('=')[-1]

                replace_file_one(replacing, f'"{args.positional}"\n')
                #replace_file_two(replacing, f'"{args.positional}"\n')


# --------------------------------------------------
if __name__ == '__main__':
    main()
