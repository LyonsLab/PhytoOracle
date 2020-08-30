#!/usr/bin/env python3
"""
Author : emmanuelgonzalez
Date   : 2020-07-08
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

    parser.add_argument('file',
                        metavar='str',
                        help='File where replacing is needed')

    parser.add_argument('-s',
                        '--sd',
                        help='Date of the scan being processed',
                        metavar='str',
                        type=str,
                        required=True)

    # parser.add_argument('-i',
    #                     '--int',
    #                     help='A named integer argument',
    #                     metavar='int',
    #                     type=int,
    #                     default=0)

    # parser.add_argument('-f',
    #                     '--file',
    #                     help='A readable file',
    #                     metavar='FILE',
    #                     type=argparse.FileType('r'),
    #                     default=None)

    # parser.add_argument('-o',
    #                     '--on',
    #                     help='A boolean flag',
    #                     action='store_true')

    return parser.parse_args()


# --------------------------------------------------
def replace(line2replace, date, fil):
    f = open(fil, 'r')
    fdata = f.read()
    f.close()
    nline = fdata.replace(line2replace, date+ '\n')
    #print(nline)
    f = open(fil, 'w')
    f.write(nline)
    f.close()
    print("Change complete")


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()
    with open(args.file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '.csv' in line:
                end = '_coordinates_CORRECTED.csv'
                replacement = 'GPS_CSV=${HPC_PATH}${ORTHO_OUT}' + f'"{args.sd}/' + f'{args.sd}' + f'{end}"'
                # print(f'Original: {line}')
                # print(f'Replacement: {replacement}')
                replace(line, replacement, args.file)
    #     global line2replace
    #     line2replace = lines.split()[2]
    #     print("Replacing: "+line2replace)
    # replace(line2replace, date)


# --------------------------------------------------
if __name__ == '__main__':
    main()
