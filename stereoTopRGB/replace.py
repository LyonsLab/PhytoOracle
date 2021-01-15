#!/usr/bin/env python3
"""
Author : emmanuelgonzalez
Date   : 2020-05-03
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

    parser.add_argument('replace',
                        metavar='str',
                        help='The string that will be inserted')

    return parser.parse_args()


# --------------------------------------------------
def find():
    args = get_args()
    with open("entrypoint.sh", 'r') as f:
        lines = f.readlines()[2]
        print(lines)
        #global line2replace
        line2replace = lines.split()[2]
        print("Replacing " + line2replace + " with " + args.replace)
	#print(f'Replacing {line2replace} with {args.replace}')
    replace(line2replace, args.replace)


# --------------------------------------------------
def replace(line2replace, date):
    f = open("entrypoint.sh", 'r')
    fdata = f.read()
    f.close()
    nline = fdata.replace(line2replace, date)
    f = open("entrypoint.sh", 'w')
    f.write(nline)
    f.close()
    #print("Change complete")


# --------------------------------------------------
if __name__ == '__main__':
    find()
