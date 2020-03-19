#!/usr/bin/env python3
"""
Purpose: Quickly change the date within PhytoOracle's entrypoint.
Run: ./search_replace.py <date> (format example: 2020-01-10)
Author:  Emmanuel Gonzalez
"""
import fire
import os 
import sys

def find(date):
    with open("entrypoint.sh", 'r') as f:
        lines = f.readlines()[2]
        print(lines)
        global line2replace
        line2replace = lines.split()[2]
        print("Replacing: "+line2replace)
    replace(line2replace, date)

def replace(line2replace, date):
    f = open("entrypoint.sh", 'r')
    fdata = f.read()
    f.close()
    nline = fdata.replace(line2replace, date)
    f = open("entrypoint.sh", 'w')
    f.write(nline)
    f.close()
    print("Change complete")

if __name__ == "__main__":
    fire.Fire(find)