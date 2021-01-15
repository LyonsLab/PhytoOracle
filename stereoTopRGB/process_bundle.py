#!/usr/bin/env python3

import json
import sys
import os
import subprocess
import time

description = """
process all data_set within a bundle
"""

def main():
    try:
        args = parse_args()
        bundle = read_bundle_json(args["bundle_json"])
        for data_set in bundle["DATA_SETS"]:
            process_one_set(data_set)
    except:
        print_help()
        raise

def print_help():
    print(description)
    print("Usage:")
    print("./process_bundle.py <bundle_json>")
    print()

def parse_args():
    if len(sys.argv) != 2:
        raise Exception("Wrong number of args, need 1")
    args = {}
    args["bundle_json"] = sys.argv[1]
    return args

# Read the json file contains info for 1 bundle
def read_bundle_json(json_filename):
    with open(json_filename, "r") as outfile:
        bundle_json = json.load(outfile)
        return bundle_json

# Run process_one_set.sh 
# data_set argument is added into the environment
def process_one_set(data_set):
    assert(isinstance(data_set, dict))
    my_env = os.environ.copy()
    my_env.update(data_set) # add data_set into the environment
    try:
        proc = subprocess.check_output(["/bin/bash", "process_one_set.sh"], env=my_env)
    except subprocess.SubprocessError as e:
        print("Error when running for data_set: ", data_set)
        print(e)
        sys.exit(proc.returncode)
        

main()
