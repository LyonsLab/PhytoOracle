#!/usr/bin/env python3

import json
import sys
import os
import subprocess

description = """
bundle up X number of data sets into one bundle, and generate a json file for the bundle list
"""

def main():
    try:
        args = parse_args()
        file_list = read_file_list_json(args["in_json"])
        bundle_list = bundle_data(file_list, args["data_per_bundle"])
        write_to_file(args["out_json"], bundle_list)
    except:
        print_help()
        raise

def parse_args():
    if len(sys.argv) != 4:
        raise Exception("Wrong number of args, need 3")
    args = {}
    args["in_json"] = sys.argv[1]
    args["out_json"] = sys.argv[2]
    args["data_per_bundle"] = int(sys.argv[3])
    return args

def print_help():
    print(description)
    print("Usage:")
    print("./gen_bundle_list.py <raw_data_files.json> <bundle_list.json> <num_data_set_per_bundle>")
    print()

def read_file_list_json(json_filename):
    with open(json_filename, "r") as infile:
        json_obj = json.load(infile)
        return json_obj["DATA_FILE_LIST"]

def bundle_data(file_list, data_per_bundle):
    data_sets = []
    bundle_list = []
    for index, file in enumerate(file_list):
        if index % data_per_bundle == 0 and index != 0:
            bundle = {}
            bundle["DATA_SETS"] = data_sets
            bundle["ID"] = len(bundle_list)
            bundle_list.append(bundle)
            data_sets = []
        data_sets.append(file)
    bundle = {}
    bundle["DATA_SETS"] = data_sets
    bundle["ID"] = len(bundle_list)
    bundle_list.append(bundle)
    return bundle_list

def write_to_file(out_filename, bundle_list):
    json_obj = {}
    json_obj["BUNDLE_LIST"] = bundle_list
    with open(out_filename, "w") as outfile:
        dump_str = json.dumps(json_obj, indent=2, sort_keys=True)
        #print(dump_str)
        outfile.write(dump_str)

main()
