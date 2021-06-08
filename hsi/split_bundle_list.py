#!/usr/bin/env python3

import json
import sys

description = """
split the bundle_list.json into multiple json file, each contains info relavant to 1 bundle
"""

def main():
    try:
        args = parse_args()
        bundle_list = read_bundle_list_json(args["bundle_list_json"])
        gen_json_for_all_bundle(bundle_list, args["base_path"])
    except:
        print_help()
        raise

def parse_args():
    if len(sys.argv) != 3:
        raise Exception("Wrong number of args, need 2")
    args = {}
    args["bundle_list_json"] = sys.argv[1]
    args["base_path"] = sys.argv[2]
    if args["base_path"][-1] != '/':
        raise Exception("Base path should end with /")
    return args

def print_help():
    print(description)
    print("Usage:")
    print("./split_bundle_list.py <bundle_list_json> <base_path-for-output>")
    print()

def read_bundle_list_json(json_filename):
    with open(json_filename, "r") as outfile:
        json_obj = json.load(outfile)
        return json_obj["BUNDLE_LIST"]

def gen_json_for_all_bundle(bundle_list, base_path):
    for i in range(len(bundle_list)):
        outfilename = "bundle_{0}.json".format(i)
        with open(base_path + outfilename, "w") as outfile:
            dump_str = json.dumps(bundle_list[i], indent=2, sort_keys=True)
            outfile.write(dump_str)


main()


