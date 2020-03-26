#!/usr/bin/env python

import sys
import subprocess
import json

description = """
generate a json that contains UUID and path-to-data of the data set for all data sets.
print the json to stdout
"""

def get_irods_path_list(base_dir):
    path_list = []
    # list files inside base_dir
    ls_output = subprocess.check_output(["ls", base_dir]).decode("utf-8")
    path_list = ls_output.split()
    for path in path_list:
        path = path.strip()
    return path_list

def find_UUID_for_all_path(base_dir, path_list):
    data_sets = []
    for path in path_list:
        # get all data files in a data set via ls
        files = subprocess.check_output(["ls", base_dir + path]).decode("utf-8")[1:]
        for filename in files.split():
            index = filename.find("_metadata.json")
            # find the UUID from the filename of metadata file
            if index != -1:
                UUID = filename[:index]
                data_sets.append((base_dir + path, UUID))
    return data_sets

def main():
    try:
        parse_args()
    except:
        print_help()
        raise

    # get the base path from cmd line arg
    base_dir = sys.argv[1]
    if(base_dir[-1] != "/"):
        base_dir += "/"
    path_list = get_irods_path_list(base_dir)
    data_sets = find_UUID_for_all_path(base_dir, path_list)

    if len(path_list) != len(data_sets):
        raise Exception("Error! number of directory is not matching with the number of data sets")

    print_json(data_sets)


def parse_args():
    if len(sys.argv) == 1:
        raise Exception("Error! base path for the archive directory is not specified")
    elif len(sys.argv) != 2:
        raise Exception("Error! wrong number of args")

    if len(sys.argv[1]) == 0:
        raise Exception("Error! path empty")

def print_help():
    print(description)
    print("Usage:")
    print("./gen_files_list.py <path-to-base-data-dir>")
    print()

def print_json(data_sets):
    # Generate & print the json file to stdout
    json_obj = {}
    json_obj["DATA_FILE_LIST"] = []
    for data in data_sets:
        data_obj = {}
        data_obj["RAW_DATA_PATH"] = data[0] + "/"
        data_obj["UUID"] = data[1]
        json_obj["DATA_FILE_LIST"].append(data_obj)
    dump_str = json.dumps(json_obj, indent=2, sort_keys=True)
    print(dump_str)

main()

