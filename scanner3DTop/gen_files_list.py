#!/usr/bin/env python

import sys
import subprocess
import json

description = """
generate a json that contains UUID and path-to-data of the data set for all data sets.
print the json to stdout
"""

def main():
    try:
        args = parse_args()
    except:
        print_help()
        raise

    UUID_indexed_data_sets = {}
    for data_dir in args["data_dirs"]:
        # get the base path from cmd line arg
        base_dir = data_dir["base_path"]
        if(base_dir[-1] != "/"):
            base_dir += "/"
        path_list = get_sub_path_list(base_dir)
        prev_len = len(UUID_indexed_data_sets)
        UUID_indexed_data_sets = path_for_all_UUID(base_dir, path_list, UUID_indexed_data_sets, data_dir["label"], data_dir["file_ending"])
        #assert(len(UUID_indexed_data_sets) == prev_len or prev_len == 0)

        if len(path_list) != len(UUID_indexed_data_sets):
            raise Exception("Error! number of sub directory is not matching with the number of data sets")

    data_sets = []
    for UUID in UUID_indexed_data_sets:
        data_set = {}
        data_set["UUID"] = UUID
        data_set.update(UUID_indexed_data_sets[UUID])
        data_sets.append(data_set)

    print_json(data_sets)

def parse_args():
    argv_len = len(sys.argv)
    if argv_len == 1:
        raise Exception("Error! wrong number of args, need at least 3")
    if (argv_len - 1) % 3 == 1:
        raise Exception("Error! missing label-in-json and filename-ending for the last data directory")
    if (argv_len - 1) % 3 == 2:
        raise Exception("Error! missing filename-ending for the last data directory")
    args = {}
    args["data_dirs"] = []
    num_data_dir = (argv_len - 1) / 3
    num_data_dir = int(num_data_dir)
    for i in range(num_data_dir):
        data_dir = {}
        data_dir["base_path"] = sys.argv[i * 3 + 1]
        data_dir["label"] = sys.argv[i * 3 + 2]
        data_dir["file_ending"] = sys.argv[i * 3 + 3]
        args["data_dirs"].append(data_dir)
    return args

def print_help():
    print(description)
    print("Usage:")
    print("./gen_files_list.py <path-to-base-data-dir> <label-in-json> <filename-ending> ...")
    print()
    print("You need to specify base-path, label-in-json and filename-ending for each data level")
    print("e.g.")
    print("./gen_files_list.py 2018-05-15/PNG/ PNG _metadata.json 2018-05-15/PLY/ PLY __Top-heading-west_0.ply")
    print()

def get_sub_path_list(base_dir):
    path_list = []
    # list files inside base_dir
    ls_output = subprocess.check_output(["ls", base_dir]).decode("utf-8")
    path_list = ls_output.split()
    for path in path_list:
        path = path.strip()
    return path_list

def path_for_all_UUID(base_dir, path_list, data_sets, label, file_ending):
    for path in path_list:
        # get all data files in a data set via ls
        files = subprocess.check_output(["ls", base_dir + path]).decode("utf-8")[1:]
        for filename in files.split():
            index = filename.find(file_ending)
            # find the UUID from the filename of metadata file
            if index != -1:
                UUID = filename[:index]
                if UUID not in data_sets:
                    data_sets[UUID] = {}
                data_sets[UUID][label] = base_dir + path + "/"
    return data_sets

def print_json(data_sets):
    # Generate & print the json file to stdout
    json_obj = {}
    json_obj["DATA_FILE_LIST"] = data_sets
    dump_str = json.dumps(json_obj, indent=2, sort_keys=True)
    print(dump_str)

main()

