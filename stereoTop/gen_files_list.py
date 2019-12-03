#!/usr/bin/env python

import sys
import subprocess


def get_irods_path_list():
    path_list = []
    # get ils output of base path via stdin
    if len(sys.argv) == 1:
        stdin_input = []
        # get stdin into a list
        for line in sys.stdin:
            stdin_input.append(line)
        # parse out the path
        for line in stdin_input[1:]:# ignore the 1st line
            assert(len(line.strip().split()) == 2)
            path = line.strip().split()[1]
            path_list.append(path)
    # get the base path from cmd line arg
    else:
        ils_output = subprocess.check_output(["ils", sys.argv[1]]).decode("utf-8").split()[1:]
        path_list = ils_output.split()[1:]
        for path in path_list:
            path = path.strip().split()[1]
    return path_list

def find_UUID_for_all_path(path_list):
    data_sets = []
    for path in path_list:
        # get all data files via ils
        files = subprocess.check_output(["ils", path]).decode("utf-8")[1:]
        for filename in files.split():
            index = filename.find("_metadata.json")
            # find the UUID from the filename of metadata file
            if index != -1:
                UUID = filename[:index]
                data_sets.append((path, UUID))
    return data_sets

def main():
    path_list = get_irods_path_list()
    data_sets = find_UUID_for_all_path(path_list)

    assert(len(path_list) == len(data_sets))
    print("{")
    print("\"NUM_SET\": %d," % (len(data_sets)))
    print("\"IRODS_DIR_PATH_LIST\": [")
    for data in data_sets:
        print("\"%s\"," % (data[0]))
    print("],")
    print("\"UUID_LIST\": [")
    for data in data_sets:
        print("\"%s\"," % (data[1]))
    print("]")
    print("}")

main()

