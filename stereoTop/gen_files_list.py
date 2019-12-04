#!/usr/bin/env python

import sys
import subprocess


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
    if len(sys.argv) == 1:
        raise("Error! base path for the archive directory is not specified")
    elif len(sys.argv) != 2:
        raise("Error! wrong number of args")

    if len(sys.argv[1]) == 0:
        raise("Error! path empty")

    # get the base path from cmd line arg
    base_dir = sys.argv[1]
    if(base_dir[-1] != "/"):
        base_dir += "/"
    path_list = get_irods_path_list(base_dir)
    data_sets = find_UUID_for_all_path(base_dir, path_list)

    if len(path_list) != len(data_sets):
        raise ("Error! number of directory is not matching with the number of data sets")

    # Generate & print the jx file to stdout
    print("{")
    print("\"DATA_FILE_LIST\": [")
    for data in data_sets:
        print("{")
        # PATH
        print("\"PATH\": \"%s/\"," % data[0])
        # UUID
        print("\"UUID\": \"%s\"" % data[1])
        print("},")
    print("]")
    print("}")

main()

