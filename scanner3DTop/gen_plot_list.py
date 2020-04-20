#!/usr/bin/env python

import sys
import os
import json

description = """
generate a json that contains UUID and path-to-data of the data set for all data sets.
print the json to stdout
"""

def get_path_list(base_dir):
    path_list = []
    with os.scandir(base_dir) as it:
        for entry in it:
            if entry.is_dir():
                path = entry.path
                if path[-1] != "/":
                    path += "/"
                path_list.append((path, entry.name))
    return path_list

def find_files_for_all_plot(base_dir, path_list, metadata_dir, label, file_ending):
    plot_list = []
    for path, plot_name in path_list:
        plot = {}
        plot["PLOT_PATH"] = path
        plot["PLOT_NAME"] = plot_name
        # scan plot dir for data files and metadata file
        with os.scandir(path) as it:
            for entry in it:
                # data files
                if entry.name.endswith(file_ending):
                    if label not in plot:
                        plot[label] = []
                    plot[label].append(entry.name)
                    # metadata file assoicate with the first data file
                    if "METADATA" not in plot:
                        UUID = entry.name[:36]
                        plot["METADATA"] = find_metadata_file(metadata_dir, UUID)
        plot_list.append(plot)
    return plot_list

def find_metadata_file(metadata_dir, UUID):
    with os.scandir(metadata_dir) as it:
        for entry in it:
            if entry.name.endswith(".json") and entry.name.startswith(UUID):
                return entry.path
    raise Exception("No metadata file found with UUID {}".format(UUID))

def main():
    try:
        parse_args()
    except Exception as e:
        print(e)
        print_help()
        raise

    # get the base path from cmd line arg
    base_dir = sys.argv[1]
    if(base_dir[-1] != "/"):
        base_dir += "/"

    metadata_dir = sys.argv[2]

    label = sys.argv[3]

    # get data file ending
    file_ending = sys.argv[4]

    path_list = get_path_list(base_dir)
    plot_list = find_files_for_all_plot(base_dir, path_list, metadata_dir, label, file_ending)

    if len(path_list) != len(plot_list):
        raise Exception("Error! number of directory is not matching with the number of data sets")

    print_json(plot_list)


def parse_args():
    if len(sys.argv) == 1:
        raise Exception("Error! base path for the archive directory is not specified")
    if len(sys.argv) == 2:
        raise Exception("Error! missing path for metadata directory")
    if len(sys.argv) == 3:
        raise Exception("Error! missing label for the data file")
    if len(sys.argv) == 4:
        raise Exception("Error! missing file ending for data files")
    elif len(sys.argv) != 5:
        raise Exception("Error! wrong number of args")

    if len(sys.argv[1]) == 0:
        raise Exception("Error! path empty")
    if sys.argv[2] == "METADATA":
        raise Exception("Error! label cannot be \"METADATA\"")

def print_help():
    print(description)
    print("Usage:")
    print("./gen_files_list.py <path-to-base-dir> <metadata-dir> <data-file-label> <data-file-ending>")
    print()

def print_json(plot_list):
    # Generate & print the json file to stdout
    json_obj = {}
    json_obj["PLOT_LIST"] = plot_list
    dump_str = json.dumps(json_obj, indent=2, sort_keys=True)
    print(dump_str)

main()

