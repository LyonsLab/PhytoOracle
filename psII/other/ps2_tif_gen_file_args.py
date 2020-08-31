import os
import json
import  sys
import subprocess

def scan_for_subdir(base_dir):
    subdir_list = []
    # list subdirectories in base_dir
    with os.scandir(base_dir) as it:
        for entry in it:
            # entry is subdir
            # type(entry) == DirEntry
            subdir_path = entry.path
            subdir_list.append(subdir_path)
    return subdir_list

def scan_for_data_files(subdir_path):
    data_files = []
    uuid_str = ""
    # scan all subdirs for files and get uuid and data_file
    with os.scandir(subdir_path) as it:
        for entry in it:
            # entry is data files
            # _metadata.json
            if entry.name.endswith("_metadata.json"):
                uuid_str = entry.name[:-14]
            elif entry.name.endswith(".bin"):
                data_files.append(entry.name)
    return (uuid_str, data_files)

def main():
    base_dir = sys.argv[1]
    if(base_dir[-1] != "/"):
        base_dir += "/"
    if len(sys.argv) == 1:
        raise Exception("Error! base path for the archive directory is not specified")
    elif len(sys.argv) != 2:
        raise Exception("Error! wrong number of args")
    if len(sys.argv[1]) == 0:
        raise Exception("Error! path empty")
    data_set_struct = {}
    data_set_struct["METADATA_FILES"] = []
    data_set_struct["BIN_FILES"] = []
    subdir_list = scan_for_subdir(base_dir)
    for subdir_path in subdir_list:
        uuid_str, data_files = scan_for_data_files(subdir_path)

        # add to METADATA_FILES
        metadata = {}
        metadata["UUID"] = uuid_str
        metadata["PATH"] = subdir_path
        metadata["METADATA"] = subdir_path + "/" + uuid_str + "_metadata.json"
        data_set_struct["METADATA_FILES"].append(metadata)

        # add to BIN_FILES
        for bin_file in data_files:
            bin = {}
            bin["BIN"] = bin_file
            bin["BIN_NO"] = bin_file[-8:-4] # extract bin no.
            bin["UUID"] = uuid_str
            bin["PATH"] = subdir_path
            data_set_struct["BIN_FILES"].append(bin)
    # Generate formatted jx to stdout
    dump_str = json.dumps(data_set_struct, indent=4, sort_keys=True)
    print(dump_str)
if __name__ == "__main__":
    main()
