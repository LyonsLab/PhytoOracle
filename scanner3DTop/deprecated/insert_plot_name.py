#!/usr/bin/env python3

import sys
import json

#
# Usage:
# insert_plot_name.py path/cleaned_metadata.json path/edited_cleaned_metadata.json plot_name
#
def main():
    if len(sys.argv) != 4:
        print_help()
        raise Exception("Wrong number of arg, need 3")
    cleaned_metadata_filename = sys.argv[1]
    edited_cleaned_metadata_filename = sys.argv[2]
    plot_name = sys.argv[3]
    with open(cleaned_metadata_filename, "r", encoding="utf-8") as infile:
        metadata_json = json.load(infile)
        # Insert a entry under content
        metadata_json["content"]["plot_name"] = plot_name
        with open(edited_cleaned_metadata_filename, "w", encoding="utf-8") as outfile:
            json.dump(metadata_json, outfile)

def print_help():
    print("Usage:")
    print("python insert_plot_name.py path/cleaned_metadata.json path/edited_cleaned_metadata.json plot_name")
    print("")


main()
