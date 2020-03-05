"""
generate_aggregate: reads all csv and json files in a ps2 collection.
    outputs a csv {filepath}_aggregated.csv with the columns [folder_name, Label, x, y, Area, Mean]

generate_fluorescence: Generates a csv with F0, FM, FV, and FV/FM.

single_process: processes a single ps2 collection using 
    generate aggregate and generate fluorescence

batch_process: uses tkinter to ask the user for a main 
    filepath containing ps2 collections and 
    uses generate_aggregate and generate_fluorescence

Jacob Long
2019-10-10    
"""

import datetime
import json
import multiprocessing
import os
import terrautils
from pathlib import Path
from terrautils.betydb import get_site_boundaries
from terrautils.spatial import scanalyzer_to_latlon   

import pandas as pd

# getting the directory that this file is running from
this_dir = os.path.dirname(os.path.abspath(__file__))

def generate_aggregate(filepath, final_output, plot_boundaries_filepath, multithresh_json, offset_x=0, offset_y=0):
    """
    reads all csv and json files in a ps2 collection.
    outputs a csv {filepath}_aggregated.csv with the columns [folder_name, Label, x, y, Area, Mean]

    parameters:
        filepath: a filepath to a top level directory containing ps2 data. 
            should have a {date}_metadata.json and {date}.csv in it
        final_output: output directory
        plot_boundaries_filepath: filepath to plot boundaries file
        multithresh_json: filepath to multithresh.json
        offset_x (float): offset for the x coordinate
        offset_y (float): offset for the y coordinate

    returns concat_df, a pandas dataframe containing aggregated data
    """

    to_concat = []

    for root, dirs, files in os.walk(filepath):

        # we dont want to check the parent directory
        if root == filepath:
            continue

        # if one of these two file does not exist, skip the folder
        try:
            json_dir = [f for f in files if f.endswith("_metadata.json")][0]
            # csv_dir = [f for f in files if f.endswith(".csv")][0]
            csv_dir = [f for f in files if f.endswith(".csv")][0]
                
        except:
            print(f"Skipping {root}. json or csv not found")
            continue

        # loading in json data to a dictionary
        with open(os.path.join(root, json_dir)) as f:
            json_dict = json.load(f)

        # extracting x and y coordinates
        try:
            x = json_dict['lemnatec_measurement_metadata']['gantry_system_variable_metadata']['position x [m]'] 
            y = json_dict['lemnatec_measurement_metadata']['gantry_system_variable_metadata']['position y [m]'] 

        except:
            print(f"invalid json file found in {root}")
            continue

        # reading the csv
        df = pd.read_csv(os.path.join(root, csv_dir))

        # creating columns for x and y from json and a column for folder name 
        # adding in offsets
        df['x'] = float(x) + offset_x
        df['y'] = float(y) + offset_y
        df['folder_name'] = os.path.basename(root)

        # sorting columns
        try:
            df = df[['folder_name', 'Label', 'x', 'y', 'Area', 'Mean', 'Min', 'Max']]
        except KeyError as e:
            print(root, e)
            continue

        # extracting multithresh values from a json file
        with open(multithresh_json) as f:
            multithresh_list = json.loads(f.read())

        # assigning each image a multithreshold value
        try:
            df['MultiThr'] = multithresh_list
        except ValueError as e:
            # the length of multithresh list does not match the amound of images taken. 
            # skip this
            print(root, e)
            continue

        to_concat.append(df)

    # making sure there are objects in the list to concat
    if to_concat:
        concat_df = pd.concat(to_concat)
    else:
        return

    # reading a file defining plot boundaries
    plot_boundaries_df = terrutils.betydb.get_site_boundaries()
    plot_boundaries_df = plot_boundaries_df[['Plot', 'X Start', 'X End', 'Y Start', 'Y End']]


    # finding all records in concat df that fall inside a plot's
    # boundaries and labeling them with that plot's id
    concat_df['x'] = scanalyzer_to_latlon(concat_df['x'])
    concat_df['y'] = scanalyzer_to_latlon(concat_df['y'])
    
    for index, row in plot_boundaries_df.iterrows():
    # Translate gantry X,Y to lat-lon
    # Create an ogr POINT type
    concat_df.loc[plot_bounds.contains(new_point), 'Plot'] = int(row['Plot'])
    
    # Calculating things
    # Area x Mean x Multithresh
    concat_df['AreaXMeanXMultiThr'] = concat_df['Area'] * concat_df['Mean'] * concat_df['MultiThr']

    # Area x Multithresh
    concat_df['AreaXMultiThr'] = concat_df['Area'] * concat_df['MultiThr']

    for label in set(concat_df['Label']):
        print(label, end='\r')

        # creating a list of true/false to tell 
        # which records to select from concat_df
        chosen_records = concat_df['Label'] == label

        # Sum(Area x Mean x Multithresh)
        concat_df.loc[chosen_records, 'Sum AreaXMeanXMultiThr'] = concat_df.loc[chosen_records, 'AreaXMeanXMultiThr'].sum()

        # Sum(Area x Multithresh)
        concat_df.loc[chosen_records, 'Sum AreaXMultiThr'] = concat_df.loc[chosen_records, 'AreaXMultiThr'].sum()
    
    concat_df['Sum_AreaXMeanXMultiThr_over_Sum_AreaXMultiThr'] = concat_df['Sum AreaXMeanXMultiThr'] / concat_df['Sum AreaXMultiThr']


    # creating an output based on given filepath
    # concat_df.to_csv("_aggregated.csv", index=False)
    #concat_df.to_csv(f"{os.path.basename(filepath)}_aggregated.csv", index=False)

    if not os.path.isdir(final_output):
        os.mkdir(final_output)
    aggregated_file = os.path.basename(filepath) + "_aggregated.csv"
    aggre_path = Path.cwd() / final_output / aggregated_file
    concat_df.to_csv(aggre_path, index=False)

    return concat_df

def generate_fluorescence(filepath, final_output, generate_file=False):
    """
    Generates a csv with F0, FM, FV, and FV/FM.
    
    parameters:
        filepath: a filepath to a top level directory containing ps2 data. 
            should have {foldername}_aggregated.csv in it
        final_output: output directory
        generate_file (bool): determines if a file is created

    returns a pandas dataframe containing fluorescence data
    """

    print("\ngenerating fluorescence for", os.path.basename(filepath))

    # finding the aggregated file
    # aggregated_filepath = os.path.join(filepath, os.path.basename(filepath) + "_aggregated.csv")
    aggregated_file = os.path.basename(filepath) + "_aggregated.csv"
    aggregated_filepath = Path.cwd() / final_output / aggregated_file

    if os.path.exists(aggregated_filepath):
        df = pd.read_csv(aggregated_filepath)
      
    else:
        # if it does not exist, return
        print(f"can't generate fluorescence file for {filepath}. missing {aggregated_filepath}")
        return

    # a list to create a dataframe out of each of the aggregated csvs
    df_data = []

    # doing calculations for each plot
    for plot in set(df['Plot']):

        # the first value from the second picture
        try:
            F0 = list(df.loc[df['Plot'] == plot, 'Sum_AreaXMeanXMultiThr_over_Sum_AreaXMultiThr'])[5]
        except IndexError as e:
            # an IndexError will be thrown if no records were found for a given plot.
            # this could happen if the plot was not listed in the Plot boundaries.xlsx
            # and the previous step has been run
            print(f'No data found in plot {plot}.', e)
            continue
        
        # extracting image number from the label string and converting it to an int for filtering
        df['img_num'] = df['Label'].str.slice(start=-8, stop=-4).astype(int)

        # maximum value of rawData images 2-46
        FM = df.loc[
            (df['Plot'] == plot) &
            ((df['img_num'] > 0) & (df['img_num'] <= 46)),
            'Sum_AreaXMeanXMultiThr_over_Sum_AreaXMultiThr'
        ].max()

        FV = FM - F0

        df_row_dict = {
            "Plot":  plot,
            "F0":    F0,
            "FM":    FM,
            "FV":    FV,
            "FV/FM": FV / FM,
        }

        df_data.append(df_row_dict)

    fluorescence_df = pd.DataFrame(df_data)
    fluorescence_df = fluorescence_df.sort_values(by="Plot")

    # output as a csv
    # fluorescence_df.to_csv(os.path.join(filepath, f'{os.path.basename(filepath)}_fluorescence.csv'), index=False)

    fluorescence_file = os.path.basename(filepath) + "_fluorescence.csv"
    flouro_path = Path.cwd() / final_output / fluorescence_file
    fluorescence_df.to_csv(flouro_path, index=False)

    return fluorescence_df

def single_process(filepath, plot_boundaries, multithresh_json, offset_x=0, offset_y=0):
    """
    processes a single ps2 collection using 
    generate aggregate and generate fluorescence

    filepath: a filepath to a collection of ps2 data
    plot_boundaries: a filepath to a "Plot boundaries.xlsx" file
    multithresh_json: a filepath to a multithresh.json file
    """

    if not filepath:
        return

    start_time = datetime.datetime.now()
    print(f"Start Time: {start_time}")

    generate_aggregate(filepath, filepath, plot_boundaries, multithresh_json, offset_x=offset_x, offset_y=offset_y)
    generate_fluorescence(filepath, filepath)
    print(f"Time Elapsed: {datetime.datetime.now() - start_time}")

def batch_process(filepath):
    """
    uses tkinter to ask the user for a main 
    filepath containing ps2 collections and 
    uses generate_aggregate and generate_fluorescence
    """

    if not filepath:
        return

    start_time = datetime.datetime.now()
    print(f"Start Time: {start_time}")

    colletions_to_process = []

    # finding collections in directory
    for root, dirs, files in os.walk(filepath):      
        for d in dirs:
            try:
                # this will throw an error if the folder is not formatted 
                # exactly like a date. If it is not, then it is not a 
                # top level collection folder
                datetime.datetime.strptime(d, '%Y-%m-%d')
                print(f"Found {d}")
                colletions_to_process.append(os.path.join(root, d))

            except:
                pass
        
        break # only entering the first level

    # process n number of collections at once where n is 
    # the amount of logical processors you have
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        # starmap lets a function multiprocess on a list of tuples 
        # and each tuple is passed as arguments
        plot_boundaries_path  = os.path.join(this_dir, "static", "Plot boundaries.xlsx")
        multithresh_json_path = os.path.join(this_dir, "static", "multithresh.json")

        p.starmap(
            generate_aggregate, 
            [(collection, plot_boundaries_path, multithresh_json_path) for collection in colletions_to_process]
        )

        p.starmap(
            generate_fluorescence, 
            [(collection,) for collection in colletions_to_process]
        )

    print(f"Time Elapsed: {datetime.datetime.now() - start_time}")    


# if __name__ == "__main__":
#     generate_aggregate(r"W:\GeneratedData\F013B2\PS2\2019-08-27", 'Plot boundaries.xlsx', 'multithresh.json')
#     generate_fluorescence(r"W:\GeneratedData\F013B2\PS2\2019-08-12", True)
