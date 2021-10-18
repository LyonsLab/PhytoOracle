#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2021-04-29
Purpose: Pipeline automation
"""

import argparse
import os
import sys
import numpy as np
import subprocess
import re
import pandas as pd
from datetime import datetime

# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Pipeline automation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-sea',
                        '--season',
                        help='Season to query.\
                            Allowed values are: 10, 11, 12.',
                        metavar='',
                        type=str,
                        required=True,
                        choices=['10', '11', '12'])

    parser.add_argument('-o',
                        '--ortho',
                        help='Download geoTIFF orthomosaic. For 3D pipeline only.',
                        action='store_true')

    parser.add_argument('-r',
                        '--reverse',
                        help='Reverse the order of processing list.',
                        action='store_true')

    parser.add_argument('-c',
                        '--crop',
                        help='Crop to process.',
                        choices=['sunflower', 'bean', 'sorghum'])

    parser.add_argument('-sen',
                        '--sensor',
                        help='Sensor to query.\
                            Allowed values are: EnvironmentLogger, flirIrCamera, ps2Top, scanner3DTop, stereoTop, VNIR, SWIR, PikaVNIR.',
                        metavar='',
                        type=str,
                        required=True,
                        choices=['EnvironmentLogger', 'flirIrCamera', 'ps2Top', 'scanner3DTop', 'stereoTop', 'VNIR', 'SWIR', 'PikaVNIR'])

    return parser.parse_args()


# --------------------------------------------------
def return_date_list(level_0_list):
    date_list = []
    for item in level_0_list:
        match = re.search(r'\d{4}-\d{2}-\d{2}', item)
        if match:
            date = str(datetime.strptime(match.group(), '%Y-%m-%d').date())
            date_list.append(date)
            
    return date_list


# --------------------------------------------------
def get_paths(season, sensor):

    season_dict = {'10': '/iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/',
                    '11': '/iplant/home/shared/phytooracle/season_11_sorghum_yr_2020/',
                    '12': '/iplant/home/shared/phytooracle/season_12_sorghum_soybean_sunflower_tepary_yr_2021/'}

    season_path = season_dict[season]
    sensor_path = sensor

    return season_path, sensor_path


# --------------------------------------------------
def get_rgb_ortho(season_path, sensor_path, date):

    date_string = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', date).group(), '%Y-%m-%d').date()

    date_list = [str(os.path.splitext(os.path.basename(line.strip().replace('C- ', '')))[0]) for line in os.popen(f'ils {os.path.join(season_path, "level_1", sensor_path)}').readlines()][1:]
    cyverse_paths = [line.strip().replace('C- ', '') for line in os.popen(f'ils {os.path.join(season_path, "level_1", sensor_path)}').readlines()][1:]
    
    date_list_cleaned = []
    cyverse_paths_cleaned = []

    for date, path in zip(date_list, cyverse_paths):
        try:
            dt = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', date).group(), '%Y-%m-%d').date()
            dt = pd.to_datetime(dt)
            date_list_cleaned.append(dt.date())
            cyverse_paths_cleaned.append(path)
        except:
            pass

    df = pd.DataFrame()

    df['datetime'] = date_list_cleaned
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['cyverse_path'] = cyverse_paths_cleaned
    df = df.set_index('datetime')
    
    matching_date = df.iloc[df.index.get_loc(pd.to_datetime(date_string), method='nearest')]
    query_path = matching_date['cyverse_path']

    ortho = [item.strip() for item in os.popen(f'ils {query_path}').readlines()[1:] if '.tif' in item]

    if ortho:
        orthomosaic_path = os.path.join(query_path, ortho[0])
    
    return orthomosaic_path


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()
    season_path, sensor_path = get_paths(args.season, args.sensor)

    level_0 = os.path.join(season_path, 'level_0', sensor_path)
    level_1 = os.path.join(season_path, 'level_1', sensor_path)

    level_0_list, level_1_list = [os.path.splitext(os.path.basename(item))[0].lstrip() for item in [line.rstrip() for line in os.popen(f'ils {level_0}').readlines()][1:]] \
                                , [os.path.splitext(os.path.basename(item))[0].lstrip() for item in [line.rstrip() for line in os.popen(f'ils {level_1}').readlines()][1:]]
    
    level_0_dates, level_1_dates = return_date_list(level_0_list) \
                                , return_date_list(level_1_list)
    
    process_list = list(np.setdiff1d(level_0_dates, level_1_dates))

    matching = [os.path.splitext(os.path.basename(s))[0].replace(f'{args.sensor}-', '') for s in level_0_list if any(xs in s for xs in process_list)]
    
    if args.reverse:
        matching.reverse()
    
    for item in matching:

        for date in level_0_list:

            if item in date and 'none' not in date: 
                scan = item
                #scan = os.path.splitext(os.path.basename(date))[0]
                if args.ortho:
                    orthomosaic_path = get_rgb_ortho(season_path, 'stereoTop', date)

                    if not os.path.isfile(os.path.basename(orthomosaic_path)):
                        cmd1 = f'iget -N 0 -PVT {orthomosaic_path}'
                        subprocess.call(cmd1, shell=True)

                if args.crop:
                    if args.crop in scan:
                        cmd2 = f'./run.sh {scan}'
                        subprocess.call(cmd2, shell=True)
                        print(f'INFO: {scan} processing complete.')
                else:
                    cmd2 = f'./run.sh {scan}'
                    subprocess.call(cmd2, shell=True)
                    print(f'INFO: {scan} processing complete.')


# --------------------------------------------------
if __name__ == '__main__':
    main()
