#!/usr/bin/env python3
"""
Author : eg
Date   : 2021-04-29
Purpose: Rock the Casbah
"""

import argparse
import os
import sys
import numpy as np
import subprocess
import re
from datetime import datetime

# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('positional',
    #                     metavar='str',
    #                     help='A positional argument')

    parser.add_argument('-sea',
                        '--season',
                        help='Season to query.\
                            Allowed values are: 10, 11, 12.',
                        metavar='',
                        type=str,
                        required=True,
                        choices=['10', '11', '12'])

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

def get_paths(season, sensor):

    season_dict = {'10': '/iplant/home/shared/phytooracle/season_10_lettuce_yr_2020/',
                    '11': '/iplant/home/shared/phytooracle/season_11_sorghum_yr_2020/',
                    '12': '/iplant/home/shared/phytooracle/season_12_sorghum_soybean_sunflower_tepary_yr_2021/'}

    season_path = season_dict[season]
    sensor_path = sensor

    return season_path, sensor_path

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

    process_list = np.setdiff1d(level_0_dates, level_1_dates)
    user_list = level_0_list
    matchers = process_list
    matching = [os.path.splitext(os.path.basename(s))[0].replace(f'{args.sensor}-', '') for s in user_list if any(xs in s for xs in matchers)]
    
    for item in matching:
        cmd = f'./run.sh {item}'
        subprocess.call(cmd, shell=True)
        #print(f'Season {args.season} {sensor_path} processing complete.')


# --------------------------------------------------
if __name__ == '__main__':
    main()
