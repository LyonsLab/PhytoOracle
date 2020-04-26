#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2020-04-11
Purpose: Stitch geo-corrected images into plot 'orthos'
"""

import argparse
import os
import sys
from osgeo import gdal
import numpy as np
import pandas as pd
import glob
import csv
import subprocess
import time
import time
import posixpath

start = time.time()

# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dir',
                        metavar='str',
                        help='Directory in which plot subdirectories are located')

    return parser.parse_args()


# --------------------------------------------------
def main():
    """Run 'orthomosaic' here"""

    args = get_args()
    directories = glob.glob(args.dir + 'MAC*')
    #print(directories)
    #print(os)

    num_sub = 0
    for subdire in directories:
        start2 = time.time()
        num_sub += 1
        plot = subdire.split('/')[-1]
        good_plot = plot.split(' ')
        cwd = os.getcwd()
        plot_dir = cwd + '/' + args.dir + plot#.replace(' ', '\ ') + '/'
        full_dir = os.path.join(plot_dir)
        plot_name = '_'.join(good_plot)
        #print(plot_name)
        #print(plot_dir)
        print(f'>{num_sub:5}: {subdire}')

        if num_sub == 1:
            os.chdir(subdire)
        if num_sub >= 2:
            os.chdir('../')
            cwd = os.getcwd()
            ch_path = cwd + '/' + plot
            os.chdir(ch_path)
            print(os.getcwd())

        images = glob.glob('*.tif', recursive=True)

        img_list = []
        for i in images:
            image = i.split('/')[-1]
            img_list.append(image)
        img_str = ' '.join(img_list)
        #print(img_str)

        #img_path = os.path.join(subdir)

        cmd = f'gdalbuildvrt mosaic.vrt {img_str}'
        subprocess.call(cmd, shell=True)

        cmd2 = f'gdal_translate -co COMPRESS=LZW -co BIGTIFF=YES -outsize 100% 100% mosaic.vrt {plot_name}_ortho.tif'
        subprocess.call(cmd2, shell=True)

        end = time.time()
        total_time = end - start2
        print(f'Done - Processing time: {total_time}' + "\n")


# --------------------------------------------------
if __name__ == '__main__':
    main()
