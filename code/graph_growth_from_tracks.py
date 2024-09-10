#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 12:07:42 2022

@author: angelazhao
@author: reminder that in contour csvs, ellipse angle counter-clockwise is positive)
"""

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math

def main() :
    # The filename should be a text file containing a different data folder on each line
    # Inside each data folder should be an output folder with tracks from red/SPB and green/Kar9 channels generated from ImageJ2 Trackmate
    filename = 'Kar9_Fusion_Data/data_folders/data_growing.txt'
    
    # Iterate through each data folder
    with open(filename) as file :
        for line in file :
            data_name = line.rstrip()
            run_algorithm(data_name)
            
def run_algorithm(data_name) :
    data_folder = 'Kar9_Fusion_Data/Growing/' + data_name + '/'
    df_tracks_SPB = pd.read_csv(data_folder + 'output/contour_SPB.csv')
    df_tracks_Kar9 = pd.read_csv(data_folder + 'output/contour_Kar9.csv')
    
    # Theta is the original angle of the axis 
    theta, df_tracks_SPB, df_tracks_Kar9 = process_tracks(df_tracks_SPB, df_tracks_Kar9)
    print(theta)
    
def process_tracks(df_tracks_SPB, df_tracks_Kar9) :
    df_tracks_SPB.dropna()
    df_tracks_Kar9.dropna()
    
    # Keep longest track (assumes two tracks or fewer, no equal length) for SPB
    track_no = df_tracks_SPB.iloc[0].at['TRACK_ID']
    
    mask = df_tracks_SPB['TRACK_ID'] == track_no
    df0_SPB = df_tracks_SPB[mask]
    df1_SPB = df_tracks_SPB[~mask]
    
    if (df0_SPB.shape[0] > df1_SPB.shape[0]) :
        df_tracks_SPB = df0_SPB
    else :
        df_tracks_SPB = df1_SPB
    
    # Keep longest track (assumes two tracks or fewer, no equal length) for Kar9
    track_no = df_tracks_Kar9.iloc[0].at['TRACK_ID']
    
    mask = df_tracks_Kar9['TRACK_ID'] == track_no
    df0_Kar9 = df_tracks_Kar9[mask]
    df1_Kar9 = df_tracks_Kar9[~mask]
    
    if (df0_Kar9.shape[0] > df1_Kar9.shape[0]) :
        df_tracks_Kar9 = df0_Kar9
    else :
        df_tracks_Kar9 = df1_Kar9
    
    # Sort tracks by frame number
    df_tracks_SPB.sort_values(by = ['FRAME'], inplace = True)
    df_tracks_Kar9.sort_values(by = ['FRAME'], inplace = True)
    
    num_points_SPB = df_tracks_SPB.shape[0]
    num_points_Kar9 = df_tracks_Kar9.shape[0]

    starting_frame = df_tracks_SPB.iloc[0].at['FRAME']
    
    ## Not sure what is wrong; this is infinite loop
    # print(num_points_SPB)
    # print(num_points_Kar9)
    # # Find first starting frame that exists in both SPB and Kar9 tracks
    # for i in range(0, num_points_SPB - 1) :
    #     for j in range(0, num_points_Kar9 - 1) :
    #         if df_tracks_SPB.iloc[i].at['FRAME'] < df_tracks_Kar9.iloc[j].at['FRAME'] :
    #             break
            
    #         elif df_tracks_SPB.iloc[i].at['FRAME'] > df_tracks_Kar9.iloc[j].at['FRAME'] :
    #             continue
            
    #         else :
    #             starting_frame = tuple([i, j, df_tracks_SPB.iloc[i].at['FRAME']])
    
    
    # print('Done loops')
    # Find starting x and y coordinates for SPB and Kar9
    SPB_starting_pos = tuple([df_tracks_SPB.iloc[0].at['POSITION_X'], df_tracks_SPB.iloc[0].at['POSITION_Y']])
    Kar9_starting_pos = tuple([df_tracks_Kar9.iloc[0].at['POSITION_X'], df_tracks_Kar9.iloc[0].at['POSITION_Y']])
    
    # Draw the axis from SPB to Kar9 as a line with slope m = (y-y0)/(x-x0) = tan(theta)
    
    m = (Kar9_starting_pos[1] - SPB_starting_pos[1])/(Kar9_starting_pos[0] - SPB_starting_pos[0])
    
    theta = -1 * math.atan(m)
    
    return theta, df_tracks_SPB, df_tracks_Kar9
    
if __name__ == "__main__" :
    main()