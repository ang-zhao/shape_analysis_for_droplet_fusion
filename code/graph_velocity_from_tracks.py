#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:39:56 2022

@author: angelazhao

@pre: This code is designed to use the data obtained from the ImageJ2 TrackMate plugin
@pre: To generate the files for this code, run the script 'Trackmate_particle_fusion.py' in the ImageJ2 application
@pre: Tracks data should be a csv file in the form: [Track no., POSITION_X, POSITION_Y, POSITION_T, FRAME]
@pre: Intensities data should be a csv file in the form: [Track no., SUM_INTENSITIES, POSITION_T, FRAME]

The output is 4 graphs:
    velocity_vs_time_py.png graphs the velocity vs. time and also the distance vs. time on the same graph
    velocity_vs_distance_py.png graphs the velocity vs. distance
    distance_vs_time_py.png
    intensity_vs_time_py.png graphs the total intensity of the spots over time

"""

# TODO: still can't identify which spot is smaller and which is larger. Else, assign df0 to small and df1 to large.
# TODO: did not control for cases where not the same first n timepoints are tracked for each

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter 
from statsmodels.nonparametric.kernel_regression import KernelReg

def main() :
    filename = 'Kar9_Fusion_Data/data_folders/data_demo.txt'
    with open(filename) as file :
        for line in file :
            data_name = line.rstrip()
            run_algorithm(data_name)
    
def run_algorithm(data_name) : 
    # Tracks should be in the form: [Track no., POSITION_X, POSITION_Y, POSITION_T, FRAME]
    # TODO: user must input data_folder, in which there will be an output folder containing the tracks
    data_folder = 'Kar9_Fusion_Data/' + data_name + '/'
    
    # Import tracks as dataframe
    df_tracks = pd.read_csv(data_folder + 'output/tracks.csv')
    df_tracks.dropna()

    df_tracks, is_single_track = check_single_track(df_tracks)
    
    # Import intensities as dataframe
    df_intensities = pd.read_csv(data_folder + 'output/intensities.csv')
    df_intensities.dropna()
    
    df_intensities, is_single_track = check_single_track(df_intensities)
    
    # Process intensities dataframe to split into track no. 0 and no. 1
    df0_intensities, df1_intensities = process_intensities(df_intensities, is_single_track)
    
    # If there is only one track, we cannot run most of the code. Exit with only the graph of the intensities for the one track.
    if is_single_track :
        plot_single_track(df_intensities['POSITION_T'], df_intensities['SUM_INTENSITIES'], data_folder)
        return None

    # Process tracks dataframe to return number of timepoints, tracks for spot0, and tracks for spot1 up until fusion
    max_time, max_num_points, df0, df1 = process_tracks(df_tracks)
    
    # Identify largest particle at first tracked timepoint (of the two particles), where larger is defined by the total intesity
    # TODO: the first frame for intensity0 is not the same as the first frame for intensity1. For some reason, trackmate misses many frames for total intensity
    max_intensity_on_start = None;
    assert(df0_intensities['SUM_INTENSITIES'].shape[0] > 0 and df1_intensities['SUM_INTENSITIES'].shape[0] > 0)
    if df0_intensities.iloc[0].at['SUM_INTENSITIES'] > df1_intensities.iloc[0].at['SUM_INTENSITIES']:
        max_intensity_on_start = 0
    else:
        max_intensity_on_start = 1

    # Calculate distance, velocity, and the time difference
    # velocity0_tuple = calculate_velocity(df0, distance_btw_spots)
    velocity0_np = calculate_velocity(df0)
    
    # velocity1_tuple = calculate_velocity(df1, distance_btw_spots)
    velocity1_np = calculate_velocity(df1)
    
    distance_btw_spots = find_distance(df0, df1, max_num_points, velocity0_np, velocity1_np)
    
    # Initialize arrays to plot velocities
    xd0 = distance_btw_spots['DISTANCE']
    yd0 = distance_btw_spots['VELOCITY0']
    xt0 = df0['POSITION_T']
    yt0 = velocity0_np
    
    xd1 = xd0
    yd1 = distance_btw_spots['VELOCITY1']
    xt1 = df1['POSITION_T']
    yt1 = velocity1_np
    
    # Timestamps for distance vs time graph
    dt = distance_btw_spots['POSITION_T']
    
    # Plot velocity vs distance, velocity vs time, and distance vs time
    
    plot_velocity(xd0, yd0, xt0, yt0, xd1, yd1, xt1, yt1, dt, max_intensity_on_start, max_num_points, max_time, data_folder, data_name)   
    
    # Plot intensity vs time
    plot_intensity(df0_intensities['POSITION_T'], df0_intensities['SUM_INTENSITIES'], df1_intensities['POSITION_T'], df1_intensities['SUM_INTENSITIES'], max_intensity_on_start, max_time, data_folder, data_name)
    
# Savitzky-Golay filter to smooth data
def smooth_curve(y):
    window_length = y.size
    if window_length % 2 == 0 :
        window_length = window_length - 1
    yhat = savgol_filter(y, window_length, 3)
    return yhat

# Kernel regression to smooth data
def regression_curve(x, y):
    kr = KernelReg(y, x, 'c')
    y_pred, y_std = kr.fit(x)
    return y_pred

# Find the R^2 score for the exponential fit, ie. the goodness of fit
def find_R2(y, y_fit):
    # Residual sum of squares
    ss_res = np.sum((y - y_fit) ** 2)
    
    # Total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    # R-squared
    R2 = 1 - (ss_res / ss_tot)
    
    return R2
    

def check_single_track(dataframe) :
    
    is_single_track = False
    
    track_no = dataframe.iloc[0].at['Track no.']
    
    # Split tracks
    mask = dataframe['Track no.'] == track_no
    # mask1 = dataframe['Track no.'] == 1

    df0 = dataframe[mask]
    df1 = dataframe[~mask]

    
    if df0.shape[0] - 1 < 0 :
        if df1.shape[0] - 1 < 0 :
            print('No tracks for this cell.')
        else :
            print('Only one track for this cell.')
            is_single_track = True
            return df1, is_single_track
        
    elif df1.shape[0] - 1 < 0 :
        print('Only one track for this cell.')
        is_single_track = True
        return df0, is_single_track
    
    return dataframe, is_single_track

def process_intensities(dataframe, is_single_track):
    
    if is_single_track :
        dataframe.sort_values(by = ['POSITION_T'], inplace = True)
        return dataframe, None
    
    track_no = dataframe.iloc[0].at['Track no.']
    
    # Split tracks
    mask = dataframe['Track no.'] == track_no
    # mask1 = dataframe['Track no.'] == 1

    df0 = dataframe[mask]
    df1 = dataframe[~mask]
    
    df0.sort_values(by = ['POSITION_T'], inplace = True)
    
    df1.sort_values(by = ['POSITION_T'], inplace = True)
    
    # print(df0)
    # print(df1)
    
    return df0, df1

# Find distance between the two spots at all timepoints before fusion, max_num_points
# Split tracks belonging to spot0 and spot1, stored in df0, df1
# Clip tracks to only include data up until fusion
def process_tracks(dataframe):
    track_no = dataframe.iloc[0].at['Track no.']
    
    # Split tracks
    mask = dataframe['Track no.'] == track_no
    # mask1 = dataframe['Track no.'] == 1

    df0 = dataframe[mask]
    df1 = dataframe[~mask]


    # Reorder by frame number
    df0.sort_values(by = ['POSITION_T'], inplace = True)
    df1.sort_values(by = ['POSITION_T'], inplace = True)
    
    
    # Clip tracks to only include positions before fusion
    num_points0 = df0.shape[0] - 1
    num_points1 = df1.shape[0] - 1
    max_num_points = min(num_points0, num_points1)
    
    
    if num_points0 < num_points1:
        is_shorter = 0
    else:
        is_shorter = 1
    
    # Find the amount of time for the fusion video, max_time
    if num_points0 < 0 :
        max_time = num_points1
    elif num_points1 < 0:
        max_time = num_points0
    else : 
        max_time = max(df1.iloc[num_points1].at['POSITION_T'], df0.iloc[num_points0].at['POSITION_T'])    
               
    # Find the number of timepoints before fusion, max_num_points
    counter = max_num_points
    if is_shorter == 0:
        while(True):
            if df0.iloc[num_points0].at['FRAME'] > df1.iloc[counter].at['FRAME']:
                counter += 1
                continue
            
            if df0.iloc[num_points0].at['FRAME'] < df1.iloc[counter].at['FRAME']:
                num_points1 = counter
                break
            if df0.iloc[num_points0].at['FRAME'] == df1.iloc[counter].at['FRAME']:
                num_points1 = counter
                break
            
    if is_shorter == 1:
        while(True):
            if df0.iloc[counter].at['FRAME'] < df1.iloc[num_points1].at['FRAME']:
                counter += 1
                continue
            
            if df0.iloc[counter].at['FRAME'] > df1.iloc[num_points1].at['FRAME']:
                num_points0 = counter
                break
            if df0.iloc[counter].at['FRAME'] == df1.iloc[num_points1].at['FRAME']:
                num_points0 = counter
                break
            
    
    df0_clipped = df0.head(num_points0)
    df1_clipped = df1.head(num_points1)
    
    return max_time, max_num_points, df0_clipped, df1_clipped
    
# Returns array velocity_np
# Calculates the instantaneous velocity of spot s, where dataframe are the tracks,
# from time t-1 to t, and stores it in velocity_np[t]
def calculate_velocity(dataframe) :
    
    num_rows = dataframe.shape[0]
    
    sum_of_diff_np = np.zeros(num_rows)
    
    # Calculate distance    
    for i in range(1, num_rows - 1) :
        diff = (dataframe.iloc[i].at['POSITION_X'] - dataframe.iloc[i-1].at['POSITION_X'])**2 + (dataframe.iloc[i].at['POSITION_Y'] - dataframe.iloc[i-1].at['POSITION_Y'])**2
        sum_of_diff_np[i] = abs(diff)
    
    distance_np = np.sqrt(sum_of_diff_np)
    
    # Calculate velocity and time difference
    velocity_np = np.zeros(num_rows)
    time_np = np.zeros(num_rows)

    for i in range(1, num_rows - 1) :
        time_np[i] = dataframe.iloc[i].at["POSITION_T"] - dataframe.iloc[i-1].at["POSITION_T"]
        velocity_np[i] = distance_np[i] / abs(time_np[i])
    
    return velocity_np

# Creates an array where [0]: index at timepoint t, [1]: x position at timepoint t, 
# and [2]: y position at timepoint t 
# for both df0 and df1, where t exists in both
# Define velocity at respective timepoints (ie. may not be all velociies)

def find_distance(df0, df1, max_num_points, velocity0_np, velocity1_np):
    distance_btw_spots = pd.DataFrame(np.zeros((max_num_points, 4)), columns = ["POSITION_T", "DISTANCE", "VELOCITY0", "VELOCITY1"])
    counter = 0;
    
    for i in range(0, df0.shape[0] - 1) :
        for j in range(0, df1.shape[0] - 1):
            
            if counter >= max_num_points :
                break
            
            if df1.iloc[j].at['FRAME'] < df0.iloc[i].at['FRAME']:
                continue
            
            elif df1.iloc[j].at['FRAME'] > df0.iloc[i].at['FRAME']:
                break
            
            elif df1.iloc[j].at['FRAME'] == df0.iloc[i].at['FRAME']:
                distance_btw_spots.iloc[counter].at['POSITION_T'] = df1.iloc[j].at['POSITION_T']
                diff = (df0.iloc[i].at['POSITION_X'] - df1.iloc[j].at['POSITION_X'])**2 + (df0.iloc[i].at['POSITION_Y'] - df1.iloc[j].at['POSITION_Y'])**2
                distance_btw_spots.iloc[counter].at['DISTANCE'] = np.sqrt(abs(diff))
                distance_btw_spots.iloc[counter].at['VELOCITY0'] = velocity0_np[counter]
                distance_btw_spots.iloc[counter].at['VELOCITY1'] = velocity1_np[counter] 
                counter += 1
                
            
    
    # filter(lambda v: v!=0, distance_btw_spots[:,1])    
    # print(distance_btw_spots)
            
    return distance_btw_spots.head(counter)

# Generates the plots for when there is only one track
def plot_single_track(x, y, data_folder) :
     # Label axes and title    
    ax1 = plt.subplot()
    
    ax1.plot(x, y, color = 'g')
    
    plt.title('Intensity vs. Time')
    plt.ylabel('Intensity of spots (u.a.)')
    plt.xlabel('Time (seconds)')
    
    # @pre: Set intensity y limit
    plt.ylim((0, 3500))
    # plt.xlim((0, max_time))
    
    
    plt.savefig(data_folder + "intensity_vs_time_py.png")
    plt.show()
    plt.close()
    return None

# Plots intensity vs. time for each particle
def plot_intensity(x0_intensities, y0_intensities, x1_intensities, y1_intensities, max_intensity_on_start, max_time, data_folder, data_name):
    
    
    # Unsmoothed plot
    
    ax1 = plt.subplot()

    if max_intensity_on_start == 0:
        l1, = ax1.plot(x0_intensities, y0_intensities, color = 'orange', linewidth = 1) # Highest total intensity particle on start is orange
        l2, = ax1.plot(x1_intensities, y1_intensities, color = 'g', linewidth = 1)
        plt.legend([l1, l2], ["Intensity (larger)", "Intensity (smaller)"])
    elif max_intensity_on_start == 1:
        l1, = ax1.plot(x0_intensities, y0_intensities, color = 'g', linewidth = 1) 
        l2, = ax1.plot(x1_intensities, y1_intensities, color = 'orange', linewidth = 1) # Highest total intensity particle on start is orange
        plt.legend([l2, l1], ["Intensity (larger)", "Intensity (smaller)"])
  
    # Label axes and title    
    plt.suptitle('Intensity vs. Time, SavGol')
    plt.title(data_name)
    plt.ylabel('Intensity of spots (u.a.)')
    plt.xlabel('Time (seconds)')
    
    # @pre: Set intensity y limit
    
    # plt.ylim((0, 3500))
    plt.xlim((0, max_time))
    
    
    plt.savefig(data_folder + "intensity_vs_time_nofit.png")
    plt.show()
    plt.close()
    
    # Smooth intensities using savgol smoothing
    y0_intensities_savgol = smooth_curve(y0_intensities)
    y1_intensities_savgol = smooth_curve(y1_intensities)

    # Smooth intensities using regression smoothing
    
    ax1 = plt.subplot()

    if max_intensity_on_start == 0:
        l1, = ax1.plot(x0_intensities, y0_intensities, '+', color = 'navajowhite', linewidth = 1) # Highest total intensity particle on start is orange
        l2, = ax1.plot(x1_intensities, y1_intensities, '+', color = 'darkseagreen', linewidth = 1)
        l4, = ax1.plot(x0_intensities, y0_intensities_savgol, color = 'orange', linewidth = 1)
        l5, = ax1.plot(x1_intensities, y1_intensities_savgol, color = 'g', linewidth = 1)
        plt.legend([l1, l2, l4, l5], ["Intensity (larger)", "Intensity (smaller)"])
    elif max_intensity_on_start == 1:
        l1, = ax1.plot(x0_intensities, y0_intensities, '+', color = 'darkseagreen', linewidth = 1) 
        l2, = ax1.plot(x1_intensities, y1_intensities, '+', color = 'navajowhite', linewidth = 1) # Highest total intensity particle on start is orange
        l4, = ax1.plot(x0_intensities, y0_intensities_savgol, color = 'g', linewidth = 1)
        l5, = ax1.plot(x1_intensities, y1_intensities_savgol, color = 'orange', linewidth = 1)
        plt.legend([l2, l1, l5, l4], ["Intensity (larger)", "Intensity (smaller)"])
  
    # Label axes and title    
    plt.suptitle('Intensity vs. Time, SavGol')
    plt.title(data_name)
    plt.ylabel('Intensity of spots (u.a.)')
    plt.xlabel('Time (seconds)')
    
    # @pre: Set intensity y limit
    
    # plt.ylim((0, 3500))
    plt.xlim((0, max_time))
    
    
    plt.savefig(data_folder + "intensity_vs_time_savgol.png")
    plt.show()
    plt.close()
    
    # Smooth intensities using regression smoothing
    y0_intensities_reg = regression_curve(x0_intensities, y0_intensities)
    y1_intensities_reg = regression_curve(x1_intensities, y1_intensities)

    # Smooth intensities using regression smoothing
    
    ax1 = plt.subplot()

    if max_intensity_on_start == 0:
        l1, = ax1.plot(x0_intensities, y0_intensities, '+', color = 'navajowhite', linewidth = 1) # Highest total intensity particle on start is orange
        l2, = ax1.plot(x1_intensities, y1_intensities, '+', color = 'darkseagreen', linewidth = 1)
        l4, = ax1.plot(x0_intensities, y0_intensities_reg, color = 'orange', linewidth = 1)
        l5, = ax1.plot(x1_intensities, y1_intensities_reg, color = 'g', linewidth = 1)
        plt.legend([l1, l2, l4, l5], ["Intensity (larger)", "Intensity (smaller)"])
    elif max_intensity_on_start == 1:
        l1, = ax1.plot(x0_intensities, y0_intensities, '+', color = 'darkseagreen', linewidth = 1) 
        l2, = ax1.plot(x1_intensities, y1_intensities, '+', color = 'navajowhite', linewidth = 1) # Highest total intensity particle on start is orange
        l4, = ax1.plot(x0_intensities, y0_intensities_reg, color = 'g', linewidth = 1)
        l5, = ax1.plot(x1_intensities, y1_intensities_reg, color = 'orange', linewidth = 1)
        plt.legend([l2, l1, l5, l4], ["Intensity (larger)", "Intensity (smaller)"])
  
    # Label axes and title    
    plt.suptitle('Intensity vs. Time, Regression')
    plt.title(data_name)
    plt.ylabel('Intensity of spots (u.a.)')
    plt.xlabel('Time (seconds)')
    
    # @pre: Set intensity y limit
    # plt.ylim((0, 3500))
    plt.xlim((0, max_time))
    
    
    plt.savefig(data_folder + "intensity_vs_time_reg.png")
    plt.show()
    plt.close()
    
# Plots velocity vs. time, velocity vs. time, and distance vs. time graphs 
def plot_velocity(xd0, yd0, xt0, yt0, xd1, yd1, xt1, yt1, dt, max_intensity_on_start, max_num_points, max_time, data_folder, data_name):
    
    # Smooth the y data
    
    yd0_savgol = smooth_curve(yd0)
    yd1_savgol = smooth_curve(yd1)
    
    # Perform regression on the y data
    
    yd0_reg = regression_curve(xd0, yd0)
    yd1_reg = regression_curve(xd1, yd1)
    # Plot velocity vs. distance]
    
    plot_velocity_vs_time(xd0, yd0, xt0, yt0, xd1, yd1, xt1, yt1, dt, max_intensity_on_start, max_num_points, max_time, data_folder, data_name)
    
    # plot_velocity_vs_distance(xd0, yd0, xt0, yt0, xd1, yd1, xt1, yt1, dt, max_intensity_on_start, max_num_points, max_time, data_folder, data_name)
   
    # plot_distance_vs_time(dt, xd0, max_time, data_folder, data_name)
   
    
def plot_velocity_vs_time(xd0, yd0, xt0, yt0, xd1, yd1, xt1, yt1, dt, max_intensity_on_start, max_num_points, max_time, data_folder, data_name): 
    
    ##### UNSMOOTHED #####
    # Label axes and title
    # @pre: Set velocity y axis
    plt.suptitle('Velocity vs. Time')
    plt.title(data_name)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (microns/second)')
    plt.ylim((0, 1))
    # @pre: Set time to max time tracked
    plt.xlim((0, max_time))
    
    ax1 = plt.subplot()
    
    # Plot distance on the same graph
    ax2 = ax1.twinx()
    l3, = ax2.plot(dt, xd0, color = 'black')
    
    # print("Upper on start 1: " + str(max_intensity_on_start))
    
    # Plot velocity vs. time with smoothing
    if max_intensity_on_start == 0:
        l1, = ax1.plot(xt0, yt0, color = 'navajowhite', linewidth = 1) # Highest total intensity particle on start is orange
        l2, = ax1.plot(xt1, yt1, color = 'darkseagreen', linewidth = 1)
        
        
        # Set legend
        plt.legend([l1, l2, l3], ["Velocity (larger)", "Velocity (smaller)", "Distance"])
    elif max_intensity_on_start == 1:
        l1, = ax1.plot(xt0, yt0, color = 'darkseagreen', linewidth = 1)
        l2, = ax1.plot(xt1, yt1, color = 'navajowhite', linewidth = 1) # Highest total intensity particle on start is orange
       
        
        # Set legend
        plt.legend([l2, l1, l3], ["Velocity (larger)", "Velocity (smaller)", "Distance"])

    # Label secondary axis  
    ax2.set_ylabel('Distance (microns)')
    
    # @pre: Set secondary distance y axis
    ax2.set_ylim((0, 0.8))
    
    
    plt.savefig(data_folder + "velocity_vs_time_nofit.png")
    plt.show()
    plt.close()
    
    ########################################3
    
   
    # Plot with neighbour signal smoothing, Savitzky-Golay filter
    # Smooth the y data
    yt0_savgol = smooth_curve(yt0)
    yt1_savgol = smooth_curve(yt1)
         
    # Label axes and title
    # @pre: Set velocity y axis
    plt.suptitle('Velocity vs. Time, SavGol')
    plt.title(data_name)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (microns/second)')
    plt.ylim((0, 1))
    # @pre: Set time to max time tracked
    plt.xlim((0, max_time))
    
    ax1 = plt.subplot()
    
    # Plot distance on the same graph
    ax2 = ax1.twinx()
    l3, = ax2.plot(dt, xd0, color = 'black')
    
    # print("Upper on start 1: " + str(max_intensity_on_start))
    
    # Plot velocity vs. time with smoothing
    if max_intensity_on_start == 0:
        l1, = ax1.plot(xt0, yt0, '+', color = 'navajowhite', linewidth = 1) # Highest total intensity particle on start is orange
        l2, = ax1.plot(xt1, yt1, '+', color = 'darkseagreen', linewidth = 1)
        l4, = ax1.plot(xt0, yt0_savgol, color = 'orange', linewidth = 1)
        l5, = ax1.plot(xt1, yt1_savgol, color = 'g', linewidth = 1)
        
        # Set legend
        plt.legend([l1, l2, l3, l4, l5], ["Velocity (larger)", "Velocity (smaller)", "Distance"])
    elif max_intensity_on_start == 1:
        l1, = ax1.plot(xt0, yt0, '+', color = 'darkseagreen', linewidth = 1)
        l2, = ax1.plot(xt1, yt1, '+', color = 'navajowhite', linewidth = 1) # Highest total intensity particle on start is orange
        l4, = ax1.plot(xt0, yt0_savgol, color = 'g', linewidth = 1)
        l5, = ax1.plot(xt1, yt1_savgol, color = 'orange', linewidth = 1)
        
        # Set legend
        plt.legend([l2, l1, l3, l5, l4], ["Velocity (larger)", "Velocity (smaller)", "Distance"])

    # Label secondary axis  
    ax2.set_ylabel('Distance (microns)')
    
    # @pre: Set secondary distance y axis
    ax2.set_ylim((0, 0.8))
    
    
    plt.savefig(data_folder + "velocity_vs_time_savgol.png")
    plt.show()
    plt.close()
    
   
    # Plot with regression prediction
    # Perform regression on the y data
    yt0_reg = regression_curve(xt0, yt0)
    yt1_reg = regression_curve(xt1, yt1)
    
    # Label axes and title
    # @pre: Set velocity y axis
    plt.suptitle('Velocity vs. Time, Regression')
    plt.title(data_name)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (microns/second)')
    plt.ylim((0, 1))
    # @pre: Set time to max time tracked
    plt.xlim((0, max_time))
    
    ax1 = plt.subplot()
    
    # Plot distance on the same graph
    ax2 = ax1.twinx()
    l3, = ax2.plot(dt, xd0, color = 'black')
    
    # print("Upper on start 1: " + str(max_intensity_on_start))
    
    # Plot velocity vs. time with smoothing
    if max_intensity_on_start == 0:
        l1, = ax1.plot(xt0, yt0, '+', color = 'navajowhite', linewidth = 1) # Highest total intensity particle on start is orange
        l2, = ax1.plot(xt1, yt1, '+', color = 'darkseagreen', linewidth = 1)
        l4, = ax1.plot(xt0, yt0_reg, color = 'orange', linewidth = 1.5)
        l5, = ax1.plot(xt1, yt1_reg, color = 'g', linewidth = 1.5)
        
        # Set legend
        plt.legend([l1, l2, l3], ["Velocity (larger)", "Velocity (smaller)", "Distance"])
    elif max_intensity_on_start == 1:
        l1, = ax1.plot(xt0, yt0, '+', color = 'darkseagreen', linewidth = 1)
        l2, = ax1.plot(xt1, yt1, '+', color = 'navajowhite', linewidth = 1) # Highest total intensity particle on start is orange
        l4, = ax1.plot(xt0, yt0_reg, color = 'g', linewidth = 1.5)
        l5, = ax1.plot(xt1, yt1_reg, color = 'orange', linewidth = 1.5)
        
        
        # Set legend
        plt.legend([l2, l1, l3, l4, l5], ["Velocity (larger)", "Velocity (smaller)", "Distance"])

    # Label secondary axis  
    ax2.set_ylabel('Distance (microns)')
    
    # @pre: Set secondary distance y axis
    ax2.set_ylim((0, 0.8))
    
    
    plt.savefig(data_folder + "velocity_vs_time_reg.png")
    plt.show()
    plt.close()

def plot_velocity_vs_distance(xd0, yd0, xt0, yt0, xd1, yd1, xt1, yt1, dt, max_intensity_on_start, max_num_points, max_time, data_folder, data_name):
     # Plot velocity vs. distance between spots
    ax1 = plt.subplot()
    # print("Upper on start 2: " + str(max_intensity_on_start))

    if max_intensity_on_start == 0:
        l1, = ax1.plot(xd0, yd0, color = 'orange', linewidth = 1) # Highest total intensity particle on start is orange
        l2, = ax1.plot(xd1, yd1, color = 'g', linewidth = 1)
        plt.legend([l1, l2], ["Velocity (larger)", "Velocity (smaller)"])
    elif max_intensity_on_start == 1:
        l1, = ax1.plot(xd0, yd0, color = 'g', linewidth = 1)
        l2, = ax1.plot(xd1, yd1, color = 'orange', linewidth = 1) # Highest total intensity particle on start is orange
        plt.legend([l2, l1], ["Velocity (larger)", "Velocity (smaller)"])
    

    # Label axes and title    
    plt.suptitle('Velocity vs. Distance')
    plt.title(data_name)
    plt.xlabel('Distance (microns)')
    plt.ylabel('Velocity (microns/second)')
    
    
    # @pre: Set velocity y limit
    plt.ylim((0, 1))
    
    plt.savefig(data_folder + "velocity_vs_distance_py.png")
    plt.show()
    plt.close()
    
def plot_distance_vs_time(dt, xd0, max_time, data_folder, data_name):
     # Plot distance vs. time between spots
    plt.plot(dt, xd0, color = 'black')
  
    # Label axes and title    
    plt.suptitle('Distance vs. Time')
    plt.title(data_name)
    plt.ylabel('Distance between spots (microns)')
    plt.xlabel('Time (seconds)')
    
    # @pre: Set distance y limit
    plt.ylim((0, 0.8))
    plt.xlim((0, max_time))
    
    plt.savefig(data_folder + "distance_vs_time_py.png")
    plt.show()
    plt.close()
    
    
   
if __name__ == "__main__" : 
    main()



