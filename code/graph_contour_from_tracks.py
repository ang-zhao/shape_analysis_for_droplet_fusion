#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:12:30 2022

@author: angelazhao

ALT+SHIFT+W to close two window setting



: This code is designed to use the data obtained from the ImageJ2 TrackMate plugin using the Stardist detector
: Tracks data should be a csv file in the form: ['Track no.', 'ELLIPSE_ASPECTRATIO', 'AREA', 'RADIUS', 'POSITION_T','FRAME']
@pre: Set frames_per_phase if aspect ratio vs time for every phase plot is desired


The output is one graph:
    aspect_ratio_vs_time_py.png graphs the aspect ratio vs. time of the spot that is tracked for the longest time

"""

import pandas as pd
import numpy as np 
import csv
import matplotlib.pyplot as plt
import math

from pylab import rc
from scipy import stats, optimize
from scipy.signal import savgol_filter 


def main():
    
    # @pre: Set frames_per_phase if aspect ratio vs time for every phase plot is desired
    frames_per_phase = int(9)
    
    # filename = input("Filename (full path) containing folder names for all data: ")
    
    # The filename should be a text file containing a different data folder on each line
    # Inside each data folder should be an output folder with contour.csv generated from ImageJ2 Trackmate_contour_features.py script
    # @pre: set full path to a text file containing the names of all data folders
    filename = 'Kar9_Fusion_Data/data_folders/data.txt'

    # Output file of curve_fitting parameters
    # @pre: set full path to csv file containing all the parameters of the analysis (e.g. fusion time)
    output_csv = 'Kar9_Fusion_Data/parameters.csv'

    # sort parameters
    # @pre: type "Y" into console to sort the output_csv by decreasing order of R^2
    to_sort = input("Sort parameters only? Y for yes and N for no.")
    if to_sort == "Y" :
        sort_parameters(output_csv)
        return None
    
    
    # Create or empty output file
    # output_file = 'parameters.txt'
    # open(output_file, 'w').close()
    
    
    # Create or empty output file
    open(output_csv, 'w').close()
    
    header = ['DATA_FOLDER', 'A', 'K', 'C', 'FUSION_TIME', 'R2']
    write_to_csv(output_csv, header)
    
    # Iterate through each data folder
    with open(filename) as file :
        for line in file :
            data_name = line.rstrip()
            run_algorithm(data_name, output_csv, frames_per_phase)
            
def run_algorithm(data_name, output_file, frames_per_phase) : 
    # @pre: all data folders should be in a parent folder called 'Kar9_Fusion_Data'
    data_folder = 'Kar9_Fusion_Data/' + data_name + '/'
    df_tracks = pd.read_csv(data_folder + 'output/contour.csv')
    df_tracks.dropna()

    # print("df_tracks: ")
    # print(df_tracks)
    # print(df_tracks)
    
    max_time, df = process_tracks(df_tracks)
    df.dropna()
    
    x = df['POSITION_T']
    y = df['ELLIPSE_ASPECTRATIO']
    print("SHAPE: " + str(df.shape[0]))
    
    
    # Write parameters into file
    A, K, C, y_fit = plot_aspect_ratio(df, x, y, max_time, data_folder, data_name)
    
    print("A: " + str(A))
    print("K: " + str(K))
    print("C: " + str(C))

    
    # Calculate the goodness of fit, R-squared score
    R2 = find_R2(y, y_fit)
    
    # parameters = data_name + '\nA: ' + str(A) + '\nB: ' + str(K) + '\nC: ' + str(C) + '\n\n'
    # write_to_file(output_file, parameters)
    
    # Time of fusion, defined as T = -1/K
    fusion_time = 1/K
    
    data = [data_name, A, K, C, fusion_time, R2]
    write_to_csv(output_file, data)
    
    # Plot aspect ratio vs time; one line for every phase
    tracks_array, time_array, num_phases = process_per_phase(df, frames_per_phase)
    plot_per_phase(num_phases, tracks_array, time_array, max_time, data_folder, data_name)
    
    # Plot intensity
    y_intensity = df['TOTAL_INTENSITY_CH1']
    plot_single_intensity(x, y_intensity, data_folder, data_name)
    
    # Plot aspect ratio vs velocity
    # velocity = calculate_velocity(df)
    # plot_aspect_ratio_vs_velocity(velocity, y, max_time, data_folder, data_name)

def plot_aspect_ratio_vs_velocity(x, y, max_time, data_folder, data_name):
    # Label title and axes
    plt.suptitle('Aspect Ratio vs. Velocity')
    plt.title(data_name)
    plt.ylabel('Ellipse aspect ratio')
    plt.xlabel('Velocity (microns/second)')

    # Set limits
    # @pre: Set aspect ratio y limit
    plt.ylim((0.75, 2.75))

    plt.plot(x, y, '+')
    
    # Savgol fit
    yhat = smooth_curve(y)
    # plt.plot(x, yhat, linewidth = 1, color = 'navajowhite')
    
    # Linear regression
    # Plot linear regression fit
    trend = linear_regression(x, y)
    trendpoly = np.poly1d(trend)
    y_fit = trendpoly(x)
    plt.plot(x, y_fit, linewidth = 1, color = 'darkseagreen')
    R2 = find_R2(y, y_fit)
    print('R2 = ' + str(R2))
    
    
    plt.savefig(data_folder + 'aspect_ratio_vs_velocity_points_stardist.png')
    plt.show()
    plt.clf()
    
# Find the R^2 score for the exponential fit, ie. the goodness of fit
def find_R2(y, y_fit):
    # Residual sum of squares
    ss_res = np.sum((y - y_fit) ** 2)
    
    # Total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    # R-squared
    R2 = 1 - (ss_res / ss_tot)
    
    print(R2)
    return R2
    
# Savitzky-Golay filter to smooth data
def smooth_curve(y):
    window_length = y.size
    if window_length % 2 == 0 :
        window_length = window_length - 1
    yhat = savgol_filter(y, window_length, 3)
    return yhat
    
# Linear fit (regression)
def linear_regression(x, y) :
    model = np.polyfit(x, y, 1)
    return model

# Graphs ellipse aspect ratio vs. time
def plot_aspect_ratio(dataframe, x, y, max_time, data_folder, data_name):
    
    # Label title and axes
    plt.suptitle('Aspect Ratio vs. Time')
    plt.title(data_name)
    plt.ylabel('Ellipse aspect ratio')
    plt.xlabel('Time (seconds)')
    
    # Set limits
    # @pre: Set aspect ratio y limit
    plt.ylim((0.75, 2.75))
    plt.xlim((0, max_time))
    
    plt.plot(x, y)
    
    # Fit data to an exponential, y = Ae^(Bx) => log(y) = log(A) + Bx => fit log(y) against x
    # Because polyfit favours small values (bias), we add a weight proportional to y
    # Only fit data after fusion start
    # trend = np.polyfit(x_clipped, np.log(y_clipped), 1, w = np.sqrt(x))
    
    start, end, max_y = find_fusion_time(dataframe, x, y, max_time)

    x_clipped, y_clipped = clip_fusion(x, y, start, end)
    # print("y_clipped: ")
    # print(y_clipped)
    # trend = np.polyfit(np.log(x_clipped), y_clipped, 1)
    # # trend = np.polyfit(x_clipped, np.log(y_clipped), 1)
    # print(trend)
    
    # trendline = np.poly1d(trend)
    # plt.plot(x_clipped, trendline(y_clipped))
    
    A, K, C = exp_fit(x_clipped, y_clipped)
    # fit_y = model_func(x_clipped, A, K, max_y)
    
    fit_y = model_func(x_clipped, A, K, C)
    # A, K = exp_fit(x_clipped, y_clipped)
    # fit_y = (model_fusion(x_clipped, A, K))
    # print('A: ' + str(A))
    # print('K: ' + str(K))
    # print('C: ' + str(C))
    
    plt.plot(x_clipped, fit_y)
    
    plt.savefig(data_folder + 'aspect_ratio_vs_time.png')
    plt.savefig(data_folder + 'aspect_ratio_vs_time.eps')
    plt.show()
    plt.clf()
    
    return A, K, C, fit_y
    
# Find the R^2 score for the exponential fit, ie. the goodness of fit
def find_R2(y, y_fit):
    # Residual sum of squares
    ss_res = np.sum((y - y_fit) ** 2)
    
    # Total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    
    # R-squared
    R2 = 1 - (ss_res / ss_tot)
    
    return R2
    

# Full model for exponential fit
def model_func(t, A, K, C):
    return A * np.exp(-K * t) + C

# Partial model for exponential fit
def partial_func(t, A, K):
    return A * np.exp(-K * t)

# Where t is x-axis (time), A is A_knot, K is tau
def partial_model_fusion(t, A, K):
    return 1 + (A-1)**(-K * t)

def full_model_fusion(t, A, K, C):
    return (A-1) ** (-K * t) + C

# Fit data to exponential
def exp_fit(x, y):
    try : 
        opt_parms, parm_cov = optimize.curve_fit(partial_func, x, y, maxfev = 10000)
        A, K = opt_parms
        opt_parms, parm_cov = optimize.curve_fit(model_func, x, y, p0 = (A, K, 0), maxfev = 10000)
        A, K, C = opt_parms
    except :
        A = -1
        K = -1
        C = -1
    
    # opt_parms, parm_cov = optimize.curve_fit(partial_model_fusion, x, y, maxfev = 10000)
    # A, K = opt_parms
    
    # opt_parms, parm_cov = optimize.curve_fit(full_model_fusion, x, y, p0 = (2, K, 1), maxfev = 10000)
    # A, K, C = opt_parms
    # C = 1
    return A, K, C

# Clip x and y so that it only includes points after fusion start, assuming fusion start is when aspect ratio is maximized
def find_fusion_time(dataframe, x, y, max_time):
    pd.set_option('display.max_rows', 300)
    
    max_value = max(y)
    # print("maxy : " + str(max_value))
    # max_index = dataframe.loc[dataframe['ELLIPSE_ASPECTRATIO'] == max_value].index[0]
    # df_max_index = int(dataframe.iloc[max_index].at['FRAME'])
    # df_clipped = dataframe.loc[dataframe['FRAME'] >= df_max_index]
    
    # y_clipped = df_clipped['ELLIPSE_ASPECTRATIO']
    # x_clipped = df_clipped['POSITION_T']
    y_np = y.to_numpy(y)
    fusion_start_index = np.argmax(y_np)
    
    # Fusion is approximately 80 frames
    if y.shape[0] > fusion_start_index + 80 : 
        temp_end_index = fusion_start_index + 80
    else :
        temp_end_index = y.shape[0] - 1
    # x_np = x.to_numpy(x)
    # start_time = x[fusion_start_index]
    # start_row = dataframe.loc[dataframe['POSITION_T'] == start_time]
    # start_frame = dataframe.iloc[start_row].at['FRAME']
    # end_frame = start_frame + 80
    # end_row = dataframe.loc[dataframe['FRAME'] == end_frame]
    y_fusion = y.iloc[fusion_start_index:temp_end_index]
    fusion_end_index = np.argmin(y_fusion) + fusion_start_index
    # fusion_end_index = temp_end_index
    
    # print('max_value = ' + str(max_value))
    # print('max_index = ' + str(max_index))
    # #print(y)
    # # print(y_np[max_index])
    # # print("dataframe: " + str(dataframe.iloc[max_index].at['ELLIPSE_ASPECTRATIO']))
    # # x_clipped = x.iloc[max_index:]
    # # y_clipped = y.iloc[max_index:]
    
    # x_clipped = x.iloc[max_index:]
    # y_clipped = y.iloc[max_index:]
    # print(y.shape[0])
    # print(y_clipped.shape[0])
    # print('y shape: ' + str(y.shape[0]))
    # print('start: ' + str(fusion_start_index))
    # print('end: ' + str(fusion_end_index))
    
    return fusion_start_index, fusion_end_index, max_value

def clip_fusion(x, y, start, end):
    # to_clip = input('Does the fusion need to be clipped? Enter Y for yes, N for no.')
    # if to_clip == 'Y' :
    #      x_clipped = x.iloc[start:end]
    #      y_clipped = y.iloc[start:end]
         
    # else :
    #      x_clipped = x.iloc[start:]
    #      y_clipped = y.iloc[start:]
    
    x_clipped = x.iloc[start:]
    y_clipped = y.iloc[start:]
         
   
    return x_clipped, y_clipped

# For plotting aspect ratio vs velocity
def calculate_velocity(dataframe) :
    
    num_rows = dataframe.shape[0]
    
    sum_of_diff_np = np.zeros(num_rows)
    # print(num_rows)
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
        # print(i)
        # print(time_np[i])
        velocity_np[i] = distance_np[i] / abs(time_np[i])
    
    max_velocity = max(velocity_np)
    
    # print(max_velocity)
    
    return velocity_np


def process_tracks(dataframe):
    
    ## TODO: Change ['Track no.'] into ['TRACK_ID']
    
    track_no = dataframe.iloc[0].at['TRACK_ID']
    
    # Split tracks
    mask = dataframe['TRACK_ID'] == track_no
    # mask1 = dataframe['TRACK_ID'] == 1

    df0 = dataframe[mask]
    df1 = dataframe[~mask]
    
    # print('df0: ' + str(max(df0['ELLIPSE_ASPECTRATIO'])))


    # Reorder by frame number
    df0.sort_values(by = ['POSITION_T'], inplace = True)
    df1.sort_values(by = ['POSITION_T'], inplace = True)
    
    
    # Clip tracks to only include positions before fusion
    num_points0 = df0.shape[0] - 1
    num_points1 = df1.shape[0] - 1
    
    # print("num0 = " + str(num_points0))
    # print("num1 = " + str(num_points1))

    
    # Find the length of the longest track
    if num_points0 < 0 :
        assert num_points1 >= 0
        max_time = df1.iloc[num_points1].at['POSITION_T']
    elif num_points1 < 0:
        max_time = df0.iloc[num_points0].at['POSITION_T']
    else : 
        max_time = max(df1.iloc[num_points1].at['POSITION_T'], df0.iloc[num_points0].at['POSITION_T'])

    
    # Return the track with the most time points (ie. the one that is kept after fusion)
    if num_points0 < num_points1:
        return max_time, df1
    else:
        # print('df0_b: ' + str(max(df0['ELLIPSE_ASPECTRATIO'])))
        return max_time, df0
    
    
def write_to_file(output_file, string) :
    file = open(output_file, 'a')
    file.write(str(string))
    file.close

def write_to_csv(output_csv, row) : 
    file = open(output_csv, 'a')
    writer = csv.writer(file)
    writer.writerow(row)
    
# Sorts the parameters.csv file generated by curve-fitting, sorted by decreasing R2 score
def sort_parameters(filename):
    # Create or empty output file
    # @pre: output_file should be full path to output csv, including the name of the csv file
    output_file = "Kar9_Fusion_Data/sorted_parameters.csv"
    open(output_file, 'w').close()
    parameters = pd.read_csv(filename)
    parameters.sort_values(by = ['R2'], ascending = False, inplace = True, ignore_index = True)
    parameters.to_csv(output_file)
    
def process_per_phase(dataframe, frames_per_phase) :
    num_phases = int(math.floor(dataframe.shape[0]/(frames_per_phase + 1)))
    # print(type(num_phases))
    # print(type(frames_per_phase))

    tracks_per_phase = np.zeros((num_phases, frames_per_phase))
    time_per_phase = np.zeros((num_phases, frames_per_phase))
    
    for j in range(0, frames_per_phase) :
        for i in range(0, num_phases) :
            index = i * (frames_per_phase + 1) + j
            if index >= dataframe.shape[0]:
                break
            tracks_per_phase[i][j] = dataframe.iloc[index].at['ELLIPSE_ASPECTRATIO']
            time_per_phase[i][j] = dataframe.iloc[index].at['POSITION_T']
    # print(tracks_per_phase)
    # print(dataframe.shape)
    # print(time_per_phase)
    return tracks_per_phase, time_per_phase, num_phases

def plot_per_phase(num_phases, tracks_array, time_array,  max_time, data_folder, data_name) :
     
    # Label title and axes
    plt.suptitle('Aspect Ratio vs. Time by Phase')
    plt.title(data_name)
    plt.ylabel('Ellipse aspect ratio')
    plt.xlabel('Time (seconds)')
    
    # Set limits
    # @pre: Set aspect ratio y limit
    plt.ylim((0.75, 2.75))
    plt.xlim((0, max_time))
    
    for i in range(0, time_array.shape[1]) :
        color = 'C' + str(i)
        plt.plot(time_array[:, i], tracks_array[:, i], color = color, linewidth = 0.7)
        
    
    plt.savefig(data_folder + 'aspect_ratio_vs_time_per_phase.png')
    plt.savefig(data_folder + 'aspect_ratio_vs_time_per_phase.eps')

    plt.show()
    plt.clf()
    
def plot_single_intensity(x, y, data_folder, data_name) :
     # Label axes and title    
    ax1 = plt.subplot()
    
    ax1.plot(x, y, color = 'g')
    
    plt.suptitle('Intensity vs. Time')
    plt.title(data_name)

    plt.ylabel('Intensity of spots (u.a.)')
    plt.xlabel('Time (seconds)')
    
    # @pre: Set intensity y limit
    plt.ylim((0, 5000))
    # plt.xlim((0, max_time))
    
    
    plt.savefig(data_folder + "intensity_vs_time_py.png")
    plt.savefig(data_folder + "intensity_vs_time_py.eps")

    plt.show()
    plt.close()

    
if __name__ == "__main__" :
    main()