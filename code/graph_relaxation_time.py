#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:16:09 2022

@author: angelazhao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def main() :
    # Full or relative path to folder containing all data
    # @pre: folder should be the path of the parent folder containing all the folders that contain  each imaging acquisition
    folder = "Kar9_Fusion_Data/"
    # @pre: filename refers to the path to the csv containing the fusion time parameters generated by graph_contour_from_tracks.py
    filename = 'Kar9_Fusion_Data/parameters.csv'

    # # Output a csv of sorted parameters by decreasing R2
    # sort_parameters(filename)
    
    # Remove poor fits and outlier fusion times
    parameters_df = pd.read_csv(folder + "sorted_parameters.csv")
    print(parameters_df['FUSION_TIME'])

    parameters_df = process_data(parameters_df)
    print(parameters_df)
    diameter_np = get_radius(folder, parameters_df) * 2
    # check_size = diameter_np.size == parameters_df.size
    # print(diameter_np)
    parameters_df['DIAMETER'] = diameter_np.tolist()
    
    # Plot relaxation (fusion) time against diameter
    x = parameters_df['DIAMETER']
    y = parameters_df['FUSION_TIME']
    
    plot_T_vs_diameter(x, y, folder)
    
def plot_T_vs_diameter(x, y, folder) :
    
    # Label title and axes
    plt.title("Relaxation Time vs Diameter")
    plt.xlabel("Diameter (microns)")
    plt.ylabel("Relaxation Time (seconds)")
    
    # plt.xlim((0.02, 0.15))
    
    # Plot data points
    plt.plot(x, y, 'o')
    
    # Plot linear regression fit
    trend = linear_regression(x, y)
    trendpoly = np.poly1d(trend)
    y_fit = trendpoly(x)
    plt.plot(x, y_fit)
    
    # Print R2 for fit
    find_R2(y, y_fit)
    
    # @pre: check saving folder and filename
    # Save figure
    plt.savefig(folder + 'relaxation_time_vs_diameter.png')   
    plt.show()
    plt.clf()
    
    # Label title and axes
    plt.suptitle("Relaxation Time vs Diameter")
    plt.title("Log-log plot")
    plt.xlabel("Diameter (microns)")
    plt.ylabel("Relaxation Time (seconds)")
    
    
    # Plot data points
    plt.loglog(x, y, 'o')
    
    # Plot linear regression fit
    trend = linear_regression(x, y)
    trendpoly = np.poly1d(trend)
    y_fit = trendpoly(x)
    plt.loglog(x, y_fit)
    
    # Print R2 for fit
    find_R2(y, y_fit)
    
    # @pre: check saving folder and filename
    # Save figure
    plt.savefig(folder + 'relaxation_time_vs_diameter_loglog.png')   
    plt.show()
    plt.clf()
    
def linear_regression(x, y) :
    model = np.polyfit(x, y, 1)
    return model
    
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

def process_data(dataframe) :
    # Remove poor fits and clearly wrong fusion times
    mask1 = dataframe['FUSION_TIME'] > 0
    mask2 = dataframe['FUSION_TIME'] < 5
    
    df1 = dataframe[mask1]
    df2 = df1[mask2]
    
    # for index, row in dataframe.iterrows() :
    #     fusion_time = dataframe.iloc[index].at['FUSION_TIME']
    #     print('fusion time' + str(index) + " = " + str(fusion_time))
    #     if fusion_time < 0 :
    #         dataframe.drop(index, inplace = True)
    #     if fusion_time > 100 :
    #         print('fusion time' + str(fusion_time))
    #         dataframe.drop(index, inplace = True)
    
    # dataframe = pd.DataFrame(data = np.where((dataframe['FUSION_TIME'] > 0) | (dataframe['FUSION_TIME'] < 100)))
    
    return df2
    # Remove linear fits?
    
def get_radius(folder, dataframe) :
    radius_np = np.zeros(dataframe.shape[0])
    
    for index in range(0, dataframe.shape[0] - 1) :
    #for index, row in dataframe.iterrows() :
        data_folder = dataframe.iloc[index].at['DATA_FOLDER']
        contours = pd.read_csv(folder + data_folder + "/output/contour.csv")
        radius_np[index] = calculate_mean_radius_before_fusion(contours)
    
    return radius_np
        

# Clip x and y so that it only includes points after fusion start, assuming fusion start is when aspect ratio is maximized
def calculate_mean_radius_before_fusion(dataframe):
    
    # Find fusion start at max ELLIPSE_ASPECTRATIO
    dataframe.sort_values(by = 'FRAME', inplace = True, ignore_index = True)
    y = dataframe['ELLIPSE_ASPECTRATIO']
    
    y_np = y.to_numpy(y)
    fusion_start_index = np.argmax(y_np)
    
    # Calculate mean of all the radius measurements before fusion_start_index/2
    index = math.ceil(fusion_start_index/2)
    sum_radius = 0
    for i in range(0, index) :
        sum_radius = sum_radius + dataframe.iloc[i].at['RADIUS']
    
    mean_radius = sum_radius/(index + 1)
    
    return mean_radius
    
# Sorts the parameters.csv file generated by curve-fitting, sorted by decreasing R2 score
def sort_parameters(filename):
    # Create or empty output file
    output_file = "Kar9_Fusion_Data/sorted_parameters.csv"
    open(output_file, 'w').close()
    parameters = pd.read_csv(filename)
    parameters.sort_values(by = ['R2'], ascending = False, inplace = True, ignore_index = True)
    parameters.to_csv(output_file)

if __name__ == "__main__" : 
    main()