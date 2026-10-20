#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:33:38 2024

@author: bella
"""

#%% Import Libraries

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import signal


#%% Detect HS and TOs from Cranial-Caudal Chest Acceleration


def get_chest_accel_events(data_chest_accel, subject_id):
    
    """
    This function uses wavelet transformation to identify indices that indicate
    heal strike and toe off events from chest acceleration walking data.

    Parameters
    ----------
    data_chest_accel : length of sample x 4 Array of float
        Column 1: Time (seconds)
        Column 2: Accel_x(g), Anterior-Posterior
        Column 3: Accel_y (g), Cranial-Caudal
        Column 4: Accel_z (g), Medial - Lateral 

    Returns
    -------
    HS_inds : tuple of size 2
        First tuple element is array of int representing the indices where HS is detected
    TO_inds : tuple of size 2
        First tuple element is array of int representing the indices where TO is detected.

    """
    scale = 20
    
    # Detect HSs
    
    # Step 1: Define your data (time and linear acceleration)
    time_chest = data_chest_accel.iloc[:, 0] # time array from dictionary -- already in s
    signal_accel = data_chest_accel.iloc[:, 2]     # cranial-caudal chest accel (y-direction)
    
    # check for match
    data_chest_accel.set_index(time_chest, inplace=True)
    
    # If you want to check the sampling freq:
    time_s = time_chest  
    dt = np.mean(np.diff(time_s)) # calculate the time difference 
    fs = 1/dt            # calculate the sampling frequency 
    
    # for a better wavelet analysis, use only walking data. So either crop data here or when it is imported
    plt.figure(subject_id, clear=True)
    plt.plot(time_chest, signal_accel, label='raw_data')
    plt.title(f'Chest Accel - Subject {subject_id}')
    plt.ylabel('CC Accel. (g)')
    plt.xlabel('Time (seconds)')
    
    # Step 2: Define a mother wavelet
    mother_wavelet = signal.morlet2
    
    # Step 3: Define a scale
    scales = np.arange(1,65)  # range of scales (20)
    
    # Step 4: Apply Cont. Wavelet Transformation
    cwt_hs = abs(signal.cwt(signal_accel, mother_wavelet, widths=scales))   
 
    # Step 5: Plot Contour Plot to determine which scale is best 
    fig_hs, ax_hs = plt.subplots()  
    ax_hs.clear()
    im_hs = ax_hs.contourf(np.arange(len(signal_accel)), scales, cwt_hs, cmap='viridis')
    ax_hs.set_ylabel('Scale')
    ax_hs.set_xlabel('Data Point')
    ax_hs.set_title(f'Contour Plot for Accel Heel Strikes - Subject {subject_id}')
    fig_hs.colorbar(im_hs)
    fig_hs.tight_layout()
    
    # Step 6: Plot wavlet transformation ontop of raw accel signal 
    # goal: select a scale that allows wavelet transform peaks to allign with heel strikes
    plt.figure(subject_id)
    plt.plot(time_chest, abs(cwt_hs[scale,:]), label = f'morlet scale {scale}')
    plt.legend()
    # The highest peaks from cwt_hs are associated HS as can be seen above
    
    
    # Detect TOs
    
    # Step 1: Define your data (time and linear acceleration)
    # Same as step 1 for HS
    
    # Step 2: Define a mother wavelet
    mother_wavelet = signal.morlet2
    
    # Step 3: Define a scale
    scales = np.arange(1,65)  # range of scales (20)
    
    # Step 4: Apply Cont. Wavelet Transformation
    cwt_to = abs(signal.cwt(signal_accel, mother_wavelet, widths=scales))

    # Step 5: Plot Contour Plot for toe-offs to determine which scale is best 
    fig_to, ax_to = plt.subplots()  
    ax_to.clear()
    im_to = ax_to.contourf(np.arange(len(signal_accel)), scales, cwt_to, cmap='viridis')
    ax_to.set_ylabel('Scale')
    ax_to.set_xlabel('Data Point')
    ax_to.set_title(f'Contour Plot for Accel Toe-Offs - Subject {subject_id}')
    fig_to.colorbar(im_to)
    fig_to.tight_layout()
    
    
    # Find Peaks for HS
    HS_inds = signal.find_peaks(cwt_hs[scale,:], prominence = 0.19 )
    plt.figure(subject_id)
    plt.plot(time_chest.iloc[HS_inds[0]], abs(cwt_hs[scale, HS_inds[0]]), 'v', label='HS Events')
    plt.legend()
    
    # Find Throughts for TO (not always peaks or troughs-- whatever 
    #consistent element you can find that aligns with your events of interest)
    TO_inds = signal.find_peaks(-cwt_hs[scale,:], prominence = 0.19 )
    plt.plot(time_chest.iloc[TO_inds[0]], abs(cwt_hs[scale, TO_inds[0]]), 's', label='TO Events')
    plt.legend()
             
             
    return HS_inds, TO_inds



def get_shank_gyro_events(data_shank_gyro, subject_id):
    '''
    Performs wavelet transformation on parameter data to identify points that indicate
    heal strike and toe off events from shank gyroscope walking data.

    Parameters
    ----------
    data_shank_gyro : length of sample x 4 Array of float
        Column 1: Timestamps (microseconds) in POSIXTIME
        Column 2: Gyro_x(deg/s), Anterior-Posterior
        Column 3: Gyro_y (deg/s), Cranial-Caudal
        Column 4: Gyro_z (deg/s), Medial - Lateral 

    Returns
    -------
    HS_inds : tuple of size 2
        First tuple element is array of int representing the indices where HS is detected
    TO_inds : tuple of size 2
        First tuple element is array of int representing the indices where TO is detected.

    '''
   
    # define scale
    gyro_scale = 25
    
    # define data
    time_shank = data_shank_gyro.iloc[:, 0] # time array from dictionary -- already in s
    signal_gyro = data_shank_gyro.iloc[:, 3] # ML gyro data
    
    # check match
    data_shank_gyro.set_index(time_shank, inplace=True)
    
    # plot raw data
    plt.figure(int(subject_id)+30, clear=True)
    plt.plot(time_shank, signal_gyro, label='raw_data')
    plt.title(f'Shank Gyro - Subject {subject_id}')
    plt.ylabel('ML Gyro. (deg/s)')
    plt.xlabel('Time (seconds)')
    
    # define mother wavelet
    mother_wavelet_gyro = signal.morlet2
    
    # define a scale
    gyro_scales = np.arange(1,65)  
    
    # apply transformation
    gyro_cwt_hs = abs(signal.cwt(signal_gyro, mother_wavelet_gyro, widths=gyro_scales))   

    # contour plot, HS
    gyro_fig_hs, gyro_ax_hs = plt.subplots()  
    gyro_ax_hs.clear()
    gyro_im_hs = gyro_ax_hs.contourf(np.arange(len(signal_gyro)), gyro_scales, gyro_cwt_hs, cmap='viridis')
    gyro_ax_hs.set_ylabel('Scale')
    gyro_ax_hs.set_xlabel('Data Point')
    gyro_ax_hs.set_title(f'Contour Plot for Gyro Heel Strikes - Subject {subject_id}')
    gyro_fig_hs.colorbar(gyro_im_hs)
    gyro_fig_hs.tight_layout()
    
    # add morlet scale to raw data
    plt.figure(int(subject_id)+30)
    plt.plot(time_shank, abs(gyro_cwt_hs[gyro_scale,:]), label = f'morlet scale {gyro_scale}')
    plt.legend()
    
    # apply transformation
    gyro_cwt_to = abs(signal.cwt(signal_gyro, mother_wavelet_gyro, widths=gyro_scales))

    # Step 5: Plot Contour Plot for toe-offs to determine which scale is best 
    gyro_fig_to, gyro_ax_to = plt.subplots()  
    gyro_ax_to.clear()
    gyro_im_to = gyro_ax_to.contourf(np.arange(len(signal_gyro)), gyro_scales, gyro_cwt_to, cmap='viridis')
    gyro_ax_to.set_ylabel('Scale')
    gyro_ax_to.set_xlabel('Data Point')
    gyro_ax_to.set_title(f'Contour Plot for Gyro Toe-Offs - Subject {subject_id}')
    gyro_fig_to.colorbar(gyro_im_to)
    gyro_fig_to.tight_layout()
    
    # Find Peaks for HS
    gyro_HS_inds = signal.find_peaks(gyro_cwt_hs[gyro_scale,:], prominence = 0.19 )
    plt.figure(int(subject_id)+30)
    plt.plot(time_shank.iloc[gyro_HS_inds[0]], abs(gyro_cwt_hs[gyro_scale, gyro_HS_inds[0]]), 'v', label='HS Events')
    plt.legend()
    
    # Find Throughts for TO (not always peaks or troughs-- whatever 
    #consistent element you can find that aligns with your events of interest)
    gyro_TO_inds = signal.find_peaks(-gyro_cwt_hs[gyro_scale,:], prominence = 0.19 )
    plt.plot(time_shank.iloc[gyro_TO_inds[0]], abs(gyro_cwt_hs[gyro_scale, gyro_TO_inds[0]]), 's', label='TO Events')
    plt.legend()
    
    return gyro_HS_inds, gyro_TO_inds

def check_events(HS_inds, TO_inds):
    # turn hs and to into lists
    HS_inds_list = list(HS_inds)
    TO_inds_list = list(TO_inds)
    
    # create a data frame for strike type (hs vs to) and times of events
    df = pd.DataFrame(columns=['Strike type', 'Time'])
    
    # want to use shortest list range
    if len(HS_inds_list) < len(TO_inds_list):
        max_index = len(HS_inds_list)
    else:
        max_index = len(TO_inds_list)
          
    # compare hs and to lists
    # each loop will be new versions of the lists based on what is deleted     
    for i in range(max_index):
        # get 0 index in each list 
        TO_val = TO_inds_list[0]
        HS_val = HS_inds_list[0]
        # if current to time is less than hs time, remove the to from the list
        # and append to data frame
        if TO_val < HS_val:
            v = TO_inds_list.pop(0)
            df.loc[len(df.index)] = ['TO', v]
        # otherwise, remove hs from list and append to data frame
        else:
            v = HS_inds_list.pop(0)
            df.loc[len(df.index)] = ['HS', v]
    
    
    # now have a DF of HS and TO times in ascending order
    # does not assure that HS and TO are alternating, just times
    # now need to make sure there are not 2 HS's before the next TO for example
    
    # separate hs and to
    HS = df[df['Strike type'] == 'HS']['Time']
    TO = df[df['Strike type'] == 'TO']['Time']
    
    # find times between same events
    hs_diffs = np.diff(HS)
    to_diffs = np.diff(TO)
 
    # find mean and std
    hs_mean = np.mean(np.abs(hs_diffs))
    to_mean = np.mean(np.abs(to_diffs))
    hs_std = np.std(np.abs(hs_diffs))
    to_std = np.std(np.abs(to_diffs))
    
    # set threshold based on mean and std
    hs_threshold = hs_mean - 2 * hs_std
    to_threshold = to_mean - 2 * to_std
    
    # initiaize lists to delete
    to_delete = []
    hs_delete = []
    
    # for each index in to indices, if the diff is < threshold, delete from indices
    for index, diff in enumerate(to_diffs):
        if diff < to_threshold:
          
            to_delete.append(index+1) # add one to index because want to keep the first index as the real point and delete the one after as the false detected point
    # new to inds
    TO = np.delete(TO, to_delete)
            
    # for each index in to indices, if the diff is < threshold, delete from indices
    for index, diff in enumerate(hs_diffs):
        if diff < hs_threshold:
          
            hs_delete.append(index+1) # add one to index because want to keep the first index as the real point and delete the one after as the false detected point
    
    # new to inds
    HS = np.delete(HS, to_delete)
    # print to see what it deleted 
    print (to_delete)     
    print(hs_delete)
    
    return df, HS, TO


# def check_events(HS_inds, TO_inds):
#     min_length = min(len(TO_inds), len(HS_inds))
#     print(min_length)
    
#     to_delete = []
#     if HS_inds[0] <= TO_inds[0]:
#         for i in range(min_length):
#             if HS_inds[i] >= TO_inds[i]:
#                 to_delete.append(i)
#     else:
#         to_delete.append(0)
#         for i in range(min_length):
#             if HS_inds[i] >= TO_inds[i+1]:
#                 to_delete.append(i+1)
#     print(to_delete)                    
#     TO_inds = np.delete(TO_inds, to_delete)
#     return TO_inds
    
        

# def normalize_gait():
    