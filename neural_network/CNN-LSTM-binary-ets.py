import datetime

import numpy as np
from EuroTruck.ProcessEuroTruckData.process_ets_data import data


def seperate_ground_truth(data,size,step):
    awake_window = []
    light_drowsy_window = []
    drowsy_window = []
    time = next(iter(data.table))
    while time + datetime.timedelta(seconds=size) in data.table:
        awake_count = 0
        light_drowsy_count = 0
        drowsy_count = 0
        for i in range(size):
            check_time = time + datetime.timedelta(seconds=i)
            if check_time not in data.table:
                break
            if data.table[check_time]["groud_truth"] == [1]:
                awake_count+=1
            if data.table[check_time]["groud_truth"] == [2] or data.table[check_time]["groud_truth"] == [3]:
                light_drowsy_count+=1
            if data.table[check_time]["groud_truth"] == [4]:
                drowsy_count+=1    
                
        if awake_count == size:
            awake_window.append([time + datetime.timedelta(seconds=i) for i in range(size)])
        if light_drowsy_count == size:
            light_drowsy_window.append([time + datetime.timedelta(seconds=i) for i in range(size)])
        if drowsy_count == size:
            drowsy_window.append([time + datetime.timedelta(seconds=i) for i in range(size)])

        step_seconds = datetime.timedelta(seconds=step)
        time += step_seconds

    # drowsy window is a list of list of datetime objects which is the key
    return awake_window, light_drowsy_window, drowsy_window

def define_feature_matrix(data,windows):
    if windows == []:
        return np.empty((0, 350,2))
    feature_matrix = []
    for window in windows:
        feature_matrix_per_window = []

        SWA_colunm = []
        for time in window:
            SWA_colunm.extend(data.table[time]["SWA_data"])
        feature_matrix_per_window.append(SWA_colunm)

        SWV_column = []
        for time in window:
            SWV_column.extend(data.table[time]["SWV_data"])
        feature_matrix_per_window.append(SWV_column)

        LD_column = []
        for time in window:
            LD_column.extend(data.table[time]["lateral_displacement_data"])
        feature_matrix_per_window.append(LD_column)
        LA_column = []

        for time in window:
            LA_column.extend(data.table[time]["lateral_acceleration_data"])
        feature_matrix_per_window.append(LA_column)
        
        feature_matrix_per_window=np.transpose(np.array(feature_matrix_per_window))
        feature_matrix.append(feature_matrix_per_window)
    feature_matrix=np.array(feature_matrix)
    return feature_matrix

awake_window, light_drowsy_window, drowsy_window=seperate_ground_truth(data,10,1)
awake_feature_matrix=define_feature_matrix(data,awake_window)
light_drowsy_feature_matrix=define_feature_matrix(data,light_drowsy_window)
drowsy_feature_matrix=define_feature_matrix(data,drowsy_window)

