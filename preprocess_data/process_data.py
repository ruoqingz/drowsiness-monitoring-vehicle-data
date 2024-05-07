import pandas as pd
import os
import json


def process_data(extract_name, json_file):
    folder_path = '/Users/ruotsing/Documents/DMS/Preprocessed_Data/simulator data'
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dict_data = {}
    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        column_name = os.path.splitext(file_name)[0]

        column_data = df[extract_name].tolist()
        dict_data[column_name] = column_data

        with open(json_file, 'w') as f:
            json.dump(dict_data, f)


if __name__ == '__main__':
    process_data('VI_CarRealTime.Inputs.Driver_Demands.str_swa', 'SWA_data.json')
    process_data('VI_CarRealTime.Outputs.chassis_displacements.lateral',
                 'lateral_displacement_data.json')
    process_data('VI_CarRealTime.Outputs.chassis_accelerations.lateral',
                 'lateral_acceleration_data.json')
    process_data('VI_CarRealTime.Outputs.chassis_accelerations.longitudinal',
                 'longitudinal_acceleration_data.json')
    process_data('VI_CarRealTime.Outputs.chassis_displacements.yaw', 'YA_data.json')
