# this model is to make the data from the csv file into a dictionary
import pandas as pd
import datetime
import pickle
import os


class TelemetryData:
    def __init__(self, volunteer_name):
        self.name = volunteer_name
        self.table = {}

    def add(self, time, variable_name, value):
        if time not in self.table:
            self.table[time] = {}
        if variable_name not in self.table[time]:
            self.table[time][variable_name] = []
        self.table[time][variable_name].append(value)

    def delete(self, time):
        if time in self.table:
            del self.table[time]

    def get_min_sampling_rate(self):
        rate = 40
        for time in self.table:
            if len(self.table[time]["SWA_data"]) < rate:
                rate = len(self.table[time]["SWA_data"])
        return 30  #I changed here to force the rate to be 30

    def trim_variable_length(self, variable_name, length):
        for time in self.table:
            self.table[time][variable_name] = self.table[time][variable_name][:length]


def standardize_data(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


def process_data(dataframe_ets, dataframe_sw, volunteer_name):
    telemetry_data = TelemetryData(volunteer_name)
    slot = []
    datetime_start = []
    datetime_end = []
    for index, row in dataframe_ets.iterrows():  # iterate through each row in the dataframe
        if index > 0:
            prev_row = dataframe_ets.iloc[index - 1]
            if row["cruiseControlOn"] == True and prev_row["cruiseControlOn"] == False:
                # get the start time when cruise control is on
                datetime_start.append(row["realwordTime"])
                slot.append(index)

            if ((row["cruiseControlOn"] == False and prev_row["cruiseControlOn"] == True) or
                    (index == len(dataframe_ets) - 1 and row["cruiseControlOn"] == True)):
                # get the end time when cruise control is on
                datetime_end.append(prev_row["realwordTime"])
                slot.append(index)

    for i in range(0, len(slot), 2):
        start = slot[i]
        end = slot[i + 1]
        dataframe_ets.loc[start:end, "userSteer"] = standardize_data(dataframe_ets.loc[start:end, "userSteer"].fillna(0))
        dataframe_ets.loc[start:end, "userSteer_derivative"] = standardize_data(
            dataframe_ets.loc[start:end, "userSteer_derivative"])
        dataframe_ets.loc[start:end, "placement_y"] = standardize_data(dataframe_ets.loc[start:end, "placement_y"].fillna(0))
        dataframe_ets.loc[start:end, "acceleration_y"] = standardize_data(
            dataframe_ets.loc[start:end, "acceleration_y"].fillna(0))
        for index, row in dataframe_ets.iloc[start:end].iterrows():
            datetime_current = row["realwordTime"]
            telemetry_data.add(datetime_current.replace(microsecond=0), "SWA_data", row["userSteer"])
            telemetry_data.add(datetime_current.replace(microsecond=0), "SWV_data", row["userSteer_derivative"])
            telemetry_data.add(datetime_current.replace(microsecond=0), "lateral_displacement_data", row["placement_y"])
            telemetry_data.add(datetime_current.replace(microsecond=0), "lateral_acceleration_data",
                               row["acceleration_y"])
            # telemetry_data.add(datetime_current.replace(microsecond=0), "YA_data", row["placement_yaw"])

    for item in datetime_start:
        telemetry_data.delete(item.replace(microsecond=0))
    for item in datetime_end:
        telemetry_data.delete(item.replace(microsecond=0))

    min_rate = telemetry_data.get_min_sampling_rate()
    telemetry_data.trim_variable_length("SWA_data", min_rate)
    telemetry_data.trim_variable_length("SWV_data", min_rate)
    telemetry_data.trim_variable_length("lateral_displacement_data", min_rate)
    telemetry_data.trim_variable_length("lateral_acceleration_data", min_rate)
    # telemetry_data.trim_variable_length("YA_data", min_rate)

    pd.set_option('future.no_silent_downcasting', True)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('CALIBRATE', 0)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('AWAKE', 1)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('WEARY', 2)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('FATIGUED', 3)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('DROWSY', 4)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('Not valid!', -1)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('OFFWRIST', -1)
    date_format_sw = "%Y-%m-%d %H:%M:%S"
    for index, row in dataframe_sw.iterrows():
        datetime_current = datetime.datetime.strptime(row["TIME"], date_format_sw)
        if datetime_current in telemetry_data.table.keys():
            telemetry_data.add(datetime_current, "groud_truth", row["DS"])

    # there could be gap in ground truth but I hope there is no
    # 假设 telemetry_data.table 是需要处理的字典
    telemetry_data.table = {time: data for time, data in telemetry_data.table.items() if len(data) >= 5}

    return telemetry_data


def prepare_export_data(telemetry_files, predictS_files, name):
    ets_data_list = [pd.read_csv(file) for file in telemetry_files]
    ets_data = pd.concat(ets_data_list, ignore_index=True)

    ets_data["realwordTime"] = pd.to_datetime(ets_data["realwordTime"])
    ets_data['time_Diff'] = ets_data['realwordTime'].diff().dt.total_seconds()
    ets_data=ets_data[ets_data['time_Diff'] > 0].reset_index(drop=True)
    ets_data['userSteer_derivative'] = ets_data['userSteer'].diff() / ets_data['time_Diff']

    ground_truth_data_list = [pd.read_excel(file, engine='openpyxl') for file in predictS_files]
    ground_truth_data = pd.concat(ground_truth_data_list, ignore_index=True)
    ground_truth_data=ground_truth_data.drop_duplicates(subset=['TIME'], keep='first').reset_index(drop=True)
    data = process_data(ets_data, ground_truth_data, name)
    return data


os.chdir("/Users/ruotsing/PycharmProjects/DMS/EuroTruck/ProcessEuroTruckData")
csv_files_rq = ["telemetry_0530_1_rq.csv", "telemetry_0530_2_rq.csv", "telemetry_0530_3_rq.csv",
                "telemetry_0531_1_rq.csv",
                "telemetry_0531_2_rq.csv", "telemetry_0531_3_rq.csv", "telemetry_0531_4_rq.csv",
                "telemetry_0531_5_rq.csv"]
csv_files_michele = ["telemetry_0604_1_michele.csv", "telemetry_0604_2_michele.csv", "telemetry_0604_3_michele.csv",
                     "telemetry_0604_4_michele.csv"]
csv_files_sara= ["telemetry_0607_1_sara.csv", "telemetry_0607_2_sara.csv"]
csv_files = csv_files_rq + csv_files_michele + csv_files_sara

ground_truth_data_files_rq = ["device_history_0530_rq.xlsx", "device_history_0531_rq.xlsx"]
ground_truth_data_files_michele = ["device_history_0604_michele.xlsx"]
ground_truth_data_files_sara = ["device_history_0607_sara.xlsx"]
ground_truth_data_files = ground_truth_data_files_rq + ground_truth_data_files_michele + ground_truth_data_files_sara


data_rq = prepare_export_data(csv_files_rq, ground_truth_data_files_rq, "ruoqing")
data_michele = prepare_export_data(csv_files_michele, ground_truth_data_files_michele, "michele")
data_sara = prepare_export_data(csv_files_sara, ground_truth_data_files_sara, "sara")
data_group = prepare_export_data(csv_files, ground_truth_data_files, "ruoqing+michele+sara")
