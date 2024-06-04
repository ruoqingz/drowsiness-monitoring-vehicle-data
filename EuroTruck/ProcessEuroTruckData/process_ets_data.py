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
        return rate

    def trim_variable_length(self, variable_name, length):
        for time in self.table:
            self.table[time][variable_name] = self.table[time][variable_name][:length]


def process_data(dataframe_ets, dataframe_sw, volunteer_name):
    telemetry_data = TelemetryData(volunteer_name)
    slot=[]
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
        for index, row in dataframe_ets.iloc[start:end].iterrows():
            datetime_current = row["realwordTime"]
            telemetry_data.add(datetime_current.replace(microsecond=0), "SWA_data", row["userSteer"])
            telemetry_data.add(datetime_current.replace(microsecond=0), "SWV_data", row["userSteer_derivative"])
            telemetry_data.add(datetime_current.replace(microsecond=0), "lateral_displacement_data", row["placement_y"])
            telemetry_data.add(datetime_current.replace(microsecond=0), "lateral_acceleration_data", row["acceleration_y"])
            # telemetry_data.add(datetime_current.replace(microsecond=0), "YA_data", row["placement_yaw"])

    for item in datetime_start:
        telemetry_data.delete(item.replace(microsecond=0))
    for item in datetime_end:
        telemetry_data.delete(item.replace(microsecond=0))

    min_rate=telemetry_data.get_min_sampling_rate()
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


os.chdir("/Users/ruotsing/PycharmProjects/DMS/EuroTruck/ProcessEuroTruckData")
csv_files = ["telemetry_1.csv", "telemetry_2.csv", "telemetry_3.csv"]
name = "ruoqing"
ets_data_list = [pd.read_csv(file) for file in csv_files]
ets_data = pd.concat(ets_data_list, ignore_index=True)

ets_data["realwordTime"] = pd.to_datetime(ets_data["realwordTime"])
ets_data['time_Diff'] = ets_data['realwordTime'].diff().dt.total_seconds()
ets_data['userSteer_derivative'] = ets_data['userSteer'].diff() / ets_data['time_Diff']
# ets_data['placement_yaw_derivative'] = ets_data['placement_yaw'].diff() / ets_data['time_Diff']

ground_truth_data = pd.read_excel("device_history_ruoqing.xlsx", engine='openpyxl')
data = process_data(ets_data, ground_truth_data, name)

