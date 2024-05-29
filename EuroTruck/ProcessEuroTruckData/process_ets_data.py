## this model is to make the data from the csv file into a dictionary
import pandas as pd
import datetime


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

    def get(self, variable_name):
        return None

def process_data(dataframe_ets,dataframe_sw, volunteer_name):
    telemetry_data = TelemetryData(volunteer_name)
    date_format_ets = "%Y-%m-%d %H:%M:%S.%f"
    flag= True # flag to check if it is the first second after the cruise control is on
    for index, row in dataframe_ets.iterrows(): # iterate through each row in the dataframe
        if row["cruiseControlOn"]: # if cruise control is on
            if flag:
                datetime_start = datetime.datetime.strptime(row["realwordTime"], date_format_ets) # get the time when cruise control is on
                flag = False

            else:
                datetime_current = datetime.datetime.strptime(row["realwordTime"], date_format_ets) # get the current time
                if (datetime_current - datetime_start).seconds > 0:
                    telemetry_data.add(datetime_current.replace(microsecond=0), "SWA_data", row["userSteer"])
                    telemetry_data.add(datetime_current.replace(microsecond=0), "lateral_displacement_data", row["placement_y"])
                    telemetry_data.add(datetime_current.replace(microsecond=0), "lateral_acceleration_data", row["acceleration_y"])
                    telemetry_data.add(datetime_current.replace(microsecond=0), "YA_data", row["placement_yaw"])

    dataframe_sw['DS'] = dataframe_sw['DS'].replace('CALIBRATE', 0)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('AWAKE', 1)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('WEARY', 2)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('FATIGUED', 3)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('DROWSY', 4)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('Not valid!', -1)
    dataframe_sw['DS'] = dataframe_sw['DS'].replace('OFFWRIST', -1)
    date_format_sw = "%Y-%m-%d %H:%M:%S"                
    for index,row in dataframe_sw.iterrows():
        datetime_current = datetime.datetime.strptime(row["TIME"], date_format_sw)
        if datetime_current in telemetry_data.table:
            telemetry_data.add(datetime_current, "groud_truth", row["DS"])
        
    return telemetry_data





name = "ruoqing"
ets_data = pd.read_csv("telemetry_5.csv")
ground_truth_data=pd.read_excel("device_history_ruoqing.xlsx", engine='openpyxl')
data=process_data(ets_data, ground_truth_data, name)
