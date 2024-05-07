import pandas as pd
import os


# filter null ones, it is not used in the project
def reform_ground_truth(ground_truth):
    ground_truth_new = {}
    for name in ground_truth.keys():
        ground_truth_new[name] = []
        for values in ground_truth[name].values():
            if values is not None:
                ground_truth_new[name].append(values)

    return ground_truth_new


# it is not used in the project
def ground_truth_sep(ground_truth, size, step):
    awake_window = {}
    light_drowsy_window = {}
    drowsy_window = {}
    for name in ground_truth.keys():
        awake_window[name] = []
        light_drowsy_window[name] = []
        drowsy_window[name] = []
        i = 0
        while i < len(ground_truth[name]) - size:
            if all(x == 1 for x in ground_truth[name][i:i + size]):
                awake_window[name].append([i + j for j in range(0, size)])
                i = i + step
            elif all(x == 2 or x == 3 for x in ground_truth[name][i:i + size]):
                light_drowsy_window[name].append([i + j for j in range(0, size)])
                i = i + step
            elif all(x == 4 for x in ground_truth[name][i:i + size]):
                drowsy_window[name].append([i + j for j in range(0, size)])
                i = i + step
            else:
                i = i + 1

    return awake_window, light_drowsy_window, drowsy_window


if __name__ == "__main__":
    pd.set_option('future.no_silent_downcasting', True)
    data = pd.DataFrame()
    folder_path = '/Users/ruotsing/Documents/DMS/Raw Data/Danisi'
    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)

            df = pd.read_excel(file_path, engine='openpyxl')
            df['DS'] = df['DS'].replace('CALIBRATE', 0)
            df['DS'] = df['DS'].replace('AWAKE', 1)
            df['DS'] = df['DS'].replace('WEARY', 2)
            df['DS'] = df['DS'].replace('FATIGUED', 3)
            df['DS'] = df['DS'].replace('DROWSY', 4)
            df['DS'] = df['DS'].replace('Not valid!', -1)
            df['DS'] = df['DS'].replace('OFFWRIST', -1)

            column_name = os.path.splitext(filename)[0]
            data[column_name] = df['DS']

    data['vasanth'] = data['vasanth'][1962:5217].reset_index(drop=True)
    data['catia'] = data['catia'][1360:5103].reset_index(drop=True)
    data['michele'] = data['michele'][1755:5524].reset_index(drop=True)
    data['stefano'] = data['stefano'][670:4180].reset_index(drop=True)
    data['emanuele'] = data['emanuele'][420:4136].reset_index(drop=True)
    data['gea'] = data['gea'][756:4438].reset_index(drop=True)
    data['giulio'] = data['giulio'][368:4169].reset_index(drop=True)

    data.to_json('ground_truth.json', orient='columns')
