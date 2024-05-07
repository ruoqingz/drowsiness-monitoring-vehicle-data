import matplotlib.pyplot as plt
import json

import numpy as np


def sliding_window(data, w_l, w_step):
    window = []
    for i in range(0, len(data) - w_l + 1, w_step):
        window_data = []
        for val in data[i:i + w_l]:
            window_data.append(val)
        window.append(window_data)
    return window


def detect_drowsiness(ws, w_l, alpha, delta):
    drowsy = []
    for i in range(0, len(ws)):
        if (abs(max(ws[i], key=abs))) >= alpha:  # filter normal movements
            m = ws[i].index(max(ws[i], key=abs))
            s = 0
            e = w_l - 1
            if m != s and m != e and \
                    abs(ws[i][m] - ws[i][s]) / abs(m - s) >= delta and \
                    abs(ws[i][m] - ws[i][e]) / abs(m - e) >= delta:  # detect rapid change
                drowsy.append(1)
            else:
                drowsy.append(0)
        else:
            drowsy.append(0)
    return drowsy


def main():
    with open('../preprocess_data/SWA_data.json', 'r') as f:
        swa_data = json.load(f)

    time_interval = 0.01
    w_l = 300  # 3s
    w_step = 100  # 1s
    # alpha = 0.7
    delta = 0.01
    indexes = {}
    swa_velocity = {}
    var = {}
    # swa_acc={} I checked the acceleration, it doesn't give more info
    for name in swa_data.keys():
        delta_displacement = np.diff(swa_data[name])
        swa_velocity[name] = np.gradient(delta_displacement, time_interval)

        var[name]=np.var(swa_velocity[name])

        ws = sliding_window(swa_velocity[name], w_l, w_step)

        drowsy = detect_drowsiness(ws, w_l, 4*var[name], delta)

        indexes[name] = [i for i, val in enumerate(drowsy) if val == 1]  # at which second，the driver feels drowsy

    return indexes,var


if __name__ == '__main__':
    indexes ,var = main()
