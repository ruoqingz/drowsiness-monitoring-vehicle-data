import json
import statistics
import numpy as np
import math
import matplotlib.pyplot as plt


def vector_difference(x, y):
    result = []
    if len(x) == len(y) and len(x) != 0:
        for i in range(0, len(x)):
            result.append(abs(x[i] - y[i]))
        return result
    elif len(x) == 0 or len(y) == 0:
        raise ValueError('Invalid input: Lists must have non-zero length')
    elif len(x) != len(y):
        raise ValueError('Invalid input: Lists must have the same length')


def sliding_window(data, w_l, w_step):
    window = []
    for i in range(0, len(data) - w_l + 1, w_step):
        window_data = []
        for val in data[i:i + w_l]:
            window_data.append(val)
        window.append(window_data)
    return window


def compute_c_log(data, m, r):
    xs = []
    for i in range(0, len(data) - m + 1):
        x = []
        for val in data[i:i + m]:
            x.append(val)
        xs.append(x)

    B = []
    count = 0
    for i in range(0, len(data) - m + 1):
        for j in range(0, len(data) - m + 1):
            if max(vector_difference(xs[i], xs[j])) < r:
                count += 1
        B.append(count)
        count = 0

    base = math.e
    c_m_log = [math.log(i / (len(data) - m + 1), base) for i in B]

    return c_m_log


def approximate_entropy(m, data):
    r = 0.2 * statistics.stdev(data)

    c_m_log = compute_c_log(data, m, r)
    c_m_1_log = compute_c_log(data, m + 1, r)

    ap_en = sum(c_m_log[0:len(data) - m + 1]) / (len(data) - m + 1) - sum(c_m_1_log[0:len(data) - m]) / (len(data) - m)

    # print(ap_en)
    return abs(ap_en)


def ap_linear_approximation(series, threshold):
    segments = [[]]
    b_array = [[]]
    index = 0
    start = 0
    end = len(series) - 1
    segments[index] = list(range(start, end + 1))
    b_array[index] = np.polyfit(np.array(segments[index]), series, 1)
    approximated = np.polyval(b_array[index], np.array(segments[index]))

    while True:
        while abs(max(vector_difference(approximated, series[start:end + 1]))) > threshold or \
                abs(min(vector_difference(approximated, series[start:end + 1]))) > threshold:
            end = end - 1
            segments[index] = list(range(start, end + 1))
            b_array[index] = np.polyfit(np.array(segments[index]), series[start:end + 1], 1)
            approximated = np.polyval(b_array[index], np.array(segments[index]))
            if end <= start:
                break
        index += 1
        start = end + 1
        end = len(series) - 1
        segments.append(list(range(start, end + 1)))
        if len(segments[index]) <= 1:
            segments.pop()
            break
        b_array.append([])
        b_array[index] = np.polyfit(np.array(segments[index]), series[start:end + 1], 1)
        approximated = np.polyval(b_array[index], np.array(segments[index]))

    approximated_array = []
    for i in range(0, len(segments)):
        # approximated_array.append(np.polyval(b_array[i], segments[i]))  #返回每一个分段的合集
        approximated_array.extend(np.polyval(b_array[i], segments[i]).tolist())  # 消除了分段

    return segments, approximated_array, b_array


def dynamic_time_warping(q_array, c_array):
    gamma = []
    for i in range(0, len(q_array)):
        row = []
        for j in range(0, len(c_array)):
            if i == 0 and j == 0:
                row.append(0)
            elif i == 0:
                row.append(float('inf'))
            elif j == 0:
                row.append(float('inf'))
            else:
                row.append((q_array[i] - c_array[j]) ** 2 + min(row[j - 1], gamma[i - 1][j - 1], gamma[i - 1][j]))
        gamma.append(row)
    return gamma[len(q_array) - 1][len(c_array) - 1]


# def fitted_value(segments, b_array, w_l):
#     fitted = []
#     new_axis = []
#     for i in range(0, len(segments)):
#         new_axis.append(np.arange(segments[i][0], segments[i][-1]+1, 1 / w_l))
#         fitted.extend(np.polyval(b_array[i], new_axis[-1]))
#     axis_list=[]
#     for arr in new_axis:
#         axis_list.extend(arr.tolist())
#     plt.plot(axis_list,fitted)
#     plt.savefig('fitted_plot.svg')
#
#     fitted = sliding_window(fitted, w_l, w_l)
#     return fitted


def self_adaptive_segments(approximated_array):  # the way to find "self adaptive" is not clear
    S = []
    S = sliding_window(approximated_array, 30, 30)
    spacer = list(range(0, len(approximated_array), 30))

    # i = 0
    # S = []
    # variation = []
    # spacer = []
    #
    # while (i < len(approximated_array) - 30):
    #     for j in range(i + 30, i + 60):
    #         variation.append(np.var(approximated_array[i:j]))
    #     min_value = min(variation)
    #     min_index = variation.index(min_value)
    #     S.append(approximated_array[i:i + (min_index + 30)])
    #     spacer.append(i)
    #     i += (min_index + 30)
    #     variation.clear()

    return S, spacer


def find_reference_sample(s):
    for i in range(len(s) - 1, -1, -1):
        if np.var(s[i]) < 0.01:
            return i
    return 0


def detect_drowsiness_pattern(approximated_array):
    f = []
    S, spacer = self_adaptive_segments(approximated_array)

    # for k in range(1, len(S) - 1):
    #     if len(S[k]) < l - 1:
    #         threshold = 0.31
    #     else:
    #         threshold = 2
    threshold = 1
    dwt = []
    for k in range(0, len(S)):
        i = find_reference_sample(S[0:k])
        dwt = dynamic_time_warping(S[i], S[k])
        if dwt < threshold:
            f.append(0)
        else:
            f.append(1)

    return dwt, f


# def standard_linear_time_series(fitted, j):
#     S = fitted[j - 1]
#     length = j - (j - 1)
#     for i in range(j - 2, -1, -1):
#         cov = np.cov(fitted[i], fitted[j])[0, 1]
#         print(cov)
#         if cov < 0.01:
#             if j - i > length:
#                 S = fitted[i]
#                 length = j - i
#     return S, length


# def detect_drowsiness_pattern(l, fitted):
#     f = []
#     for j in range(1, len(fitted)):
#         results = standard_linear_time_series(fitted, j)
#         S, length = results
#         if length < l - 1:
#             threshold = 0.31
#         else:
#             threshold = 1
#         if dynamic_time_warping(S, fitted[j]) < threshold:
#             f.append(0)
#         else:
#             f.append(1)
#     print(f)
#     plt.plot(f)
#     plt.savefig('results.svg')
#     return f


if __name__ == '__main__':
    with open('../preprocess_data/SWA_data.json', 'r') as f:
        swa_data = json.load(f)

    print("running")
    indexes = {}
    for name in swa_data.keys():
        print(name," get in")
        w_l = 200
        w_step = 200
        windows = sliding_window(swa_data[name], w_l, w_step)
        print('here1')
        m = 2  # number of embedded dimensions
        ap_en = []  # array of ap_en of every window
        for i in range(0, len(windows)):
            ap_en.append(approximate_entropy(m, windows[i]))
        print('here2')
        max_error = 0.2
        result = ap_linear_approximation(ap_en, max_error)
        segments, approximated_array, b_array = result
        print('here3')
        # t_sampling = 60  # unit[seconds]
        # t_window = w_l / 100  # sampling frequency is 100Hz
        # l = t_sampling / t_window
        # spacer = list(range(0, len(approximated_array), 30))
        dwt, f = detect_drowsiness_pattern(approximated_array)
        indexes[name] = [i for i, val in enumerate(f) if val == 1]
        print(name)
        # plt.plot(ap_en)
        # plt.plot(approximated_array)
        # for spa in spacer:
        #     plt.axvline(x=spa)
        # plt.savefig('result_vasanth.svg')
