import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split

from random_forest import ground_truth_sep, reform_ground_truth
from sklearn.metrics import mutual_info_score, confusion_matrix
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pyswarms as ps


def corresponding_data(true_window, false_window, data, sample_rate):
    true_data = {}
    false_data = {}

    for name in true_window.keys():
        true_data[name] = []
        for i in range(len(true_window[name])):
            true_data[name].append(data[name][
                                   true_window[name][i][0] * sample_rate:(
                                           true_window[name][i][-1] * sample_rate + sample_rate)])
    for name in false_window.keys():
        false_data[name] = []
        for i in range(len(false_window[name])):
            false_data[name].append(data[name][
                                    false_window[name][i][0] * sample_rate:(
                                            false_window[name][i][-1] * sample_rate + sample_rate)])

    return true_data, false_data


def compute_katz_fractal_dimension(list_data):
    L = np.abs(np.diff(list_data)).sum()  # sum of euler distance
    d = np.max(list_data) - list_data[0]  # max distance from the first point
    n = len(list_data) - 1  # number of points
    result = np.log(n) / (np.log(n) + np.log(d / L + 1e-10))
    return result


def compute_shannon_entropy(data):
    unique_data = np.unique(data)
    proba = [np.sum(data == i) / len(data) for i in unique_data]
    entropy = -np.sum(np.multiply(proba, np.log2(proba)))
    return entropy


def extracting_feature(data):
    range_ = []  # 下划线是为了避免和python的关键字冲突
    std_dev = []
    energy = []
    zero_crossing_rate = []
    first_quartile = []
    second_quartile = []
    third_quartile = []
    katz_fractal_dimension = []
    skewness = []
    kurtosis = []
    sample_entropy = []
    shannon_entropy = []
    frequency_variability = []
    spectral_entropy = []
    spectral_flux = []
    center_gravity_frequency = []
    dominant_frequency = []
    average_psd = []
    for name in data.keys():
        for i in range(len(data[name])):
            range_.append(max(data[name][i]) - min(data[name][i]))
            std_dev.append(np.std(data[name][i]))
            energy.append(np.sum(np.square(data[name][i])))
            zero_crossing_rate.append(np.sum(np.abs(np.diff(np.sign(data[name][i])))) / 2)
            first_quartile.append(np.percentile(data[name][i], 25))
            second_quartile.append(np.percentile(data[name][i], 50))
            third_quartile.append(np.percentile(data[name][i], 75))
            katz_fractal_dimension.append(compute_katz_fractal_dimension(data[name][i]))
            skewness.append(np.mean((data[name][i] - np.mean(data[name][i])) ** 3) / (np.std(data[name][i]) ** 3))
            kurtosis.append(np.mean((data[name][i] - np.mean(data[name][i])) ** 4) / (np.std(data[name][i]) ** 4))
            sample_entropy.append(-np.log(np.sum(np.exp(-np.abs(np.diff(data[name][i])))) / len(data[name][i])))
            shannon_entropy.append(compute_shannon_entropy(data[name][i]))

            fft_data = np.fft.fft(data[name][i])
            fft_data = np.abs(fft_data)

            frequency_variability.append(np.var(fft_data))
            spectral_entropy.append(-np.sum(np.multiply(fft_data, np.log(fft_data))))
            spectral_flux.append(np.sum(np.abs(np.diff(fft_data))))
            center_gravity_frequency.append(np.sum(np.multiply(fft_data, range(len(fft_data)))) / np.sum(fft_data))
            dominant_frequency.append(np.argmax(fft_data))
            average_psd.append(np.mean(fft_data ** 2 / len(fft_data) / 100))

    feature_matrix = np.column_stack((range_, std_dev, energy, zero_crossing_rate, first_quartile, second_quartile,
                                      third_quartile, katz_fractal_dimension, skewness, kurtosis, sample_entropy,
                                      shannon_entropy, frequency_variability, spectral_entropy, spectral_flux,
                                      center_gravity_frequency, dominant_frequency, average_psd))
    return feature_matrix


def fisher_index(feature_matrix, label):
    mean_whole = []
    mean_0 = []  # awake
    mean_1 = []  # drowsy
    std_dev_whole = []
    std_dev_0 = []
    std_dev_1 = []
    zero_index = np.where(label == 0)
    one_index = np.where(label == 1)
    n_0 = len(zero_index)
    n_1 = len(one_index)

    fisher_index_list = []

    for i in range(feature_matrix.shape[1]):
        mean_whole.append(np.mean(feature_matrix[:, i]))
        mean_0.append(np.mean(feature_matrix[zero_index, i]))
        mean_1.append(np.mean(feature_matrix[one_index, i]))
        std_dev_whole.append(np.std(feature_matrix[:, i]))
        std_dev_0.append(np.std(feature_matrix[zero_index, i]))
        std_dev_1.append(np.std(feature_matrix[one_index, i]))

        fisher_index_list.append(
            (n_0 * (mean_0[i] - mean_whole[i]) ** 2 + n_1 * (mean_1[i] - mean_whole[i]) ** 2)
            / (n_0 * (std_dev_0[i] ** 2) + n_1 * (std_dev_1[i] ** 2)))

    return fisher_index_list


def correlation_index(feature_matrix, label):
    correlation_index_list = []
    for i in range(feature_matrix.shape[1]):
        correlation_index_list.append(
            np.cov(feature_matrix[:, i], label)[0, 1] / np.std(feature_matrix[:, i]) / np.std(label))
    return correlation_index_list


def T_test_index(feature_matrix, label):
    mean_0 = []  # awake
    mean_1 = []  # drowsy
    std_dev_0 = []
    std_dev_1 = []
    zero_index = np.where(label == 0)
    one_index = np.where(label == 1)
    n_0 = len(zero_index)
    n_1 = len(one_index)

    T_test_index_list = []

    for i in range(feature_matrix.shape[1]):
        mean_0.append(np.mean(feature_matrix[zero_index, i]))
        mean_1.append(np.mean(feature_matrix[one_index, i]))
        std_dev_0.append(np.std(feature_matrix[zero_index, i]))
        std_dev_1.append(np.std(feature_matrix[one_index, i]))

        T_test_index_list.append(
            np.abs(mean_0[i] - mean_1[i]) / np.sqrt(std_dev_0[i] ** 2 / n_0 + std_dev_1[i] ** 2 / n_1))

    return T_test_index_list


def mutual_information_index(feature_matrix, label):
    mutual_info_score_list = []
    for i in range(feature_matrix.shape[1]):
        mutual_info_score_list.append(mutual_info_score(feature_matrix[:, i], label.astype(float)))
    return mutual_info_score_list


def fuzzy_inference(input, gaussian_params, F_score, R_score, T_score,
                    I_score):  # gaussian_params is a list of 4 lists of 3 lists
    # each list contains 3 lists,each list contains two elements,
    # the first element is the mean, the second element is the standard deviation
    F = ctrl.Antecedent(np.arange(min(input[:, 0]), max(input[:, 0]), 0.001), 'F')
    R = ctrl.Antecedent(np.arange(min(input[:, 1]), max(input[:, 1]), 0.001), 'R')
    T = ctrl.Antecedent(np.arange(min(input[:, 2]), max(input[:, 2]), 0.001), 'T')
    I = ctrl.Antecedent(np.arange(min(input[:, 3]), max(input[:, 3]), 0.001), 'I')

    F['low'] = fuzz.gaussmf(F.universe, gaussian_params[0], gaussian_params[1])
    F['medium'] = fuzz.gaussmf(F.universe, gaussian_params[2], gaussian_params[3])
    F['high'] = fuzz.gaussmf(F.universe, gaussian_params[4], gaussian_params[5])
    R['low'] = fuzz.gaussmf(R.universe, gaussian_params[6], gaussian_params[7])
    R['medium'] = fuzz.gaussmf(R.universe, gaussian_params[8], gaussian_params[9])
    R['high'] = fuzz.gaussmf(R.universe, gaussian_params[10], gaussian_params[11])
    T['low'] = fuzz.gaussmf(T.universe, gaussian_params[12], gaussian_params[13])
    T['medium'] = fuzz.gaussmf(T.universe, gaussian_params[14], gaussian_params[15])
    T['high'] = fuzz.gaussmf(T.universe, gaussian_params[16], gaussian_params[17])
    I['low'] = fuzz.gaussmf(I.universe, gaussian_params[18], gaussian_params[19])
    I['medium'] = fuzz.gaussmf(I.universe, gaussian_params[20], gaussian_params[21])
    I['high'] = fuzz.gaussmf(I.universe, gaussian_params[22], gaussian_params[23])

    # plt.figure()
    # plt.plot(F.universe, F['low'].mf, 'b', linewidth=1.5, label='Low')
    # plt.plot(F.universe, F['medium'].mf, 'g', linewidth=1.5, label='Medium')
    # plt.plot(F.universe, F['high'].mf, 'r', linewidth=1.5, label='High')
    # plt.show()

    count = 0
    low_ID = 0
    middle_ID = 0
    high_ID = 0
    # Generate fuzzy 81 rules
    for termF in F.terms.values():
        for termR in R.terms.values():
            for termT in T.terms.values():
                for termI in I.terms.values():
                    if termF.label == 'low':
                        count += 0
                    elif termF.label == 'medium':
                        count += 1
                    else:
                        count += 2
                    if termR.label == 'low':
                        count += 0
                    elif termR.label == 'medium':
                        count += 1
                    else:
                        count += 2
                    if termT.label == 'low':
                        count += 0
                    elif termT.label == 'medium':
                        count += 1
                    else:
                        count += 2
                    if termI.label == 'low':
                        count += 0
                    elif termI.label == 'medium':
                        count += 1
                    else:
                        count += 2
                    if count < 2.75:
                        low_ID += np.fmin(fuzz.interp_membership(F.universe, F[termF.label].mf, F_score),
                                          np.fmin(fuzz.interp_membership(R.universe, R[termF.label].mf, R_score),
                                                  np.fmin(
                                                      fuzz.interp_membership(T.universe, T[termF.label].mf, T_score),
                                                      fuzz.interp_membership(I.universe, I[termF.label].mf, I_score))))
                    elif count < 5.5:
                        middle_ID += np.fmin(fuzz.interp_membership(F.universe, F[termF.label].mf, F_score),
                                             np.fmin(fuzz.interp_membership(R.universe, R[termF.label].mf, R_score),
                                                     np.fmin(
                                                         fuzz.interp_membership(T.universe, T[termF.label].mf, T_score),
                                                         fuzz.interp_membership(I.universe, I[termF.label].mf,
                                                                                I_score))))
                    else:
                        high_ID += np.fmin(fuzz.interp_membership(F.universe, F[termF.label].mf, F_score),
                                           np.fmin(fuzz.interp_membership(R.universe, R[termF.label].mf, R_score),
                                                   np.fmin(
                                                       fuzz.interp_membership(T.universe, T[termF.label].mf, T_score),
                                                       fuzz.interp_membership(I.universe, I[termF.label].mf,
                                                                              I_score))))
                    count = 0  # #
    # print(low_ID, middle_ID, high_ID)
    if low_ID + middle_ID + high_ID == 0:
        return 0
    else:
        return (low_ID * 0 + middle_ID * 0.5 + high_ID * 1) / (low_ID + middle_ID + high_ID)


def particle_swarm_optimization():
    # Set-up all the hyperparameters
    options = {'c1': 2, 'c2': 2, 'w': 0.95}
    # Call an instance of PSO
    # dimension is gaussian parameters
    lower_bound = np.tile([-np.inf, 0], (1, 12))
    upper_bound = np.tile([np.inf, np.inf], (1, 12))
    init_pos = np.array([[[min(fuzzy_input[:, 0]), 0.17], [np.mean(fuzzy_input[:, 0]), 0.17], [max(fuzzy_input[:, 0]), 0.17]],
                            [[min(fuzzy_input[:, 1]), 0.17], [np.mean(fuzzy_input[:, 1]), 0.17], [max(fuzzy_input[:, 1]), 0.17]],
                            [[min(fuzzy_input[:, 2]), 0.17], [np.mean(fuzzy_input[:, 2]), 0.17], [max(fuzzy_input[:, 2]), 0.17]],
                            [[min(fuzzy_input[:, 3]), 0.17], [np.mean(fuzzy_input[:, 3]), 0.17], [max(fuzzy_input[:, 3]), 0.17]]])
    init_pos = init_pos.reshape(24)
    n_particles = 10
    init_pos = np.tile(init_pos, (n_particles, 1))  # 10 particles
    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=24, options=options,
                                        init_pos=init_pos, bounds=(lower_bound, upper_bound))
    # Perform the optimization
    cost_pso, pos = optimizer.optimize(pso_cost, iters=5)

    return pos


def pso_cost(x):
    # x is a 2D array where each row represents a particle(every row is a set of gaussian parameters)
    print("particle:", x)
    n_particles = x.shape[0]
    j = [pso_cost_per_particle(x[i, :]) for i in range(n_particles)]
    return np.array(j)


def pso_cost_per_particle(gaussian_params):
    # input is 36x4
    use_index = []
    for i in range(fuzzy_input.shape[0]):
        ID = fuzzy_inference(fuzzy_input, gaussian_params, fuzzy_input[i, 0], fuzzy_input[i, 1], fuzzy_input[i, 2],
                             fuzzy_input[i, 3])
        print(ID)
        if ID > 0.5:
            use_index.append(i)
    if gaussian_params[0] <= gaussian_params[2] <= gaussian_params[4] and gaussian_params[6] <= gaussian_params[8] <= \
            gaussian_params[10] and gaussian_params[12] <= gaussian_params[14] <= gaussian_params[16] and \
            gaussian_params[18] <= gaussian_params[20] <= gaussian_params[22]:
        cost = compute_cost(use_index, feature_matrix, label)
    else:
        cost = 1000000

    return cost


def compute_cost(use_index, feature_matrix, label):
    if len(use_index) == 0:
        test_label_pred = np.ones(len(test_label)) * (-1)
    else:
        use_data = feature_matrix[:, use_index]
        clf = svm.SVC(kernel='rbf', gamma=1, C=100)
        clf.fit(use_data, label)
        test_label_pred = clf.predict(test_feature_matrix[:, use_index])
        conf_matrix = confusion_matrix(test_label, test_label_pred)
        print("confusion matrix:", conf_matrix)

    cost = np.sum((test_label - test_label_pred) ** 2)
    print("cost:", cost)

    return cost


if __name__ == '__main__':
    with open('preprocess_data/ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    ground_truth = reform_ground_truth(ground_truth)

    with open('preprocess_data/SWA_data.json', 'r') as f:
        SWA_data = json.load(f)
    SWV_data = {}
    for name in SWA_data.keys():
        delta_displacement = np.diff(SWA_data[name])
        SWV_data[name] = np.gradient(delta_displacement, 0.01)

    window_length = 3  # 3s
    window_step = 1  # 1s
    sample_rate = 100  # 100Hz
    true_window, false_window = ground_truth_sep(ground_truth, window_length, window_step)

    true_data_SWA, false_data_SWA = corresponding_data(true_window, false_window, SWA_data, sample_rate)
    true_data_SWV, false_data_SWV = corresponding_data(true_window, false_window, SWV_data, sample_rate)

    true_feature_matrix = np.column_stack((extracting_feature(true_data_SWA), extracting_feature(true_data_SWV)))

    false_feature_matrix = np.column_stack((extracting_feature(false_data_SWA), extracting_feature(false_data_SWV)))

    # feature_matrix = np.row_stack((true_feature_matrix, false_feature_matrix))
    # label = np.concatenate((np.zeros(len(true_feature_matrix)), np.ones(len(false_feature_matrix))))
    # label = label.astype(int)
    # feature_matrix, test_feature_matrix, label, test_label = train_test_split(feature_matrix,
    #                                                                           label, test_size=0.2,
    #                                                                           random_state=42)

    np.random.seed(105)
    selected_true_cases = false_feature_matrix.shape[0]
    selected_true_feature_indices = np.random.choice(true_feature_matrix.shape[0], selected_true_cases, replace=False)
    selected_true_feature_matrix = true_feature_matrix[selected_true_feature_indices]
    downsampled_feature_matrix = np.concatenate((selected_true_feature_matrix, false_feature_matrix), axis=0)
    downsampled_label = np.concatenate(
        (np.zeros(len(selected_true_feature_matrix)), np.ones(len(false_feature_matrix))))
    downsampled_label = downsampled_label.astype(int)

    feature_matrix, test_feature_matrix, label, test_label = train_test_split(downsampled_feature_matrix,
                                                                              downsampled_label, test_size=0.2,
                                                                              random_state=42)

    fisher_index_list = fisher_index(feature_matrix, label)
    correlation_index_list = correlation_index(feature_matrix, label)
    T_test_index_list = T_test_index(feature_matrix, label)
    mutual_information_index_list = mutual_information_index(feature_matrix, label)
    # fisher_index_list = fisher_index(downsampled_feature_matrix, downsampled_label)
    # correlation_index_list = correlation_index(downsampled_feature_matrix, downsampled_label)
    # T_test_index_list = T_test_index(downsampled_feature_matrix, downsampled_label)
    # mutual_information_index_list = mutual_information_index(downsampled_feature_matrix, downsampled_label)

    fuzzy_input = np.column_stack((fisher_index_list, correlation_index_list,
                                   T_test_index_list, mutual_information_index_list))
    # gaussian_params = np.tile([0, 0.17, 0.3, 0.17, 0.6, 0.17], (1, 4))
    # gaussian_params = gaussian_params.reshape(24)
    # i = 0
    # ID = fuzzy_inference(fuzzy_input, gaussian_params, fuzzy_input[i, 0], fuzzy_input[i, 1], fuzzy_input[i, 2],
    #                      fuzzy_input[i, 3])
    pos = particle_swarm_optimization()

    # here to test the result
    # pos = [0.80753983, 0.41912503, 1.08671787, 0.88493274, 1.63123535, 1.12406503,
    #        0.41439769, 0.74947166, 0.58506228, 0.072914, 1.35887405, 0.82842229,
    #        1.37680377, 0.56362073, 0.47432573, 0.14828735, 1.52673697, 0.67513707,
    #        0.39519975, 0.65690738, 0.99528688, 1.30793362, 1.11056771, 0.64152692]
    #
    # pso_cost_per_particle(pos)
    # compute_cost([3,7,11,16,25,29,34],test_feature_matrix,test_label)
