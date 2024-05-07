import json
import statistics
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, ShuffleSplit, learning_curve
from sklearn.metrics import accuracy_score, precision_recall_curve, confusion_matrix, RocCurveDisplay, auc, make_scorer, \
    f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
from preprocess_data.process_ground_truth import reform_ground_truth


def ground_truth_sep(ground_truth, size, step):
    true_window = {}
    false_window = {}
    for name in ground_truth.keys():
        true_window[name] = []
        false_window[name] = []
        i = 0
        while i < len(ground_truth[name]) - 2 * size:
            if all(x == 1 for x in ground_truth[name][i:i + size]):
                true_window[name].append([i + j for j in range(0, size)])
                i = i + step
            elif all(x == 4 for x in ground_truth[name][i:i + size]):
                false_window[name].append([i + j for j in range(0, size)])
                i = i + step
            else:
                i = i + 1

    # # 遍历true_window
    # for name in true_window:
    #     for i, window in enumerate(true_window[name]):
    #         if len(window) != 30:
    #             print(f"In true_window, the list at position {i} in {name} is not of length 30.")
    #
    # # 遍历false_window
    # for name in false_window:
    #     for i, window in enumerate(false_window[name]):
    #         if len(window) != 30:
    #             print(f"In false_window, the list at position {i} in {name} is not of length 30.")

    return true_window, false_window


def compress_data(data, resolution):
    compressed = {}
    for name in data.keys():
        compressed[name] = {}
        compressed[name]['average'] = []
        compressed[name]['variance'] = []
        compressed[name]['peak'] = []
        for i in range(0, len(data[name]), resolution):
            compressed[name]['average'].append(statistics.mean(data[name][i:i + resolution]))
            compressed[name]['variance'].append(statistics.variance(data[name][i:i + resolution]))
            compressed[name]['peak'].append(max(data[name][i:i + resolution], key=abs))

    return compressed


def feature_matrix(true_window, false_window, compressed_swa, compressed_longitudinal_acc, compressed_lateral_acc,
                   flag):
    true_cases = 0
    false_cases = 0
    b = 1
    # true | true_window['catia'][0](这是对应的秒数，一个数列）compressed_swa['catia']['average'][true_window['catia'][0]]
    matrix = []
    for name in true_window.keys():
        true_cases += len(true_window[name])
        for i in range(0, len(true_window[name])):
            temp = []
            if flag == 1:
                for j in true_window[name][i]:
                    temp.extend(compressed_swa[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_swa[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_swa[name]['peak'][b * j:b * (j + 1)])
            if flag == 2:
                for j in true_window[name][i]:
                    temp.extend(compressed_lateral_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['peak'][b * j:b * (j + 1)])
            if flag == 3:
                for j in true_window[name][i]:
                    temp.extend(compressed_longitudinal_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['peak'][b * j:b * (j + 1)])
            if flag == 4:
                for j in true_window[name][i]:
                    temp.extend(compressed_lateral_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['peak'][b * j:b * (j + 1)])
                for j in true_window[name][i]:
                    temp.extend(compressed_longitudinal_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['peak'][b * j:b * (j + 1)])
            if flag == 5:
                for j in true_window[name][i]:
                    temp.extend(compressed_swa[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_swa[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_swa[name]['peak'][b * j:b * (j + 1)])
                for j in true_window[name][i]:
                    temp.extend(compressed_lateral_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['peak'][b * j:b * (j + 1)])
                for j in true_window[name][i]:
                    temp.extend(compressed_longitudinal_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['peak'][b * j:b * (j + 1)])
            matrix.append(temp)

    for name in false_window.keys():
        false_cases += len(false_window[name])
        for i in range(0, len(false_window[name])):
            temp = []
            if flag == 1:
                for j in false_window[name][i]:
                    temp.extend(compressed_swa[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_swa[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_swa[name]['peak'][b * j:b * (j + 1)])
            if flag == 2:
                for j in false_window[name][i]:
                    temp.extend(compressed_lateral_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['peak'][b * j:b * (j + 1)])
            if flag == 3:
                for j in false_window[name][i]:
                    temp.extend(compressed_longitudinal_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['peak'][b * j:b * (j + 1)])
            if flag == 4:
                for j in false_window[name][i]:
                    temp.extend(compressed_lateral_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['peak'][b * j:b * (j + 1)])
                for j in false_window[name][i]:
                    temp.extend(compressed_longitudinal_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['peak'][b * j:b * (j + 1)])
            if flag == 5:
                for j in false_window[name][i]:
                    temp.extend(compressed_swa[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_swa[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_swa[name]['peak'][b * j:b * (j + 1)])
                for j in false_window[name][i]:
                    temp.extend(compressed_lateral_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_lateral_acc[name]['peak'][b * j:b * (j + 1)])
                for j in false_window[name][i]:
                    temp.extend(compressed_longitudinal_acc[name]['average'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['variance'][b * j:b * (j + 1)])
                    temp.extend(compressed_longitudinal_acc[name]['peak'][b * j:b * (j + 1)])
            matrix.append(temp)
    return matrix, true_cases, false_cases


def plot_learning_curve(X, y, rf_classifier):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # scorer = make_scorer(f1_score, pos_label=1)
    scorer = make_scorer(roc_auc_score)

    # 计算学习曲线
    train_sizes, train_scores, test_scores = learning_curve(rf_classifier, X, y, cv=cv, n_jobs=-1,
                                                            train_sizes=np.linspace(0.5, 1.0, 10),
                                                            scoring=scorer)

    # 计算平均和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # 绘制学习曲线
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # 绘制阴影区域
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # 创建图例和标签
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("roc_auc Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_auc_roc(X, y, cv, rf_classifier, n_splits):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    accuracy = []

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        rf_classifier.fit(X[train], y[train])

        viz = RocCurveDisplay.from_estimator(rf_classifier, X[test], y[test],
                                             name=f"ROC fold {fold}", alpha=0.3, lw=1, ax=ax,
                                             plot_chance_level=(fold == n_splits - 1),
                                             )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        y_test_predict = rf_classifier.predict(X[test])
        conf_matrix = confusion_matrix(y[test], y_test_predict)
        print(f"Confusion Matrix for fold {fold}:\n{conf_matrix}")
        accuracy.append(accuracy_score(y[test], y_test_predict))

    print("mean accuracy is: %0.3f" % (np.average(accuracy)))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    print("Mean ROC (AUC= %0.2f ± %0.2f)" % (mean_auc, std_auc))

    ax.plot(mean_fpr, mean_tpr, color="b", label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), lw=2,
            alpha=0.8, )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color="grey", alpha=0.2, label=r"$\pm$ 1 std. dev.", )

    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           )
    ax.legend(loc="lower right")

    # plt.show()


def plot_precision_recall(rf_classifier, X, y):
    y_predict = rf_classifier.predict(X)
    conf_matrix = confusion_matrix(y, y_predict)

    y_scores = rf_classifier.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_scores)
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, label='Random Forest')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid()
    plt.show()


def error_rate_curve_oob(X, y, n_estimator):
    oob_errors = []

    n_estimators_range = range(10, n_estimator, 10)  # Change the range as needed

    for n_estimators in n_estimators_range:
        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, criterion='entropy', max_features='sqrt',
                                               oob_score=True, max_depth=10, random_state=42)

        rf_classifier.fit(X, y)

        oob_error = 1 - rf_classifier.oob_score_

        oob_errors.append(oob_error)

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, oob_errors, marker='o', linestyle='-')
    plt.title('Error-rate Curve for Random Forest')
    plt.xlabel('Number of Trees')
    plt.ylabel('Out-of-Bag Error')
    plt.grid(True)
    plt.show()


def save_variable(variable, filename):
    with open(filename + '.pickle', 'wb') as f:
        pickle.dump(variable, f)


if __name__ == '__main__':
    with open('preprocess_data/SWA_data.json', 'r') as f:
        SWA_data = json.load(f)

    with open('preprocess_data/lateral_acceleration_data.json', 'r') as f:
        lateral_acceleration_data = json.load(f)

    with open('preprocess_data/longitudinal_acceleration_data.json', 'r') as f:
        longitudinal_acceleration_data = json.load(f)

    with open('preprocess_data/ground_truth.json', 'r') as f:
        ground_truth = json.load(f)

    ground_truth = reform_ground_truth(ground_truth)
    save_variable(ground_truth, "ground_truth")

    size = 30
    resolution = 100  # ,b=1
    flag = 5  # 1 for swa, 2 for lateral_acc, 3 for longitudinal_acc, 4 for all acc, 5 for swa+acc

    true_window, false_window = ground_truth_sep(ground_truth, size, 1)

    compressed_swa = compress_data(SWA_data, resolution)
    compressed_longitudinal_acc = compress_data(longitudinal_acceleration_data, resolution)
    compressed_lateral_acc = compress_data(lateral_acceleration_data, resolution)

    input_matrix, true_cases, false_cases = feature_matrix(true_window, false_window, compressed_swa,
                                                           compressed_longitudinal_acc, compressed_lateral_acc, flag)
    output_matrix = np.concatenate((np.zeros((true_cases, 1)), np.ones((false_cases, 1))), axis=0)
    output_matrix = output_matrix.tolist()
    output_matrix = np.ravel(output_matrix)

    X = np.array(input_matrix)
    y = np.array(output_matrix)
    y = y.astype(int)

    np.random.seed(105)
    selected_true_cases = false_cases
    selected_true_input_indices = np.random.choice(X[:true_cases].shape[0], selected_true_cases, replace=False)
    selected_true_input_matrix = X[selected_true_input_indices]
    downsampled_X = np.concatenate((selected_true_input_matrix, X[true_cases:]), axis=0)
    downsampled_y = np.ravel(np.concatenate((np.zeros((selected_true_cases, 1)), np.ones((false_cases, 1))), axis=0))

    smote = SMOTE(sampling_strategy='minority', k_neighbors=3, random_state=42)
    X_sum, y_sum = smote.fit_resample(X, y)

    rf_classifier = RandomForestClassifier(n_estimators=500, criterion='entropy', oob_score=True,
                                           max_features='sqrt', max_depth=10,
                                           random_state=42)  # number of trees in the forest

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # plot_precision_recall(rf_classifier,X,y)

    # plot_auc_roc(downsampled_X, downsampled_y, cv, rf_classifier, n_splits)
    plot_auc_roc(X_sum, y_sum, cv, rf_classifier, n_splits)

    # plot_learning_curve(X_sum, y_sum, rf_classifier)

    # error_rate_curve_oob(downsampled_X, downsampled_y, 500)
