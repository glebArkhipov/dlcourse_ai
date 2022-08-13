import numpy as np
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy


def find_k_with_highest_metric(k_to_metric):
    # return max(k_to_f1.keys(), key=(lambda k: k_to_f1[k]))
    max_val = list(k_to_metric.values())[0]
    max_key = list(k_to_metric.keys())[0]
    for k, metric in k_to_metric.items():
        if metric > max_val:
            max_key = k
            max_val = metric
    return max_key


def cross_validate_accuracy(train_x, train_y, num_folds, k_choices):
    return cross_validation(
        train_x, train_y,
        num_folds, k_choices,
        lambda prediction, fold_test_y: multiclass_accuracy(prediction, fold_test_y)
    )


def cross_validate_f1(train_x, train_y, num_folds, k_choices):
    return cross_validation(
        train_x, train_y,
        num_folds, k_choices,
        lambda prediction, fold_test_y: binary_classification_metrics(prediction, fold_test_y)[2]
    )


def cross_validation(train_x, train_y, num_folds, k_choices, metrics_calc):
    train_folds_x = np.array_split(train_x, num_folds)
    train_folds_y = np.array_split(train_y, num_folds)

    k_to_metrics = {}

    for k in k_choices:
        knn_classifier = KNN(k=k)
        metrics = []
        for i in range(num_folds):
            fold_test_x = train_folds_x[i]
            fold_test_y = train_folds_y[i]
            fold_train_x = remove_fold_and_join(train_folds_x, i)
            fold_train_y = remove_fold_and_join(train_folds_y, i)
            knn_classifier.fit(fold_train_x, fold_train_y)
            prediction = knn_classifier.predict(fold_test_x)
            metric = metrics_calc(prediction, fold_test_y)
            metrics.append(metric)
        metric_average = sum(metrics) / len(metrics)
        k_to_metrics[k] = metric_average

    return k_to_metrics


def remove_fold_and_join(folds, index2remove):
    return np.concatenate(
        list(map(
            lambda x: x[1],
            filter(
                lambda x: x[0] != index2remove,
                enumerate(folds)
            )
        ))
    )
