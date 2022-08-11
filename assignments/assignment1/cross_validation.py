import numpy as np
from knn import KNN
from metrics import binary_classification_metrics


def find_k_with_highest_f1(k_to_f1):
    return max(k_to_f1.keys(), key=(lambda k: k_to_f1[k]))


def cross_validate(train_x, train_y, num_folds, k_choices):
    train_folds_x = np.array_split(train_x, num_folds)
    train_folds_y = np.array_split(train_y, num_folds)

    k_to_f1 = {}

    for k in k_choices:
        knn_classifier = KNN(k=k)
        f1s = []
        for i in range(num_folds):
            fold_test_x = train_folds_x[i]
            fold_test_y = train_folds_y[i]
            fold_train_x = remove_fold_and_join(train_folds_x, i)
            fold_train_y = remove_fold_and_join(train_folds_y, i)
            knn_classifier.fit(fold_train_x, fold_train_y)
            prediction = knn_classifier.predict(fold_test_x)
            _, _, f1, _ = binary_classification_metrics(prediction, fold_test_y)
            f1s.append(f1)
        f1_average = sum(f1s)/len(f1s)
        k_to_f1[k] = f1_average

    return k_to_f1


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
