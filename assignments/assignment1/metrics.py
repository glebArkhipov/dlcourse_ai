def binary_classification_metrics(predictions, realities):
    '''
    Computes metrics for binary classification

    Arguments:
    predictions, np array of bool (num_samples) - model predictions
    realities, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''

    predictions_and_ground_truths = list(zip(predictions, realities))

    true_positive_num = len(list(filter(lambda x: x[0] == True and x[0] == x[1], predictions_and_ground_truths)))
    predicted_positive_num = len(list(filter(lambda x: x == True, predictions)))
    realities_positive_num = len(list(filter(lambda x: x == True, realities)))
    match_num = len(list(filter(lambda x: x[0] == x[1], predictions_and_ground_truths)))

    precision = true_positive_num / predicted_positive_num
    recall = true_positive_num / realities_positive_num
    accuracy = match_num / len(predictions)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(predictions, realities):
    '''
    Computes metrics for multiclass classification

    Arguments:
    predictions, np array of int (num_samples) - model predictions
    realities, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    predictions_and_ground_truths = list(zip(predictions, realities))
    match_num = len(list(filter(lambda x: x[0] == x[1], predictions_and_ground_truths)))

    return match_num / len(predictions)
