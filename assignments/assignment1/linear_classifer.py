import numpy as np


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    """
    minimized_preds = predictions - np.max(predictions, axis=-1, keepdims=True)
    predictions_exp = np.power(np.e, minimized_preds)
    sum_predictions_exp = np.sum(predictions_exp, axis=-1, keepdims=True)
    probs = predictions_exp / sum_predictions_exp
    return probs


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    """
    if type(target_index) != int:
        cross_entropies = np.zeros(probs.shape[0])
        for i in range(probs.shape[0]):
            prob_of_true_class = probs[i, target_index[i]]
            cross_entropy = -1 * np.log(prob_of_true_class)
            cross_entropies[i] = cross_entropy
        return np.mean(cross_entropies)
    else:
        prob_of_true_class = probs[target_index]
        cross_entropy = -1 * np.log(prob_of_true_class)
        return cross_entropy


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    # ∂L/∂zi = probs[i] - real_probs[i]
    # gradient of loss - vector showing how to make loss greater
    # gradient of loss - vector where elements are derivatives of each prediction
    # if i == target_index, then real_probs[i] = 0, so ∂L/∂zi = probs[i]
    # if i != target_index, then real_probs[i] = 1, so ∂L/∂zi = probs[i] -1
    dprediction = probs.copy()
    if type(target_index) != int:
        for i in range(dprediction.shape[0]):
            dprediction[i][target_index[i]] -= 1
        dprediction /= probs.shape[0]
    else:
        dprediction[target_index] -= 1
    return loss, dprediction


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """

    loss = reg_strength * np.sum(W ** 2)
    grad = 2.0 * reg_strength * W

    return loss, grad


def linear_softmax(X, W, target_index):
    """
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    """
    predictions = np.dot(X, W)

    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    # dL/dW = dL/dpred * dpred/dW
    # dpred/dW = X
    # С = A x B
    # dC/dA = B
    dW = np.dot(X.T, dprediction)

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5, epochs=1):
        """
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        """

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            loss = 0
            for batch in batches_indices:
                batch_X = X[batch]
                batch_y = y[batch]
                loss_batch, dW = linear_softmax(batch_X, self.W, batch_y)
                l2_loss, dW_by_l2_loss = l2_regularization(self.W, reg)
                loss_batch = loss_batch + l2_loss
                dW = dW + dW_by_l2_loss
                self.W = self.W - learning_rate * dW
                loss = loss_batch

            loss_history.append(loss)
            # print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        preds = X.dot(self.W)
        probs = softmax(preds)
        y_pred = np.argmax(probs, axis=1)
        return y_pred
