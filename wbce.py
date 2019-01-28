import numpy as np
from keras import backend as K
from functools import partial, update_wrapper


def weighted_binary_crossentropy(indices):
    """A weighted binary crossentropy loss function
    that works for multilabel classification
    """
    # obtain dataset here
    data = full_dataset_here[indices]
    # create a 2 by N array with weights for 0's and 1's
    weights = np.zeros((2, data.shape[1]))
    # calculates weights for each label in a for loop
    for i in range(data.shape[1]):
        weights_n, weights_p = (data.shape[0]/(2 * (data[:,i] == 0).sum())), (data.shape[0]/(2 * (data[:,i] == 1).sum()))
        # weights could be log-dampened to avoid extreme weights for extremly unbalanced data.
        weights[1, i], weights[0, i] = weights_p, weights_n

    # The below is needed to be able to work with keras' model.compile()
    def wrapped_partial(func, *args, **kwargs):
        partial_func = partial(func, *args, **kwargs)
        update_wrapper(partial_func, func)
        return partial_func

    def wrapped_weighted_binary_crossentropy(y_true, y_pred, class_weights):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
	# cross-entropy loss with weighting
        out = -(y_true * K.log(y_pred)*class_weights[1] + (1.0 - y_true) * K.log(1.0 - y_pred)*class_weights[0])

        return K.mean(out, axis=-1)

    return wrapped_partial(wrapped_weighted_binary_crossentropy, class_weights=weights)

# IMPLEMENTATION EXAMPLE:
# ..
# custom_loss = weighted_binary_crossentropy(indexes_for_class_weighting)
# model.compile(optimizer=SGD(), loss=custom_loss, metrics='accuracy')
# ..
