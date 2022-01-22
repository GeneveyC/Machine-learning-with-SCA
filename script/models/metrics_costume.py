from tensorflow.keras import backend as bk


# Rank of the correct label among others (0 means top-ranked).
def mean_rank(y_true, y_pred):
    # obtain probas for correct guesses
    probas = y_true * y_pred
    probas = bk.max(probas,1)
    # repeat them to be able to compare to y_pred
    probas = bk.expand_dims(probas)
    nb_classes = bk.int_shape(y_pred)[1]
    probas = bk.concatenate([probas]*nb_classes,1)
    # get better-ranked intermediate variables
    positions = bk.greater_equal(y_pred,probas)
    positions = bk.cast(positions,bk.floatx())
    total = bk.sum(positions,1)
    total = total - 1
    return bk.mean(total)