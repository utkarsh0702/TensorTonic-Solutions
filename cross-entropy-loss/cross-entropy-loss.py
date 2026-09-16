import numpy as np

def cross_entropy_loss(y_true: list[int], y_pred: list[list[float]]) -> float:
    """
    Returns the mean multiclass cross-entropy loss as a Python float.
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_out = y_pred[np.arange(len(y_true)), y_true]
    out = -np.log(y_out)
    return np.mean(out)