import numpy as np

def td_value_update(V: list, s: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as V.
    """
    # Write code here
    delta = alpha * (r + gamma*V[s_next] - V[s])
    V[s] = V[s] + delta
    return np.array(V)