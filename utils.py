import numpy as np
import matplotlib.pyplot as plt

def set_seeds(seed):
    """
    Sets the random seeds for reproducibility.

    Parameters:
        - seed : int
            The seed to set for reproducibility.
    """

    np.random.seed(seed)