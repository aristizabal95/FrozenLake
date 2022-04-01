import numpy as np

def argmax(arr: np.ndarray, random_generator=np.random.default_rng()) -> int:
    """Returns the index of the greates value. Ties are broken randomly

    Args:
        arr (np.ndarray): Numpy array to compute argmax on
        random_generator (Generator): Numpy random generator

    Returns:
        int: index of the greatest value in the array
    """
    noise = random_generator.random(arr.shape)
    filter_max = (arr == arr.max()).astype(int)
    return np.argmax(filter_max * noise)