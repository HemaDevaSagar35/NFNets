import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    
    for path in os.listdir(data_dir):
        if "data_batch" in path:
            with open(data_dir + '/' + path, 'rb') as f:
                dic = pickle.load(f, encoding = 'bytes')
                x_train.append(dic[b'data'])
                y_train.append(dic[b'labels'])
        elif 'test_batch' in path:
             with open(data_dir + '/' + path, 'rb') as f:
                dic = pickle.load(f, encoding = 'bytes')
                x_test.append(dic[b'data'])
                y_test.append(dic[b'labels'])
                
    return np.concatenate(x_train, axis=0), np.concatenate(y_train, axis=0), np.concatenate(x_test, axis=0), np.concatenate(y_test, axis=0)

    ### END CODE HERE

    #return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    path = os.path.join(data_dir, 'private_test_images_2022.npy')
    x_test = np.load(path)

    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    samples = x_train.shape[0]
    split_index = int(train_ratio*samples)
    
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

