import numpy as np
import cv2

def soft_resize_labels(labels, newsize, valid_threshold, void_label):
    possible_labels = set(np.unique(labels))
    if void_label in possible_labels:
        possible_labels.remove(void_label)
    possible_labels = np.asarray(list(possible_labels))

    label_vol = np.zeros((labels.shape[0], labels.shape[1], len(possible_labels)))
    for i, l in enumerate(possible_labels):
        label_vol[:,:, i] = (labels == l)

    label_vol = cv2.resize(label_vol, newsize)

    # If there is only a single label, then the resize function returns a 2D tensor
    if len(label_vol.shape) == 2:
        label_vol = np.reshape(label_vol, (*label_vol.shape, 1))

    max_idx = np.argmax(label_vol, 2) #The max label using this mapping
    max_val = np.max(label_vol,2) #It's value

    max_idx = possible_labels[max_idx] #Remap to original values
    max_idx[max_val < valid_threshold] = void_label #Set the void label according to threshold.

    return max_idx.astype(labels.dtype)

def mM(array):
    return np.min(array), np.max(array)
