import numpy as np

def downscale_labels(factor, labels, threshold):
    if factor == 1: #TODO unclean :(
        return labels
    m = np.min(labels)
    M = np.max(labels)
    if m < -1:
        raise Exception('Labels should not have values below -1')
    h,w = labels.shape
    h_n = int(np.ceil(float(h)/factor))
    w_n = int(np.ceil(float(w)/factor))
    label_sums = np.zeros((h_n, w_n, M+2))
    for y in xrange(0, h):
        for x in xrange(0, w):
            label_sums[y/factor, x/factor, labels[y,x]] +=1

    hit_counts = np.sum(label_sums,2)

    label_sums = label_sums[:,:,:-1]
    new_labels = np.argsort(label_sums, 2)[:,:,-1].astype(np.int32)
    counts = label_sums.reshape(h_n*w_n, M+1)
    counts = counts[np.arange(h_n*w_n),new_labels.flat]
    counts = counts.reshape((h_n, w_n))

    hit_counts *=threshold
    new_labels[counts < hit_counts] = -1
    return new_labels

def mM(array):
    return np.min(array), np.max(array)