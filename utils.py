import numpy as np
import cv2
import signal
import threading



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


def mMm(array):
    return np.min(array), np.max(array), np.mean(array)


class Uninterrupt(object):
    '''
    Use as:
    with Uninterrupt() as u:
        while not u.interrupted:
            # train
    I stole this from #from https://github.com/lucasb-eyer/lbtoolbox/blob/master/lbtoolbox/util.py
    Sorry and thank you :p
    '''
    def __init__(self, sig=signal.SIGINT):
        self.sig = sig

    def __enter__(self):
        self.interrupted = False
        self.released = False

        self.orig_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.release()
            self.interrupted = True

        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type_, value, tb):
        self.release()

    def release(self):
        if not self.released:
            signal.signal(self.sig, self.orig_handler)
            self.released = True



class ThreadedFunction(object):
    def __init__(self, f, **kwargs):
        self.f = f
        self.kwargs = kwargs
        self.next_output = None
        self.lock = threading.Lock()
        thread = threading.Thread(target=self._compute_next)
        thread.start()

    def __call__(self):
        #Wait for the lock to be released, this implies self.next != None
        self.lock.acquire()
        output = self.next_output
        self.next_output = None

        thread = threading.Thread(target=self._compute_next)
        thread.start()
        self.lock.release()
        return output

    def _compute_next(self):
        self.lock.acquire()
        self.next_output = self.f(**self.kwargs)
        self.lock.release()
