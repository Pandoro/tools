import numpy as np
import cv2
import signal
import multiprocessing
import queue



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
    def __init__(self, function, prefetch_count, **kwargs):
        """Parallelize a function to prefetch results using mutliple processes.

        Args:
            function: Function to be executed in parallel.
            prefetch_count: Number of samples to prefetch.
            kwargs: Keyword args passed to the executed function.
        """
        self.function = function
        self.prefetch_count = prefetch_count
        self.kwargs = kwargs
        self.output_queue = multiprocessing.Queue(maxsize=prefetch_count)
        self.procs = []
        for i in range(self.prefetch_count):
            p = multiprocessing.Process(
                target=ThreadedFunction._compute_next,
                args=(self.function, self.kwargs, self.output_queue))
            p.daemon = True  # To ensure it is killed if the parent dies.
            p.start()
            self.procs.append(p)

    def fill_status(self, normalize=False):
        """Returns the fill status of the underlying queue.

        Args:
            normalize: If set to True, normalize the fill status by the max
                queue size. Defaults to False.

        Returns:
            The possibly normalized fill status of the underlying queue.
        """
        return (self.output_queue.qsize() /
            (self.output_queue.maxsize if normalize else 1))

    def __call__(self):
        """Obtain one of the prefetched results or wait for one.

        Returns:
            The output of the provided function and the given keyword args.
        """
        output = self.output_queue.get(block=True)
        return output

    def __del__(self):
        """Signal the processes to stop and join them."""
        for p in self.procs:
            p.terminate()
            p.join()

    def _compute_next(function, kwargs, output_queue):
        """Helper function to do the actual computation in a non_blockig way."""
        while True:
            output_queue.put(function(**kwargs))
