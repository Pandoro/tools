import numpy as np
import cv2
import signal
import multiprocessing
import queue


def soft_resize_labels(labels, new_size, valid_threshold, void_label=-1):
    """Perform a soft resizing of a label image.

    This is achieved by first creating a 3D array of size `(height, width,
    label_count)` which is then resized using OpenCV. Since all label channels
    are resized separately, no mixing of labels is performed and discrete label
    values can be retrieved afterwards.

    Args:
        label: A 2D Numpy array containing labels.
        new_size: The target size, specified as `(Width, Height)` TODO: check!
        valid_threshold: The fraction of the dominant label within a group,
            needed to be set. If it falls below this fraction, the `void_label`
            is set instead.
        void_label: The actual label value of the void label. Defaults to -1.

    Returns:
        A resized version of the label image. Using interpolation, but returning
        only valid labels.

    """
    possible_labels = set(np.unique(labels))
    if void_label in possible_labels:
        possible_labels.remove(void_label)
    possible_labels = np.asarray(list(possible_labels))

    label_vol = np.zeros(
        (labels.shape[0], labels.shape[1], len(possible_labels)))
    for i, l in enumerate(possible_labels):
        label_vol[:,:, i] = (labels == l)

    label_vol = cv2.resize(label_vol, new_size)

    # If there is only a single label, then the resize function returns a 2D
    # tensor.
    if len(label_vol.shape) == 2:
        label_vol = np.reshape(label_vol, (*label_vol.shape, 1))

    # Fin the max label using this mapping and the actual label value
    max_idx = np.argmax(label_vol, 2)
    max_val = np.max(label_vol,2)

    # Remap to original values
    max_idx = possible_labels[max_idx]
    # Set the void label according to threshold.
    max_idx[max_val < valid_threshold] = void_label

    return max_idx.astype(labels.dtype)


def mM(array):
    """Computes the min and max for a given array.

    Args:
        array: Input array for which to compute the min and max. All types for
            which numpy can compute the min/max are supported.

    Returns:
        A tuple containing the min and max for the input array.
    """
    return np.min(array), np.max(array)


def mMm(array):
    """Computes the min, max and mean for a given array.

    Args:
        array: Input array for which to compute the min, max and mean. All types
            for which numpy can compute the min/max/mean are supported.

    Returns:
        A tuple containing the min, max and mean for the input array.
    """
    return np.min(array), np.max(array), np.mean(array)


class Uninterrupt(object):
    """
    Use as:
    with Uninterrupt() as u:
        while not u.interrupted:
            # train
    I stole this from:
    https://github.com/lucasb-eyer/lbtoolbox/blob/master/lbtoolbox/util.py
    Sorry and thank you :p
    """
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
        """Helper function to do the actual computation in a non_blockig way.
        Since this will always run in a new process, we ignore the interrupt
        signal for the processes. This should be handled by the parent process
        which kills the children when the object is deleted.
        Some more discussion can be found here:
        https://stackoverflow.com/questions/1408356/keyboard-interrupts-with-pythons-multiprocessing-pool
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            output_queue.put(function(**kwargs))
