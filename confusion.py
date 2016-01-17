import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class confusion_matrix(object):
    def __init__(self, label_names, ignore_set=[]):
        '''
        label_names --  a list of label names for later displaying
        ignore_set --   a list of indices that should be ignored for actual evaluation.
                        This is needed to show the confusion matrix without preprocessing 
                        the input images.
        '''
        self.label_names_raw = label_names
        self.class_count_raw = len(label_names)

        self.ignore_set=ignore_set
        self.label_names = (np.array(label_names).copy()).tolist()
        for ig in ignore_set:
            self.label_names.remove(label_names[ig])
        self.class_count= len(self.label_names)

        self.confusion = None


    def compute(self, gts, results):
        '''
        Do the actual computation of the confusion matrix itself.
        gts -- a list of ground truth images, these should be 2d images with label indices
        results -- a list of the gt length with the corresponding result images.
        '''
        assert(len(gts)==len(results))

        self.confusion = np.zeros((self.class_count_raw+1, self.class_count_raw))
        for gt, res in zip(gts, results):
            for pos, val in Counter(zip(gt.flatten(), res.flatten())).items():
                self.confusion[pos] += val

        self.confusion = self.confusion[:self.class_count_raw]

        valid_range = range(self.class_count_raw)
        for ig in self.ignore_set:
            valid_range.remove(range(self.class_count_raw)[ig])

        self.confusion_ignored = np.zeros((len(valid_range), len(valid_range)))
        for i_n, i in enumerate(valid_range):
            for j_n,j in enumerate(valid_range):
                self.confusion_ignored[i_n, j_n] = self.confusion[i,j]


    def statistics(self, use_ignore_set=True):
        '''
        Returns the global, average and average intersection over union score.
        use_ignore_set -- set this to false to base the results using all classes.
        '''
        assert(self.confusion is not None)

        if use_ignore_set:
            confusion = self.confusion_ignored
        else:
            confusion = self.confusion
        total = np.sum(confusion)
        self.gt_sum_per_class = np.sum(confusion, 1)
        self.sum_per_class = np.sum(confusion, 0)
        global_score = np.sum(np.diag(confusion))/total
        diag = np.diag(confusion)
        union = self.gt_sum_per_class + self.sum_per_class - diag
        avg = np.nanmean(diag/self.gt_sum_per_class)
        avg_iou = np.nanmean(diag/union)

        return global_score, avg, avg_iou


    def plot(self, use_ignore_set=True, colormap=mpl.cm.Spectral_r):
        '''
        Plots the confusion matrix, once row and once column normalized.
        use_ignore_set -- set this to false to base the results using all classes.
        colormap -- the used colormap
        '''
        global_score, avg, avg_iou = self.statistics(use_ignore_set)
        if use_ignore_set:
            confusion = self.confusion_ignored
            class_count = self.class_count
            label_names = self.label_names
        else:
            confusion = self.confusion
            class_count = self.class_count_raw
            label_names = self.label_names_raw
        confusion_normalized_row = (confusion.copy().T/self.gt_sum_per_class).T
        confusion_normalized_col = confusion.copy()/self.sum_per_class
        fig, ax = plt.subplots(1,2, figsize=(15,5.5), sharey=True, sharex=True)
        ax[0].imshow(confusion_normalized_row, interpolation='nearest', cmap=colormap, vmin=0, vmax=1, aspect='auto')
        im = ax[1].imshow(confusion_normalized_col, interpolation='nearest', cmap=colormap, vmin=0, vmax=1, aspect='auto')
        cax,kw = mpl.colorbar.make_axes([a for a in ax.flat])
        plt.colorbar(im, cax=cax, **kw)
        ax[0].set_yticks(range(class_count))
        ax[0].set_xticks(range(class_count))
        _ = ax[0].set_yticklabels(label_names)
        ax[0].xaxis.tick_top()
        _ = ax[0].set_xticklabels(label_names, rotation='vertical')
        ax[1].xaxis.tick_top()
        _ = ax[1].set_xticklabels(label_names, rotation='vertical')
        _ = ax[0].set_title('row normalized', horizontalalignment='center', y=-0.1)
        _ = ax[1].set_title('column normalized', horizontalalignment='center', y=-0.1)
        _ = fig.suptitle('global:{0:.2%}, average:{1:.2%}, avg_iou:{2:.2%}'.format(global_score, avg, avg_iou), fontsize=14, fontweight='bold', x = 0.4, y = 0.03)
        return fig