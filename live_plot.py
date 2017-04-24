import matplotlib.pyplot as plt
from IPython import display
import numpy as np

class live_plot(object):
    def __init__(self, xmin=None, xmax=None, xlabel=None, ymin=None, ymax=None, ylabel=None, ylog=False, xlog=False, axes=None, figsize=(14,6)):

        if axes is None:
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_axes((0,0,1,1))
        else:
            self.ax = axes

        self.xmin = xmin
        self.xmax = xmax
        self.xlabel = xlabel
        self.ymin = ymin
        self.ymax = ymax
        self.ylabel = ylabel
        self.ylog = ylog
        self.xlog = xlog

        self.data = {}
        self.kwargs = {}
        self.names = []


    def add_plot(self, name, data=[], redraw=True, **kwargs):
        if name in self.data:
            raise Exception('A plot already exist for :' + name)
        self.names.append(name) #To keep the order.
        self.data[name] = data
        self.kwargs[name] = kwargs
        if not 'label' in kwargs:
            self.kwargs[name]['label'] = 'None supplied'
        if redraw:
            self.plot()


    def update_plot(self, name, data=None, redraw=True, **kwargs):
        if not name in self.data:
            raise Exception('Cannot update non-existing plot: '+ name)

        if data is not None:
            self.data[name] = data
        for k, v in kwargs:
            self.kwargs[name][k]= v
        if redraw:
            self.plot()


    def plot(self):
        self.ax.cla()
        self.ax.grid(True)
        if self.xmin is not None and self.xmax is not None:
            self.ax.set_xlim(self.xmin, self.xmax)

        if self.xlabel is not None:
            self.ax.set_xlabel(self.xlabel, fontsize=18)

        if self.ymin is not None and self.ymax is not None:
            self.ax.set_ylim(self.ymin, self.ymax)

        if self.ylabel is not None:
            self.ax.set_ylabel(self.ylabel, fontsize=18)

        if self.xlog:
            self.ax.set_xscale('log')
        if self.ylog:
            self.ax.set_yscale('log')

        todo = []
        for k in self.names:
            if self.ylog:
                if np.sum(self.data[k]) == 0:
                    todo.append(k)
                    continue

            if len(self.data[k]) > 0 and type(self.data[k][0]) is tuple:
                self.ax.plot(*zip(*self.data[k]), **(self.kwargs[k]))
            else:
                self.ax.plot(self.data[k], **(self.kwargs[k]))

        #Can't plot anything since all seem to be all zeros :(
        if len(todo) != len(self.names):
            for k in todo:
                if len(self.data[k]) > 0 and type(self.data[k][0]) is tuple:
                    self.ax.plot(*zip(*self.data[k]), **(self.kwargs[k]))
                else:
                    self.ax.plot(self.data[k], **(self.kwargs[k]))

        self.ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        display.display(plt.gcf())
        display.clear_output(wait=True)
