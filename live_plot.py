import matplotlib.pyplot as plt
from IPython import display

class live_plot(object):
    def __init__(self, xmin=None, xmax=None, xlabel=None, ymin=None, ymax=None, ylabel=None, axes=None, figsize=(18,6)):

        if axes is None:
            self.fig = plt.figure(figsize)
            self.ax = self.fig.add_axes((0,0,1,1))
        else:
            self.ax = axes

        self.xmin = xmin
        self.xmax = xmax
        self.xlabel = xlabel
        self.ymin = ymin
        self.ymax = ymax
        self.ylabel = ylabel

        self.data = {}
        self.kwargs = {}


    def add_plot(self, name, data=[], **kwargs):
        if name in self.data:
            raise Exception('A plot already exist for :' + name)
        self.data[name] = data
        self.kwargs[name] = kwargs
        if not 'label' in kwargs:
            self.kwargs[name]['label'] = 'None supplied'
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

        for k in self.data.keys():
            if len(self.data[k]) > 0 and type(self.data[k][0]) is tuple:
                self.ax.plot(*zip(*self.data[k]), **(self.kwargs[k]))
            else:
                self.ax.plot(self.data[k], **(self.kwargs[k]))

        self.ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        display.display(plt.gcf())
        display.clear_output(wait=True)