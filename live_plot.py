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
        self.properties = {}
        self.labels = {}


    def add_plot(self, name, properties='', data=[], label='None supplied'):
        if name in self.data:
            raise Exception('A plot already exist for :' + name)
        self.data[name] = data
        self.properties[name] = properties
        self.labels[name] = label
        self.plot()


    def update_plot(self, name, data=None, label=None, properties=None, redraw=True):
        if not name in self.data:
            raise Exception('Cannot update non-existing plot: '+ name)

        if data is not None:
            self.data[name] = data
        if label is not None:
            self.labels[name] = label
        if properties is not None:
            self.properties[name] = properties
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
            self.ax.plot(self.data[k], self.properties[k], label=self.labels[k])

        self.ax.legend()

        display.display(plt.gcf())
        display.clear_output(wait=True)