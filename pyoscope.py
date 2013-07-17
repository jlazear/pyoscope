#!/bin/env python

"""
pyoscope.py
jlazear
2013-07-02

Generic oscilloscope-like plotting for visualizing data from
instruments.

Long description

Example:

<example code here>
"""
version = 20130702
releasestatus = 'beta'

import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('wxagg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar
import matplotlib.pyplot as plt
from collections import Iterable
from types import StringTypes
import threading
import time
from readers import DefaultReader


class PyOscopeStatic(object):
    """
    Object for plotting static data sets.

    Requires the file `f` to be specified.

    The `reader` may be a customized file reading class. It should not be
    instantiated before being passed into PyOscopeStatic.

    The `interactive` flag indicates whether or not the built-in matplotlib
    event handler loop should be used. You do want the built-in MPL loop to
    be active if you are using PyOscopeStatic interactively, but it must not
    be active if the plot is to be embedded in an external application, which
    will be running its own event handling loop (the two loops would collide).
    Set to True (default) to use the built-in MPL loop, otherwise False.

    The desired reader may be specified. The remaining arguments are
    passed to the reader. See reader.ReaderInterface for reader
    implementation notes.
    """
    def __init__(self, f, reader=None, interactive=True, toolbar=True,
                 *args, **kwargs):
        # Read in data
        if reader is None:
            self.reader = DefaultReader(f, *args, **kwargs)
        else:
            self.reader = reader(f, *args, **kwargs)

        self.data = self.reader.init_data(*args, **kwargs)

        # `interactive` determines if the MPL event loop is used or a
        # raw figure is made. Set to False if using an external event handler,
        # e.g. if embedding in a separate program.
        self.interactive = interactive

        self.fig = self._create_fig(toolbar=toolbar)
        self.mode = 'none'
        self._plotdict = {'autoscalex': True,  # Autoscale is meaningless in
                          'autoscaley': True,  # Static, but useful in RT
                          'windowsize': None}

    def _create_fig(self, plotsize=(6., 4.), dpi=100, tight=True,
                    toolbar=True):
        # autolayout: Automatically call tight_layout() on newly created figure
        # toolbar: Whether or not to create toolbar attached to plot
        # Use context manager to prevent global settings from changing
        tb = 'toolbar2' if toolbar else 'None'
        rcdict = {'figure.autolayout': bool(tight), 'toolbar': tb}
        with mpl.rc_context(rc=rcdict):
            if self.interactive:
                # pyplot's figure() function creates Figure object and hooks it
                # into MPL event loop
                figname = self.__class__.__name__ + '-' + hex(id(self))
                fig = plt.figure(figname, figsize=plotsize, dpi=dpi)
            else:
                # mpl.figure.Figure is a raw Figure object
                fig = Figure(plotsize, dpi=dpi)

            # Both cases return the raw Figure object
            return fig

    def _create_axes(self, nrows=1, ncols=1, sharex=False, sharey=False,
                     subplot_kw=None, fig=None):
        """
        Create grid of axes.

        Slightly modified version of matplotlib.pyplot.subplots. See
        subplots documentation for most arguments.

        `fig` is the figure object in which the axes objects should be
        created. If it is None, uses self.fig.
        """
        if isinstance(sharex, bool):
            if sharex:
                sharex = "all"
            else:
                sharex = "none"
        if isinstance(sharey, bool):
            if sharey:
                sharey = "all"
            else:
                sharey = "none"
        share_values = ["all", "row", "col", "none"]
        if sharex not in share_values:
            raise ValueError("sharex [%s] must be one of %s" % \
                    (sharex, share_values))
        if sharey not in share_values:
            raise ValueError("sharey [%s] must be one of %s" % \
                    (sharey, share_values))
        if subplot_kw is None:
            subplot_kw = {}

        if fig is None:
            fig = self.fig
        fig.clear()

        # Create empty object array to hold all axes.  It's easiest to make it
        # 1-d so we can just append subplots upon creation, and then
        nplots = nrows*ncols
        axarr = np.empty(nplots, dtype=object)

        # Create first subplot separately, so we can share it if requested
        ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)
        #if sharex:
        #    subplot_kw['sharex'] = ax0
        #if sharey:
        #    subplot_kw['sharey'] = ax0
        axarr[0] = ax0

        r, c = np.mgrid[:nrows, :ncols]
        r = r.flatten() * ncols
        c = c.flatten()
        lookup = {
                "none": np.arange(nplots),
                "all": np.zeros(nplots, dtype=int),
                "row": r,
                "col": c,
                }
        sxs = lookup[sharex]
        sys = lookup[sharey]

        # Note off-by-one counting because add_subplot uses the MATLAB 1-based
        # convention.
        for i in range(1, nplots):
            if sxs[i] == i:
                subplot_kw['sharex'] = None
            else:
                subplot_kw['sharex'] = axarr[sxs[i]]
            if sys[i] == i:
                subplot_kw['sharey'] = None
            else:
                subplot_kw['sharey'] = axarr[sys[i]]
            axarr[i] = fig.add_subplot(nrows, ncols, i + 1, **subplot_kw)

        # returned axis array will be always 2-d, even if nrows=ncols=1
        axarr = axarr.reshape(nrows, ncols)

        # turn off redundant tick labeling
        if sharex in ["col", "all"] and nrows > 1:
        #if sharex and nrows>1:
            # turn off all but the bottom row
            for ax in axarr[:-1, :].flat:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
                ax.xaxis.offsetText.set_visible(False)

        if sharey in ["row", "all"] and ncols > 1:
        #if sharey and ncols>1:
            # turn off all but the first column
            for ax in axarr[:, 1:].flat:
                for label in ax.get_yticklabels():
                    label.set_visible(False)
                ax.yaxis.offsetText.set_visible(False)

        ret = axarr.reshape(nrows, ncols)

        return ret

    def redraw(self):
        if self.interactive:
            self.fig.canvas.draw()

    def plot(self, xs=None, ys=None, splitx=True, splity=True, sharex='col',
             sharey=False, xtrans=None, ytrans=None, legend=False, *args,
             **kwargs):
        """
        Make plot.

        Specified by `xs` and `ys`, which specify which column of data should
        go on each axis. Each of `xs` and/or `ys` may be either a single
        identifier or a list of identifiers.

        Identifiers include the column name as a string (e.g. "col1"), the
        integer column index (eg. 0), or the column data itself in either
        pandas.Series (1-dimensional) or numpy.ndarray (1-dimensional) types.
        Note that if a Series or ndarray object are passed, the plotter will
        use that exact object. Any errors (e.g. mismatched length) are then
        the responsibility of the user. Note that custom Series or ndarray
        objects must be included in a list, or else they will be treated as
        a list of identifiers themselves, i.e. pass in `xs=[myarray]` instead
        of `xs=myarray`.

        If `splitx` is True, then each identifier in `xs` will generate its
        own column of axes, i.e. there will be `len(xs)` columns of axes and
        each axes object will have only 1 x identifier on it. If it is False,
        then lines for each x identifier will be generated on a single plot,
        i.e. there will be only 1 column of axes and each plot in it will
        have at least `len(xs)` lines in it.

        If `splitx` is True, then each identifier in `ys` will generate its
        own row of axes, i.e. there will be `len(ys)` rows of axes and each
        axes object will have only 1 y identifier on it. If it is False, then
        lines for each y identifier will be generated on a single plot, i.e.
        there will only be 1 row of axes and each plot in it will have at
        least `len(ys)` lines in it.

        If both `splitx` and `splity` are True, then there will be
        `len(xs)` columns and `len(ys)` rows of axes and each will have 1
        line.

        If both `splitx` and `splity` are False, then there will be 1 axes
        and it will have `len(xs)*len(ys)` lines.

        If either `xs` or `ys` is None, then plots of the data versus their
        index are created. The index is always on the x axis.

        `xtrans` and `ytrans` are transformation functions for the x and y
        data, respectively. Their structure must match the structure of
        `xs` and `ys`.

        `legend` indicates whether to show the legend and where it should be
        shown, if not False. If False, no legends are made. If True, the
        legend is plotted in the top right. A string value may be passed to
        `legend` indicating where it should be plotted, matching the
        pyplot.legend()'s `loc` keyword argument. The legend location is
        shared for all axes.
        """
        # Argument checking
        # oneD flag indicates if x axis will be indices
        if (xs is None) and (ys is None):
            raise ValueError("Must specify at least one of xs or ys.")
        elif (ys is None):
            ys = xs
            ytrans = xtrans
            oneD = True
        elif (xs is None):
            oneD = True
        else:
            oneD = False

        if ((not isinstance(xs, Iterable))
            or isinstance(xs, StringTypes)):
            xs = [xs]

        if xtrans is None:
            xtrans = [None]*len(xs)
        elif not isinstance(xtrans, Iterable):
            xtrans = [xtrans]*len(xs)

        if ((not isinstance(ys, Iterable))
            or isinstance(ys, StringTypes)):
            ys = [ys]

        if ytrans is None:
            ytrans = [None]*len(ys)
        elif not isinstance(ytrans, Iterable):
            ytrans = [ytrans]*len(ys)

        if not legend:
            legendflag = False
            legendloc = None
        else:
            legendflag = True
            if isinstance(legend, StringTypes):
                legendloc = legend
            else:
                legendloc = None

        # Find data series to be plotted
        if not oneD:
            newxs = []
            xnames = []
            for i, x in enumerate(xs):
                if isinstance(x, StringTypes):
                    newx = self.data[x]
                    xname = x
                elif isinstance(x, int):
                    xname = self.data.columns[x]
                    newx = self.data[xname]
                elif isinstance(x, Iterable):
                    newx = x
                    xname = 'x_{i}'.format(i=i)
                xnames.append(xname)
                newxs.append(newx)
        else:
            newxs = None
            xnames = None

        newys = []
        ynames = []
        for j, y in enumerate(ys):
            if isinstance(y, StringTypes):
                newy = self.data[y]
                yname = y
            elif isinstance(y, int):
                yname = self.data.columns[y]
                newy = self.data[yname]
            elif isinstance(y, Iterable):
                newy = y
                yname = 'y_{j}'.format(j=j)
            ynames.append(yname)
            newys.append(newy)

        # Store these so we don't have to look them up again
        self._plotdict['xs'] = newxs
        self._plotdict['ys'] = newys
        self._plotdict['xnames'] = xnames
        self._plotdict['ynames'] = ynames
        self._plotdict['xtrans'] = xtrans
        self._plotdict['ytrans'] = ytrans
        self._plotdict['oneD'] = oneD
        self._plotdict['legendflag'] = legendflag
        self._plotdict['legendloc'] = legendloc

        # Create axes for plotting
        if not oneD:
            lenx = len(xs) if splitx else 1
        else:
            lenx = 1
        leny = len(ys) if splity else 1
        self.axes = self._create_axes(leny, lenx, sharex=sharex,
                                      sharey=sharey)

        # Make plots in appropriate axes
        self.mode = 'plot'
        self.lines = np.empty([lenx, leny], dtype='object')
        if oneD:
            for j, y in enumerate(newys):
                yname = ynames[j]
                ytran = ytrans[j]
                rownum = j if splity else 0
                ax = self.axes[rownum, 0]
                line = self._plotyt(ax, y, yname, transform=ytran, *args,
                                    **kwargs)
                self.lines[0, j] = line
                if legendflag:
                    ax.legend(loc=legendloc)
        else:
            for i, x in enumerate(newxs):
                for j, y in enumerate(newys):
                    xname = xnames[i]
                    yname = ynames[j]
                    xtran = xtrans[i]
                    ytran = ytrans[j]
                    rownum = j if splity else 0
                    colnum = i if splitx else 0
                    ax = self.axes[rownum, colnum]
                    line = self._plotxy(ax, x, y, xname, yname, xtrans=xtran,
                                        ytrans=ytran, *args, **kwargs)
                    self.lines[i, j] = line
                    if legendflag:
                        ax.legend(loc=legendloc)

        return self.lines

    def _plotyt(self, ax, y, yname, windowsize=None, transform=None,
                *args, **kwargs):
        """
        Plot a data set versus its indices on the specified axes.

        `windowsize` specifies the number of data points to plot. Counts
        from the end. None (default) specifies all. `windowsize`s that
        exceed the length of the data set will use the full data set.

        `transform` is a function that is applied to the data set before
        it is plotted. It must be a vectorized function, i.e. it must accept
        as its only argument the full data set and return the full transformed
        data set.

        The remaining arguments are passed to `ax.plot`.

        Returns the line object that is created.
        """
        if windowsize is None:
            ws = len(y)
        else:
            ws = min(len(y), windowsize)
        self._plotdict['windowsize'] = windowsize

        if transform is None:
            transform = lambda x: x  # Identity function

        y = y[-ws:]
        y = transform(y)
        line, = ax.plot(y, label=yname, *args, **kwargs)
        return line

    def _plotxy(self, ax, x, y, xname, yname, windowsize=None, xtrans=None,
                ytrans=None, *args, **kwargs):
        """
        Plot two data sets against each other on the specified axes.

        `windowsize` specifies the number of data points to plot. Counts
        from the end. None (default) specifies all. `windowsize`s that
        exceed the length of the data set will use the full data set.

        `xtrans` and `ytrans` are functions that are applied to the data sets
        before they are plotted. They must be vectorized functions, i.e. each
        must accept as its only argument a full data set and return the full
        transformed data set. `xtrans` modifies the x data set and `ytrans`
        modifies the y data set.

        The remaining arguments are passed to `ax.plot`.

        Returns the line object that is created.
        """
        if len(x) != len(y):
            raise ValueError("x and y values must have same length!")

        if windowsize is None:
            ws = len(y)
        else:
            ws = min(len(y), windowsize)
        self._plotdict['windowsize'] = windowsize

        if xtrans is None:
            xtrans = lambda x: x  # Identity function
        if ytrans is None:
            ytrans = lambda x: x  # Identity function

        x = x[-ws:]
        x = xtrans(x)
        y = y[-ws:]
        y = ytrans(y)

        label = "{x} (x) vs {y} (y)".format(x=xname, y=yname)
        line, = ax.plot(x, y, label=label, *args, **kwargs)
        return line

    def clear(self):
        """
        Clears current plot.
        """
        self.fig.clear()
        self.mode = 'none'


class PyOscopeRealtime(PyOscopeStatic, threading.Thread):
    """
    Object for plotting realtime data sets.

    Requires the file `f` to be specified.

    The `reader` may be a customized file reading class. It should not be
    instantiated before being passed into PyOscopeRealtime.

    The `interactive` flag indicates whether or not the built-in matplotlib
    event handler loop should be used. You do want the built-in MPL loop to
    be active if you are using PyOscopeRealtime interactively, but it must not
    be active if the plot is to be embedded in an external application, which
    will be running its own event handling loop (the two loops would collide).
    Set to True (default) to use the built-in MPL loop, otherwise False.

    The desired reader may be specified. The remaining arguments are
    passed to the reader. See reader.ReaderInterface for reader
    implementation notes.
    """
    def __init__(self, f, reader=None, interactive=True, *args, **kwargs):
        PyOscopeStatic.__init__(self, f=f, reader=reader,
                                interactive=interactive, toolbar=False,
                                *args, **kwargs)

        self._update_dict = {'none': self._pass,
                             'plot': self._update_plot}

        # Need to keep track of the backend, since not all backends support
        # all update schemes
        self._backend = plt.get_backend().lower()

        # Create a threading.Event to control the thread's existence
        self.stopped = threading.Event()  # Events default to False
        self.stopped.set()

        threading.Thread.__init__(self)
        if self.interactive:
            # Start the thread

            self.stopped.clear()
            try:
                self.start()
            except RuntimeError:
                pass

    def __del__(self):
        self.stop()

    def stop(self):
        if self.interactive:
            if self.isAlive():
                self.stopped.set()
                self.join()
        self.close()

    def close(self):
        plt.close(self.fig)
        self.reader.close()

    def run(self):
        while not self.stopped.isSet():
            try:
                self._update()
                time.sleep(0.5)
            except Exception as e:
                self.stopped.set()
                raise e
        self.close()

    def _update(self):
        self.data = self.reader.update_data()
        self._update_dict[self.mode]()

    @staticmethod
    def _pass():
        pass

    def _update_plot(self):
        update_backend = {'macosx': self._update_plot_slow,
                          'wxagg': self._update_plot_wxagg}

        update_backend[self._backend]()

    def _update_plot_slow(self):
        """
        Slowest and most platform-independent update step. Don't expect more
        than a few fps out of this method!
        """
        oneD = self._plotdict['oneD']
        xs = self._plotdict['xs']
        ys = self._plotdict['ys']
        xtrans = self._plotdict['xtrans']
        ytrans = self._plotdict['ytrans']

        if oneD:
            for j, y in enumerate(ys):
                ytran = ytrans[j]
                line = self.lines[0, j]
                self._update_line_slow(line, y=y, ytrans=ytran)
        else:
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    xtran = xtrans[i]
                    ytran = ytrans[j]
                    line = self.lines[i, j]
                    self._update_line_slow(line, x, y, xtran, ytran)

        self.autoscale_axes()
        self.fig.canvas.draw()

    def _update_line_slow(self, line, x=None, y=None,
                          xtrans=None, ytrans=None):
        """
        Updates specified line with new data.
        """
        oneD = self._plotdict['oneD']
        windowsize = self._plotdict['windowsize']
        if windowsize is None:
            ws = len(y)
        else:
            ws = min(len(y), windowsize)

        if oneD:
            newx = range(len(y))
        else:
            newx = xtrans(x)
        newx = newx[-ws:]
        newy = ytrans(y)[-ws:]

        line.set_xdata(newx)
        line.set_ydata(newy)

    def _update_plot_wxagg(self):
        self._update_plot_slow() #DELME

    def autoscale_axes(self):
        xflag = self._plotdict['autoscalex']
        yflag = self._plotdict['autoscaley']

        if (not xflag) and (not yflag):
            return

        for ax in self.axes.flatten():
            xminax, xmaxax, yminax, ymaxax = ax.axis()
            dxax = xmaxax - xminax
            dyax = ymaxax - yminax
            xmidax = xminax + dxax/2.
            ymidax = yminax + dyax/2.

            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []

            for line in ax.lines:
                xmin, xmax, ymin, ymax = self._get_minmax(line)
                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)

            xmin = min(xmins)
            xmax = max(xmaxs)
            ymin = min(ymins)
            ymax = max(ymaxs)

            xmincond = (xmin < xminax) or (xmin > xmidax)
            xmaxcond = (xmax > xmaxax) or (xmax < xmidax)
            ymincond = (ymin < yminax) or (ymin > ymidax)
            ymaxcond = (ymax > ymaxax) or (ymax < ymidax)
            cond = xmincond or xmaxcond or ymincond or ymaxcond
            if cond:
                dx = xmax - xmin
                dy = ymax - ymin
                newxmin = xmin - 0.1*dx
                newxmax = xmax + 0.1*dx
                newymin = ymin - 0.1*dx
                newymax = ymax + 0.1*dx

                if xflag:
                    ax.set_xlim([newxmin, newxmax])
                if yflag:
                    ax.set_ylim([newymin, newymax])


    @staticmethod
    def _get_minmax(line):
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        xmin = min(xdata)
        xmax = max(xdata)
        ymin = min(ydata)
        ymax = max(ydata)
        return xmin, xmax, ymin, ymax

    def autoscale(self, xflag=True, yflag=None):
        """
        Whether or not to autoscale the x or y axis.

        If only `xflag` is specified, then applies to both x and y axes.

        If both `xflag` and `yflag` are specified, then `xflag` sets the
        x axis autoscaling and `yflag` sets the y axis autoscaling.
        """
        if yflag is None:
            yflag = xflag
        self._plotdict['autoscalex'] = bool(xflag)
        self._plotdict['autoscaley'] = bool(yflag)