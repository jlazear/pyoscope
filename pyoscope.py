#!/bin/env python

"""
pyoscope.py
jlazear
2013-07-17

Generic oscilloscope-like plotting for visualizing data from
instruments.

See PyOscope class.

Example:

    rt = PyOscope(f='testdata.txt')
    rt.data.columns
    rt.plot('second', ['first', 'third'], legend=True)
"""
version = 20130717
releasestatus = 'dev'

import numpy as np
import matplotlib as mpl
# mpl.use('wxagg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from collections import Iterable
from types import StringTypes, MethodType, NoneType
import threading
from functools import wraps
from readers import DefaultReader


__all__ = ['PyOscope', 'PyOscopeStatic', 'PyOscopeRealtime']


def synchronized(lockname):
    """
    A decorator to place an instance based lock around a method.

    Argument must be string name of lock to be used, e.g.

    class A(object):
        def __init__(self):
            lock = threading.RLock()

        @synchronized('lock')
        def myfunc(self, a):
            return a**2
    """
    def _synched(func):
        @wraps(func)
        def _synchronizer(self, *args, **kwargs):
            tlock = self.__getattribute__(lockname)
            with tlock:
                return func(self, *args, **kwargs)
        return _synchronizer
    return _synched


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

    Example usage:

        >>> pos = PyOscopeStatic(f='testdata.txt')
        >>> pos.data.columns
        >>> pos.plot('second', ['first', 'third'], legend=True)
    """
    def __init__(self, f=None, reader=None, interactive=True, toolbar=True,
                 *args, **kwargs):
        # Read in data
        if f is not None:
            if reader is None:
                self.reader = DefaultReader(f, *args, **kwargs)
            else:
                self.reader = reader(f, *args, **kwargs)

            self.data = self.reader.init_data(*args, **kwargs)
            self._initialized = True
        else:
            self.reader = None
            self._initialized = False

        # `interactive` determines if the MPL event loop is used or a
        # raw figure is made. Set to False if using an external event handler,
        # e.g. if embedding in a separate program.
        self.interactive = interactive

        # Static version is not threaded, but want to make sure any subclasses
        # are thread-safe
        self.lock = threading.RLock()

        # Need to keep track of the backend, since not all backends support
        # all update schemes
        self._backend = plt.get_backend().lower()

        self.fig = self._create_fig(toolbar=toolbar)
        self.axes = None
        self.canvas = self.fig.canvas
        self.mode = 'none'
        self._plotdict = {'autoscalex': True,  # Autoscale is meaningless in
                          'autoscaley': True,  # Static, but useful in RT
                          'windowsize': None}

    @synchronized('lock')
    def switch_file(self, newfile, reader=None, *args, **kwargs):
        """
        Switch the file that is used for plotting.

        If no reader is specified, attempts to use pre-existing reader.

        If there is no pre-existing reader, creates a DefaultReader and uses
        that.
        """
        self.clear()
        if reader is not None:
            try:
                self.reader.close()
            except AttributeError:
                pass
            self.reader = reader(newfile, *args, **kwargs)

        if not self.reader:
            self.reader = DefaultReader(newfile, *args, **kwargs)

        self.data = self.reader.switch_file(newfile, *args, **kwargs)
        self._initialized = True
        try:
            return self._plot_from_dict()
        except ValueError:
            self.redraw()
            return

    @synchronized('lock')
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

    @synchronized('lock')
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
            raise ValueError("sharex [%s] must be one of %s" %
                             (sharex, share_values))
        if sharey not in share_values:
            raise ValueError("sharey [%s] must be one of %s" %
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
        lookup = {"none": np.arange(nplots),
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

    @synchronized('lock')
    def redraw(self):
        if self.interactive:
            self.canvas.draw_idle()

    @synchronized('lock')
    def plot(self, xs=None, ys=None, splitx=True, splity=True, sharex='col',
             sharey=False, xtrans=None, ytrans=None, legend=False,
             xlabels=None, ylabels=None, labels=None, *args, **kwargs):
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
        index are created. The index is always on the x axis. If both `xs`
        and `ys` are None, then all of the possible data sets are plotted
        against their indices.

        `xtrans` and `ytrans` are transformation functions for the x and y
        data, respectively. Their structure must match the structure of
        `xs` and `ys`.

        `legend` indicates whether to show the legend and where it should be
        shown, if not False. If False, no legends are made. If True, the
        legend is plotted in the top right. A string value may be passed to
        `legend` indicating where it should be plotted, matching the
        pyplot.legend()'s `loc` keyword argument. The legend location is
        shared for all axes.

        `xlabels` and `ylabels` are display-only labels for the `xs` and `ys`.
        They are displayed in place of the file-defined labels in legends.
        They must match the shape of their corresponding data identifiers,
        `xs` or `ys`. If set to None, then the column labels pulled from the
        file or assigned by default are used.

        `labels` are display-only labels that override the programmatically
        generated labels. No pre-formatting is done on this argument, so it
        must be specified correctly. For 1D arrays of plots (i.e. only `xs` or
        `ys` is specified), then `labels` must be a 1D list of matching
        length. For 2D arrays of plots (i.e. both `xs` and `ys` are
        specified), then `labels` must be a 2D array of matching length, where
        `labels[i, j]` corresponds to (`xs[i]`, `ys[j]`).
        """
        if not self._initialized:
            return

        # Argument checking
        # oneD flag indicates if x axis will be indices
        if (xs is None) and (ys is None):
            xs = [name for name in self.data.columns]
        if (ys is None):
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
        if ((not isinstance(xlabels, Iterable))
            or isinstance(xlabels, StringTypes)):
            xlabels = [xlabels]

        if xtrans is None:
            xtrans = [None]*len(xs)
        elif not isinstance(xtrans, Iterable):
            xtrans = [xtrans]*len(xs)

        if ((not isinstance(ys, Iterable))
            or isinstance(ys, StringTypes)):
            ys = [ys]
        if ((not isinstance(ylabels, Iterable))
            or isinstance(ylabels, StringTypes)):
            ylabels = [ylabels]

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
                elif isinstance(x, NoneType):
                    xname = None
                    temp = self.data.columns[0]
                    newx = range(len(self.data[temp]))
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
            elif isinstance(y, NoneType):
                yname = None
                temp = self.data.columns[0]
                newy = range(len(self.data[temp]))
            ynames.append(yname)
            newys.append(newy)

        # Store these so we don't have to look them up again
        self._plotdict['xs'] = newxs
        self._plotdict['ys'] = newys
        self._plotdict['xnames'] = xnames
        self._plotdict['ynames'] = ynames
        self._plotdict['xlabels'] = xlabels
        self._plotdict['ylabels'] = ylabels
        self._plotdict['xtrans'] = xtrans
        self._plotdict['ytrans'] = ytrans
        self._plotdict['label'] = labels
        self._plotdict['splitx'] = splitx
        self._plotdict['splity'] = splity
        self._plotdict['sharex'] = sharex
        self._plotdict['sharey'] = sharey
        self._plotdict['oneD'] = oneD
        self._plotdict['legendflag'] = legendflag
        self._plotdict['legendloc'] = legendloc

        # Abort if nothing to plot along either axis
        ly = len(ynames)
        lx = 1 if isinstance(xnames, type(None)) else len(xnames)
        if lx*ly == 0:
            self.clear()
            self.lines = []
            if self.interactive:
                self.fig.show()
                self.redraw()
            return self.lines

        # Create axes for plotting
        if not oneD:
            numxs = len(xs)
            lenx = numxs if splitx else 1
        else:
            numxs = 1
            lenx = 1
        numys = len(ys)
        leny = numys if splity else 1
        self.axes = self._create_axes(leny, lenx, sharex=sharex,
                                      sharey=sharey)

        # Make plots in appropriate axes
        self.mode = 'plot'
        self.lines = np.empty([numxs, numys], dtype='object')
        if oneD:
            for j, y in enumerate(newys):
                yname = ynames[j]
                ylbl = ylabels[j]
                ylbl = yname if (ylbl is None) else ylbl
                ytran = ytrans[j]
                label = None if (labels is None) else labels[j]
                rownum = j if splity else 0
                ax = self.axes[rownum, 0]
                line = self._plotyt(ax, y, ylbl, transform=ytran,
                                    label=label, *args, **kwargs)
                self.lines[0, j] = line
                if legendflag:
                    ax.legend(loc=legendloc)
        else:
            for i, x in enumerate(newxs):
                for j, y in enumerate(newys):
                    xname = xnames[i]
                    xlbl = xlabels[i]
                    xlbl = xname if (xlbl is None) else xlbl
                    yname = ynames[j]
                    ylbl = ylabels[j]
                    ylbl = yname if (ylbl is None) else ylbl
                    xtran = xtrans[i]
                    ytran = ytrans[j]
                    label = None if (labels is None) else labels[i][j]
                    rownum = j if splity else 0
                    colnum = i if splitx else 0
                    ax = self.axes[rownum, colnum]
                    line = self._plotxy(ax, x, y, xlbl, ylbl, xtrans=xtran,
                                        ytrans=ytran, label=label,
                                        *args, **kwargs)
                    self.lines[i, j] = line
                    if legendflag:
                        ax.legend(loc=legendloc)

        if self.interactive:
            self.fig.show()
            self.redraw()

        return self.lines

    @synchronized('lock')
    def _plot_from_dict(self, pdict=None):
        if not self._initialized:
            return

        if pdict is None:
            pdict = self._plotdict

        try:
            xnames = pdict['xnames']
            ynames = pdict['ynames']
            xlabels = pdict['xlabels']
            ylabels = pdict['ylabels']
            labels = pdict['labels']
            legendflag = pdict['legendflag']
            legendloc = pdict['legendloc']
            splitx = pdict['splitx']
            splity = pdict['splity']
            sharex = pdict['sharex']
            sharey = pdict['sharey']
            xtrans = pdict['xtrans']
            ytrans = pdict['ytrans']
        except KeyError:
            raise ValueError("Invalid plot dictionary specified:"
                             " {0}".format(repr(pdict)))

        nameflag = True
        names = xnames + ynames if xnames else ynames
        for name in names:
            flag = (name in self.data.columns)
            nameflag = (nameflag and flag)

        if not nameflag:
            raise ValueError("One or more data names not available!")

        if legendloc:
            legend = legendloc if legendflag else None
        else:
            legend = legendflag

        return self.plot(xnames, ynames, splitx, splity, sharex, sharey,
                         xtrans, ytrans, legend, xlabels, ylabels, labels)

    @synchronized('lock')
    def _plotyt(self, ax, y, yname, windowsize=None, transform=None,
                label=None, *args, **kwargs):
        """
        Plot a data set versus its indices on the specified axes.

        `yname` is the label given to the line.

        `windowsize` specifies the number of data points to plot. Counts
        from the end. None (default) specifies all. `windowsize`s that
        exceed the length of the data set will use the full data set.

        `transform` is a function that is applied to the data set before
        it is plotted. It must be a vectorized function, i.e. it must accept
        as its only argument the full data set and return the full transformed
        data set.

        `label` overrides the programmatically generated label for the line.
        Since the label for this line is directly set to `yname` in this case,
        this function is redundant. The `label` argument is included to keep
        its notation similar to _plotxy.

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

        if yname is None:
            yname = 'index'

        plabel = yname if (label is None) else label

        y = y[-ws:]
        y = transform(y)
        line, = ax.plot(y, label=plabel, *args, **kwargs)
        return line

    @synchronized('lock')
    def _plotxy(self, ax, x, y, xname, yname, windowsize=None, xtrans=None,
                ytrans=None, label=None, *args, **kwargs):
        """
        Plot two data sets against each other on the specified axes.

        `xname` and `yname` are combined to create a label for the plot. This
        label may be overridden by the `label` argument.

        `windowsize` specifies the number of data points to plot. Counts
        from the end. None (default) specifies all. `windowsize`s that
        exceed the length of the data set will use the full data set.

        `xtrans` and `ytrans` are functions that are applied to the data sets
        before they are plotted. They must be vectorized functions, i.e. each
        must accept as its only argument a full data set and return the full
        transformed data set. `xtrans` modifies the x data set and `ytrans`
        modifies the y data set.

        `label` overrides the programmatically generated label for the line.
        If `label` is not None, then xname and yname are ignored.

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

        if xname is None:
            xname = 'index'
        if yname is None:
            yname = 'index'

        x = x[-ws:]
        x = xtrans(x)
        y = y[-ws:]
        y = ytrans(y)

        plabel = "{x} (x) vs {y} (y)".format(x=xname, y=yname)
        plabel = plabel if (label is None) else label

        line, = ax.plot(x, y, label=plabel, *args, **kwargs)
        return line

    @synchronized('lock')
    def clear(self):
        """
        Clears current plot.
        """
        self.fig.clear()
        self.mode = 'none'

    @synchronized('lock')
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
                try:
                    xmin, xmax, ymin, ymax = self._get_minmax(line)
                except ValueError:
                    return
                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)

            try:
                xmin = min(xmins)
            except ValueError:
                xmin = xminax
            try:
                xmax = max(xmaxs)
            except ValueError:
                xmax = xmaxax
            try:
                ymin = min(ymins)
            except ValueError:
                ymin = yminax
            try:
                ymax = max(ymaxs)
            except ValueError:
                ymax = ymaxax

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
                newymin = ymin - 0.1*dy
                newymax = ymax + 0.1*dy

                if xflag:
                    ax.set_xlim([newxmin, newxmax])
                if yflag:
                    ax.set_ylim([newymin, newymax])

    @staticmethod
    def _get_minmax(line):
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        try:
            xmin = min(xdata)
            xmax = max(xdata)
            ymin = min(ydata)
            ymax = max(ydata)
        except ValueError:
            errmsg = "Line {0} has no data.".format(repr(line))
            raise ValueError(errmsg)
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

    def windowsize(self, windowsize=None):
        """
        Set the window size.

        The window size is the number of samples from the end of the file that
        will be plotted, i.e. the last `windowsize` samples will be shown on
        the plots. As such, it should be an integer.

        A `windowsize` of `None` indicates that the full set of data should be
        shown. This is the default setting.
        """
        try:
            windowsize = int(windowsize)
        except ValueError:
            windowsize = None
        if windowsize <= 1:  # Would plot a single point
            windowsize = None
        self._plotdict['windowsize'] = windowsize


class PyOscopeRealtime(PyOscopeStatic):
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

    The `interval` argument is the period in milliseconds between update
    events. Defaults to 500 ms (i.e. 2 Hz). If this period is too short, it
    may cause instability in the plotter. PyOscopeRealtime will not allow
    `interval` < 10.

    The desired reader may be specified. The remaining arguments are
    passed to the reader. See reader.ReaderInterface for reader
    implementation notes.

    Example usage:

        >>> rt = PyOscopeRealtime(f='testdata.txt')
        >>> rt.data.columns
        >>> rt.plot('second', ['first', 'third'], legend=True)
    """
    def __init__(self, f=None, reader=None, interactive=True, toolbar=True,
                 interval=500, *args, **kwargs):
        PyOscopeStatic.__init__(self, f=f, reader=reader,
                                interactive=interactive, toolbar=toolbar,
                                *args, **kwargs)

        self._update_dict = {'none': self._pass,
                             'plot': self._update_plot}

        if self.interactive:
            # Bind update to MPL Idle event
            interval = max(interval, 10)
            self.timer = self.canvas.new_timer(interval=interval)
            self.timer.add_callback(self._update)
            self.timer.start()

    def __del__(self):
        self.stop()

    def stop(self):
        if self.interactive:
            # Unbind update function from timer and attempt to stop timer
            # NOTE: timer.stop() does nothing with macosx backend. This is an
            # MPL bug. The timer will continue to submit events to the MPL
            # loop, but with no callbacks they will not trigger anything.
            self.timer.remove_callback(self._update)
            self.timer.stop()
        self.close()

    def close(self):
        plt.close(self.fig)
        try:
            self.reader.close()
        except AttributeError:
            pass

    @synchronized('lock')
    def _update(self):
        if not self._initialized:
            return
        self.data = self.reader.update_data()
        self.callback()
        self._update_dict[self.mode]()

    def callback(self):
        """
        A custom function that is executed after the data is updated but
        before the plot is updated.

        Useful if you want to have some custom data handling before plotting.

        Normally does nothing. Overwrite this function instead of modifying
        `run` or `_update`.

        Note that overwriting this function must be done using the
        `set_callback` method, or else you will not easily be able to access
        instance variables.
        """
        pass

    @synchronized('lock')
    def set_callback(self, newfunc):
        """
        Monkey-patch the callback function to be replaced by `newfunc`.

        Note that the call signature of `newfunc` must be:

            newfunc(s)

        and instance variables (e.g. `self.data`) are accessed in `newfunc` by
        `s.<varname>` (e.g. `s.data`). As an example:

            def timestwo(s):
                s.data = s.data*2
            rt.set_callback(timestwo)

        would multiply all of the data by 2.
        """
        self.callback = MethodType(newfunc, self)

    @staticmethod
    def _pass():
        pass

    @synchronized('lock')
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
        xnames = self._plotdict['xnames']
        ynames = self._plotdict['ynames']
        xtrans = self._plotdict['xtrans']
        ytrans = self._plotdict['ytrans']

        if not oneD:
            # xs = [self.data[xname] for xname in xnames]
            xs = []
            for xname in xnames:
                if xname is None:
                    temp = self.data.columns[0]
                    toadd = np.array(range(len(self.data[temp])))
                    xs.append(toadd)
                else:
                    xs.append(self.data[xname])
        else:
            xs = None
        # ys = [self.data[yname] for yname in ynames]
        ys = []
        for yname in ynames:
            if yname is None:
                temp = self.data.columns[0]
                toadd = np.array(range(len(self.data[temp])))
                ys.append(toadd)
            else:
                ys.append(self.data[yname])

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
        self.canvas.draw_idle()

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

        if xtrans is None:
            xtrans = lambda x: x
        if ytrans is None:
            ytrans = lambda x: x

        if oneD:
            newx = range(len(y))
        else:
            newx = xtrans(x)
        newx = newx[-ws:]
        newy = ytrans(y)[-ws:]

        line.set_xdata(newx)
        line.set_ydata(newy)

    def _update_plot_wxagg(self):
        self._update_plot_slow() #DELME #FIXME



# Realtime one is typically expected
PyOscope = PyOscopeRealtime