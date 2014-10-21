#!/bin/env python

"""
readers.py
jlazear
2013-07-02

Readers for pyoscope.

Long description

Example:

<example code here>
"""
version = 20130702
releasestatus = 'beta'

import numpy as np
import pandas as pd
from types import StringTypes
from tempfile import _TemporaryFileWrapper


class ReaderInterface(object):
    """
    A reader "interface". Simply lists the methods that a pyoscope
    reader must implement.

    Readers are not meant to be threaded. Any threading would happen
    by the object that owns the reader. They should, however,
    facilitate threading by providing the `update_data` method.

    A reader must implement the following methods:

        reader.init_data(*args, **kwargs)
            - Intial read of data
            - Returns a single numpy array with the data in columns.
              May be a structured array or Pandas DataFrame.
        reader.update_data()
            - Reads changes to the data file
            - Returns a single updated numpy/pandas array.
            - Note that this does not necessarily require making a new
              array. init_data may return a reference to an array that
              reader owns, then update_data could update the same
              array, and update_data could return a reference to the
              same array.
        reader.switch_file(f, *args, **kwargs)
            - Switches which file is being read.
        reader.close()
            - Closes and makes safe the file.

    and the following attribute:

        reader.filename
            - Filename of read file
    """
    def __init__(self, f, *args, **kwargs):
        # Load file
        if isinstance(f, file):
            self.f = f
        elif isinstance(f, StringTypes):
            self.f = open(f, 'r')
        else:
            raise TypeError('f must be a file handle or filename.')
        self.f.seek(0)
        self.filename = self.f.name

        # Other init stuff...

    def close(self):
        self.f.close()

    def init_data(*args, **kwargs):
        """
        Initial read of the data.

        Returns a single array with the data in columns. May be a
        structured array or Pandas DataFrame.
        """
        data = np.empty((1, 1))
        return data

    def update_data():
        """
        Reads changes to the data file.

        Returns a single updated numpy/pandas array.

        Note that this does not necessarily require making a new array
        (and thereby consuming more resources). One could store the
        data array in the reader object and return a reference to that
        array from `init_data`, then update the same array in
        `update_data`, and return a reference to the same array.
        """
        data = np.empty((1, 1))
        return data

    def switch_file(self, f, *args, **kwargs):
        # Load file
        if isinstance(f, file):
            self.f = f
        elif isinstance(f, StringTypes):
            self.f = open(f, 'r')
        else:
            raise TypeError('f must be a file handle or filename.')
        self.f.seek(0)

        return self.init_data(*args, **kwargs)


class DefaultReader(object):
    """
    Default reader for pyoscope. Essentially a wrapper around pandas's
    read_csv. Note that read_csv is a little funky. Check its documentation
    if you are getting unexpected behavior.

    Note: Sets header=None by default, unless overridden.

    See ReaderInterface for info on readers.
    """
    def __init__(self, f, *args, **kwargs):
        # Load file
        if isinstance(f, file) or isinstance(f, _TemporaryFileWrapper):
            mode = self.f.mode
            if ('r' in mode) or ('+' in mode):
                self.f = f
            else:
                self.f = open(f.name, 'r')
        elif isinstance(f, StringTypes):
            self.f = open(f, 'r')
        else:
            raise TypeError('f must be a file handle or filename.')
        self.f.seek(0)
        self.filename = self.f.name

    def close(self):
        self.f.close()

    def init_data(self, *args, **kwargs):
        if self.f.closed:
            raise ValueError('I/O operation on closed file.')
        self.args = args
        self.kwargs = kwargs
        if 'header' not in kwargs:
            kwargs.update(header=None)
        self.f.seek(0)
        data = pd.read_csv(self.f, *args, **kwargs)
        # data = np.loadtxt(self.f, *args, **kwargs)
        return data

    def update_data(self):
        args = self.args
        kwargs = self.kwargs
        return self.init_data(*args, **kwargs)

    def switch_file(self, f, *args, **kwargs):
        self.__init__(f)
        return self.init_data(*args, **kwargs)


class HexReader(object):
    """
    Reader for ASCII-Hex encoded data files.

    See ReaderInterface for info on readers.
    """
    def __init__(self, f, header=True, *args, **kwargs):
        # Load file
        if isinstance(f, file) or isinstance(f, _TemporaryFileWrapper):
            mode = f.mode
            if ('r' in mode) or ('+' in mode):
                self.f = f
            else:
                self.f = open(f.name, 'r')
        elif isinstance(f, StringTypes):
            self.f = open(f, 'r')
        else:
            raise TypeError('f must be a file handle or filename.')
        self.f.seek(0)
        self.filename = self.f.name

        if header:
            self.header = self._read_header()
            if 'columns' in self.header:
                cols = self.header['columns']
                self.header['columns'] = self._split_columns(cols)
            if 'navg' in self.header:
                if isinstance(self.header['navg'], StringTypes):
                    navg = self._split_columns(self.header['navg'],
                                               typecast=float)
                    self.header['navg'] = navg
        else:
            self.header = {}

        # Read first non-comment line
        line = '#'
        while line.startswith('#'):
            line = self.f.readline()

        if 'columns' in self.header:
            self.numcols = len(self.header['columns'])
        else:
            self.numcols = len(line.split())
        self._cdict = {i: self._cfunc for i in range(self.numcols)}

    @staticmethod
    def _cfunc(val):
        return float(int(val, 16))

    @staticmethod
    def _split_columns(colstr, typecast=str):
        """
        "[rdac, adc]" or "rdac, adc" or "[rdac adc]" or "rdac adc"
        all go to -> ["rdac", "adc"]

        unless typecast is specified, in which case typecast() is called
        on every value in list
        """
        colstr = colstr.strip('[]')
        if ',' in colstr:
            splitchar = ','
        else:
            splitchar = ' '
        collist = colstr.split(splitchar)
        collist = [typecast(col.strip()) for col in collist]
        return collist

    def _read_header(self, conversiondict=None):
        """
        Read the header from a data file.

        The file pointer is reset to the beginning of the file before
        parse_header returns.

        :Arguments:
            conversiondict - (dict) A dictionary containing header key
                names as keys and a function as the value. The value
                corresponding to the header key name will be operated
                on by the specified function. This is used, e.g., to
                convert a header value to a float or int. Header keys that
                are not listed in conversiondict are attempted to be cast to
                floats, but if this fails then they are unmodified, i.e. the
                corresponding values are left as strings.

        :Returns:
            headerdict - (dict) A dictionary of key-value pairs parsed
                from the header.
        """
        # Make sure we read from the beginning of the file
        self.f.seek(0)

        # Use try/finally block to guarantee that file is left in known
        # state
        counter = 0
        try:
            headerdict = {}
            for line in self.f:
                if '#' in line:
                    counter += 1
                if ':' in line:
                    slist = line.split(';')[0].split(':', 1)
                    key = slist[0].split('#')[1].strip()
                    value = slist[1].strip()
                    try:
                        headerdict[key] = float(value)
                    except ValueError:
                        headerdict[key] = value
        finally:
            self.f.seek(0)

        if conversiondict is None:
            conversiondict = {}
        for key, value in headerdict.items():
            try:
                modfunc = conversiondict[key]
                headerdict[key] = modfunc(value)
            except KeyError:
                pass

        headerdict['skiprows'] = counter

        return headerdict

    def close(self):
        self.f.close()

    def init_data(self, *args, **kwargs):
        if self.f.closed:
            raise ValueError('I/O operation on closed file.')
        self.args = args
        self.kwargs = kwargs
        if 'columns' in self.header:
            names = self.header['columns']
        else:
            names = ['col' + str(i) for i in range(self.numcols)]
        formats = [np.float]*len(names)
        dtdict = {'names': names, 'formats': formats}
        dt = np.dtype(dtdict)

        data = []
        self.f.seek(0)
        for line in self.f:
            if not line.startswith('#'):
                lsplit = line.split(' ')
                templist = []
                for i, val in enumerate(lsplit):
                    conv = self._cdict[i]
                    val = conv(val)
                    templist.append(val)
                templist = tuple(templist)
                data.append(templist)
        data = np.array(data, dtype=dt)

        if 'navg' in self.header:
            navg = self.header['navg']
            if isinstance(navg, float):
                navg = [navg]*self.numcols
        else:
            navg = [1.]*self.numcols

        for i, n in enumerate(navg):
            col = names[i]
            data[col] = data[col]/n

        data = pd.DataFrame(data)
        return data

    def update_data(self):
        args = self.args
        kwargs = self.kwargs
        return self.init_data(*args, **kwargs)

    def switch_file(self, f, *args, **kwargs):
        self.__init__(f)
        return self.init_data(*args, **kwargs)
