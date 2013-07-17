pyoscope
========

Python generic oscilloscope-like plotter

pyoscope.py
jlazear
2013-07-17

Generic oscilloscope-like plotting for visualizing data from
instruments.

Example:

    rt = PyOscope(f='testdata.txt')
    rt.data.columns
    rt.plot('second', ['first', 'third'], legend=True)

Basic Idea
----------

A generic threaded plotting object that points at a data file and
reads it in. The intent is that with these objects, a hardware
interface needs to worry only about writing data to a file in a
sensible way. Then `pyoscope` could be invoked completely
independently of the interface to handle plotting of the data in a
sufficiently flexible way. I want to decouple the plotting of data
from the acquisition of data!

Reading in should be done real-time, so file is monitored
for updates. Then should either parse header for metadata, e.g. column
names, or be passed in a dictionary with this information. Numbers of
columns should be automatically determined and referenced in some
reasonable way (so measurement-specific names do not need to be used
if unwanted or do not exist).

Installation
------------

Install directly from `github` using `pip` with

    pip install git+git://github.com/jlazear/pyoscope.git