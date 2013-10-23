from readers import HexReader
import pyoscope as pyo
import time
import numpy as np
import os
from matplotlib.backends.backend_wxagg import \
    FigureCanvasWxAgg as FigCanvas, \
    NavigationToolbar2WxAgg as NavigationToolbar


def writestuff(num=10, interval=.1, fname='testdata.txt', mode='a+',
               header=False):
    with open(fname, mode) as f:
        if header:
            f.write("# navg: 167\n# mode: data\n# columns: [locked, adc]\n")
            f.flush()
            os.fsync(f.fileno())
        for i in range(num):
            rint1 = np.random.randint(0, 65535)
            rint2 = np.random.randint(0, 65535)
            locked = hex(rint1)[2:]
            adc = hex(rint2)[2:]
            towrite = "{0} {1}\n".format(locked, adc)
            f.write(towrite)
            f.flush()
            os.fsync(f.fileno())
            time.sleep(interval)

writestuff(interval=0, num=10, mode='w', header=True)
time.sleep(.1)
pt = pyo.PyOscopeRealtime('testdata.txt', reader=HexReader, interactive=True)

