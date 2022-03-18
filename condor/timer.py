"""A tic-toc analog in Python
Adapted from http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
"""
import time

class Timer(object):
    def __init__(self, name=None, verbose=False):
        self.verbose = verbose
        if name:
            if self.verbose: print(name)

    def __enter__(self):
        self.tic = time.time()

    def __exit__(self, type, value, traceback):
        if self.verbose: print('  Elapsed time: %.2f sec.' % (time.time() - self.tic))
