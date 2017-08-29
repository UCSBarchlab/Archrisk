from collections import defaultdict
import itertools
import logging


class App(object):
    def __init__(self, tag, f, c=0):
        self.set_f(f)
        self.set_c(c)
        self.tag = tag

    def set_f(self, f):
        self.f = f

    def set_c(self, c):
        self.c = c

    def get_printable(self):
        return '{} ({}, {})'.format(self.tag, self.f, self.c)

    def gen_feed(self):
        feed_f = ('f', self.f)
        feed_c = ('c', self.c)
        return [feed_f, feed_c]
