import numpy as np
from typing import List, Optional

from core.fa import fa
from core.dfa import dfa

class Ngram(dict):
    def __init__(self, iterable=None):
        super().__init__()
        self.fa = {}
        self.counts = {}
        self.sums = {}
        self.pos: List[int] = []
        self.bool: Optional[np.ndarray] = None
        self.dt = None
        self.R = None
        self.a = None
        self.gamma = None
        self.temp_fa = []

        if iterable:
            self.update(iterable)

    def update(self, iterable):
        for item in iterable:
            if item in self:
                self[item] += 1
            else:
                self[item] = 1

    def func(self, w, overlap_mode="overlapping", min_window=None, window_expansion=None,
             algo_selector='1', polynom_degree='1'):
        if overlap_mode == "non-overlapping" and (min_window is None or window_expansion is None):
            min_window = self.wh
            window_expansion = self.wh

        if algo_selector == '2':
            count, fa_val = dfa(self.data, (w, self.wh, self.l), overlap_mode, min_window, window_expansion, int(polynom_degree))
        else:
            count, fa_val = fa(self.data, (w, self.wh, self.l), overlap_mode, min_window, window_expansion)

        self.counts[w] = count
        self.fa[w] = fa_val


class newNgram:
    def __init__(self, data, wh, l):
        self.data = data
        self.count = {}
        self.dfa = {}
        self.wh = wh
        self.l = l
        self.dt = None
        self.R = None
        self.a = None
        self.gamma = None
        self.temp_dfa = []
        self.goodness = None

    def func(self, w, overlap_mode="overlapping", min_window=None, window_expansion=None,
             algo_selector='1', polynom_degree='1'):
        if overlap_mode == "non-overlapping" and (min_window is None or window_expansion is None):
            min_window = self.wh
            window_expansion = self.wh

        if algo_selector == '2':
            count, fa_val = dfa(self.data, (w, self.wh, self.l), overlap_mode, min_window, window_expansion, int(polynom_degree))
        else:
            count, fa_val = fa(self.data, (w, self.wh, self.l), overlap_mode, min_window, window_expansion)

        self.count[w] = count
        self.dfa[w] = fa_val