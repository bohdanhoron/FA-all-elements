import matplotlib.pyplot as plt

from core.fa import fa
from core.dfa import dfa

class Ngram(dict):
    def __init__(self, iterable=None):  # Ініціалізували наш розподіл як новий об'єкт класу, додаємо наявні елементи
        super(Ngram, self).__init__()
        self.fa = {}
        self.counts = {}
        self.sums = {}
        if iterable:
            self.update(iterable)

    def update(self, iterable):  # Оновлюємо розподіл елементами з наявного ітеруємого набору даних
        for item in iterable:
            if item in self:
                self[item] += 1
            else:
                self[item] = 1

    def hist(self):
        plt.bar(self.keys(), self.values())
        plt.show()


class newNgram():
    def __init__(self, data, wh, l):
        self.data = data
        self.count = {}
        self.dfa = {}
        self.wh, self.l = wh, l

    def func(self, w, overlap_mode="overlapping", min_window=None, window_expansion=None, algo_selector='1', polynom_degree='1'):
        if overlap_mode == "non-overlapping" and (min_window is None or window_expansion is None):
            min_window = self.wh
            window_expansion = self.wh

        if algo_selector == '2':
            count, result = dfa(self.data, (w, self.wh, self.l), overlap_mode, min_window, window_expansion, int(polynom_degree))
        else:
            count, result = fa(self.data, (w, self.wh, self.l), overlap_mode, min_window, window_expansion)

        self.count[w] = count
        self.dfa[w] = result # Окремо обчислюємо MSE для count