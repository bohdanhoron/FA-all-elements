from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from numba import njit

#from app import fit, calc_non_overlapping_shift
from calculations.stats import s, mse, R
from calculations.fitting_functions import power_law

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

    """def hist(self):
        plt.bar(self.keys(), self.values())
        plt.show()"""


class newNgram():
    def __init__(self, data, wh, l):
        self.data = data
        self.count = {}
        self.dfa = {}
        self.wh, self.l = wh, l

    def func(self, w, overlap_mode="overlapping", min_window=None, window_expansion=None):
        if overlap_mode == "non-overlapping" and (min_window is None or window_expansion is None):
            min_window = self.wh
            window_expansion = self.wh

        count = self.dynamic_count((w, self.wh, self.l), overlap_mode, min_window, window_expansion)
        self.count[w] = count
        self.dfa[w] = float(mse(count)) # Окремо обчислюємо MSE для count

    #NOTE: перейменувати
    def dynamic_count(self, #data: List,
                      args: Tuple[int, int, int],
                      overlap_mode: str = "overlapping",
                      min_window: Optional[int] = None,
                      window_expansion: Optional[int] = None) -> np.ndarray:
        """
        Виконує аналіз флуктуацій (DFA) для даних. !НЕПРАВИЛЬНО, аналізу там нема, тільки розрахунок позицій нових слів у динамічному режимі
        
        Args:
            data: Вхідні дані для аналізу
            args: Кортеж (розмір вікна, зсув вікна, довжина даних)
            overlap_mode: Режим перекриття вікон ("overlapping" або "non-overlapping")
            min_window: Мінімальний розмір вікна для режиму non-overlapping
            window_expansion: Значення розширення вікна для режиму non-overlapping
            
        Returns:
            np.ndarray: Масив результатів DFA аналізу
        """
        wi, wh, l = args
        
        if overlap_mode == "overlapping":
            # Стандартний режим з фіксованим зміщенням
            window_count = len(range(0, l - wi, wh))
            #count = np.zeros(window_count, dtype=np.uint8)
            count = np.zeros(window_count, dtype=np.uint16)
            
            for index, i in enumerate(range(0, l - wi, wh)):
                temp_v = []
                x = []
                for ngram in self.data[i:i + wi]:
                    if ngram in temp_v:
                        x.append(0)
                    else:
                        temp_v.append(ngram)
                        x.append(1)
                count[index] = s(np.array(x, dtype=np.uint8))
        else:
            # Non-overlapping режим
            if min_window is None:
                min_window = wh
            if window_expansion is None:
                window_expansion = wh
                
            # Оцінюємо кількість і розташування вікон
            k = 1
            i = 0
            window_positions = []
            while i < l - wi:
                window_positions.append(i)
                shift = calc_non_overlapping_shift(k, min_window, window_expansion)
                i += shift
                k += 1
                
            count = np.zeros(len(window_positions), dtype=np.uint16)
            #count = np.zeros(len(window_positions), dtype=np.uint8)
            for index, i in enumerate(window_positions):
                temp_v = []
                x = []
                for ngram in self.data[i:i + wi]:
                    if ngram in temp_v:
                        x.append(0)
                    else:
                        temp_v.append(ngram)
                        x.append(1)
                count[index] = s(np.array(x, dtype=np.uint8))
        
        return count


    def fit(self):
        try:
            dfa_keys = sorted(list(self.dfa.keys()))
            #dfa_values = list(new_ngram.dfa.values())
            dfa_values = [self.dfa[key] for key in dfa_keys]

            #print(dfa_keys)
            #print(dfa_values)
            
            # Перевірка наявності достатньої кількості даних для підбору кривої
            if len(dfa_keys) < 2 or len(dfa_values) < 2:
                print("Недостатньо даних для підбору кривої")
                self.a = 0.0
                self.gamma = 0.0
                self.temp_dfa = [0.0] * (len(dfa_keys) if dfa_keys else 1)
                self.goodness = 0.0
            else:
                c, _ = curve_fit(power_law, dfa_keys, dfa_values, method='lm', maxfev=5000)
                self.a = round(c[0], 8)
                self.gamma = round(c[1], 8)
                
                # Оптимізуємо обчислення temp_dfa
                self.temp_dfa = [power_law(w, self.a, self.gamma) for w in dfa_keys]
                self.goodness = round(r2_score(dfa_values, self.temp_dfa), 8)
            
            # Звільняємо пам'ять від тимчасових змінних
            del dfa_keys, dfa_values
        except Exception as e:
            print(f"Помилка при підборі кривої: {e}")
            #new_ngram.a = 1.0
            #new_ngram.gamma = 0.5
            self.a = 0
            self.gamma = 0
            self.temp_dfa = []
            self.goodness = 0.0

@njit(fastmath=True)
def calc_non_overlapping_shift(k, min_window, window_expansion):
    """
    Розраховує зміщення для режиму non-overlapping
    k - номер кроку (починаючи з 1)
    """
    # Numba не працює з None значеннями, тому перевірка робиться в make_windows
    if k == 1:
        return min_window
    else:
        return min_window + (k-1) * window_expansion
