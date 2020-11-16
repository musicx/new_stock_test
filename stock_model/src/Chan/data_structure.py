import numpy as np
import pandas as pd
import datetime as dt
from collections import Counter
from functools import reduce
#import QUANTAXIS as qa

class Candle(object):
    def __init__(self, open, close, high, low, date, start_position):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.date = date  # for display purpose
        self.start_position = start_position  # as in the raw sequence

        self.left = 0
        self.right = 0

    def period(self):
        return 1 + self.left + self.right

    def higher(self, other):
        return self.high >= other.high and self.low >= other.low

    def lower(self, other):
        return self.high <= other.high and self.low <= other.low

    def contain(self, other):
        return self.high >= other.high and self.low <= other.low

    def merge(self, right_candle, is_rising=True):
        new_date = self.date if self.contain(right_candle) else right_candle.date
        if is_rising:
            candle = Candle(max(self.open, right_candle.open), max(self.close, right_candle.close),
                            max(self.high, right_candle.high), max(self.low, right_candle.low),
                            new_date, self.start_position)
        else:
            candle = Candle(min(self.open, right_candle.open), min(self.close, right_candle.close),
                            min(self.high, right_candle.high), min(self.low, right_candle.low),
                            new_date, self.start_position)
        if self.contain(right_candle):
            candle.left = self.left
            candle.right = self.right + right_candle.period()
        else:
            candle.left = self.period() + right_candle.left
            candle.right = right_candle.right
        return candle


class Fractal(object):
    def __init__(self, index, position, is_top, is_valid):
        self.index = index
        self.position = position
        self.is_top = is_top
        self.is_valid = is_valid


class Stroke(object):
    def __init__(self, start_idx, end_idx, candles):
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.start_pos = candles[start_idx].start_position + candles[start_idx].left
        self.end_pos = candles[end_idx].start_position + candles[end_idx].left

        self.start_date = candles[start_idx].date
        self.end_date = candles[end_idx].date

        self.is_rising = candles[start_idx].high < candles[end_idx].high and candles[start_idx].low < candles[end_idx].low
        self.high = max(candles[start_idx].high, candles[end_idx].high)
        self.low = min(candles[start_idx].low, candles[end_idx].low)

        '''
        span = self.high - self.low
        self.low_extreme = self.low + span * 0.238 if self.is_rising else self.high - span * 0.238
        self.low_energy = self.low + span * 0.382 if self.is_rising else self.high - span * 0.382
        self.mid_energy = self.low + span * 0.5
        self.high_energy = self.low + span * 0.382 if not self.is_rising else self.high - span * 0.382
        self.minor_target = self.high + span * 0.382 if self.is_rising else self.low - span * 0.382
        self.first_target = self.high + span * 0.618 if self.is_rising else self.low - span * 0.618
        self.major_target = self.high + span if self.is_rising else self.low - span
        self.second_target = self.high + span * 1.618 if self.is_rising else self.low - span * 1.618

        within = [(self.low_energy, '0.382 of\t{{}} {} to\t{{}}'.format('up' if self.is_rising else 'down')),
                  (self.mid_energy,  '0.5   of\t{{}} {} to\t{{}}'.format('up' if self.is_rising else 'down')),
                  (self.high_energy, '0.618 of\t{{}} {} to\t{{}}'.format('up' if self.is_rising else 'down'))]
        outside = [(self.high if self.is_rising else self.low, 'end   of\t{{}} {} to\t{{}}'.format('up' if self.is_rising else 'down')),
                   (self.minor_target, '1.382 of\t{{}} {} to\t{{}}'.format('up' if self.is_rising else 'down')),
                   (self.first_target, '1.618 of\t{{}} {} to\t{{}}'.format('up' if self.is_rising else 'down')),
                   (self.major_target, '2.0 of\t{{}} {} to\t{{}}'.format('up' if self.is_rising else 'down')),
                   (self.second_target, '2.618 of\t{{}} {} to\t{{}}'.format('up' if self.is_rising else 'down'))]

        within = [w for w in within if w[0] > 1]
        outside = [o for o in outside if o[0] > 1]

        self.support = within if self.is_rising else outside
        self.pressure = outside if self.is_rising else within

        self.end_change = False
        self.valid = True
        

    def change_endpoint(self):
        if self.is_rising:
            try:
                self.pressure.remove((self.high, 'end of\t{} up to\t{}'))
            except:
                pass
            self.support.append((self.high, 'end of\t{} up to\t{}'))
        else:
            try:
                self.support.remove((self.low, 'end of\t{} down to\t{}'))
            except:
                pass
            self.pressure.append((self.low, 'end of\t{} down to\t{}'))
    '''

    def tostring(self):
        return '{} {} to {}'.format(self.start_date, 'up' if self.is_rising else 'down', self.end_date)


class Hub(object):
    def __init__(self):
        self.strokes = []
        self.hub_high = 0
        self.hub_low = 0
        self.highest_high = 0
        self.lowest_low = 0
        self.start_pos = 0
        self.end_pos = 0

        self.is_valid = True

    #def append_stroke(self, stroke):


class ChartFrame(object):
    def __init__(self, code, frame, limit=0):  # frame should be 'day', '60min' etc
        self.raw_candles = []
        self.merged_candles = []
        self.fractals = []
        self.strokes = []
        self.hubs = []
        self.code = code
        self.frame = frame
        self.limit = limit

    def init_candles(self, ochls, dates=None):
        use_pos = dates is None or len(dates) < len(ochls)
        start_pos = len(ochls) - self.limit if 0 < self.limit < len(ochls) else 0
        for pos, ochl in enumerate(ochls):
            if pos < start_pos:
                continue
            self.raw_candles.append(Candle(ochl[0], ochl[1], ochl[2], ochl[3],
                                           'pos_{}'.format(pos) if use_pos else dates[pos], pos))
        print('init data for {}'.format(self.code))
        self.refresh_structures()

    def add_candle(self, ochl, date=None):
        use_pos = date is None
        pos = len(self.raw_candles)
        if len(self.raw_candles) >= self.limit:
            self.raw_candles.pop(0)
            if self.raw_candles[0].date.startswith('pos') or use_pos:
                for idx in range(len(self.raw_candles)):
                    self.raw_candles[idx].date = 'pos_{}'.format(idx)
        self.raw_candles.append(Candle(ochl[0], ochl[1], ochl[2], ochl[3],
                                       'pos_{}'.format(pos) if use_pos else date, pos))
        self.refresh_structures()

    def refresh_structures(self):
        self.merge_candles()
        #self.find_fractals()
        self.find_strokes()
        self.find_hubs()

    def merge_candles(self):
        merged = []
        for candle in self.raw_candles:
            if len(merged) == 0:
                merged.append(candle)
                continue
            if len(merged) > 1:
                rising = merged[-1].higher(merged[-2])
            else :
                rising = not merged[-1].higher(candle)

            if candle.contain(merged[-1]) or merged[-1].contain(candle):
                merged[-1] = merged[-1].merge(candle, rising)
            else:
                merged.append(candle)
        self.merged_candles = merged

    def find_fractals(self):
        fractals = []
        for idx, candle in enumerate(self.merged_candles[:-1]):
            position = self.merged_candles[idx].start_position + self.merged_candles[idx].left
            if (idx == 0 or candle.higher(self.merged_candles[idx - 1])) and candle.higher(self.merged_candles[idx + 1]):
                fractals.append(Fractal(idx, position, True, True))
            elif (idx == 0 or candle.lower(self.merged_candles[idx - 1])) and candle.lower(self.merged_candles[idx + 1]):
                fractals.append(Fractal(idx, position, False, True))
        self.fractals = fractals

    def find_strokes(self):
        fractals = []
        for idx, candle in enumerate(self.merged_candles[:-1]):
            insert = True
            position = self.merged_candles[idx].start_position + self.merged_candles[idx].left
            if (idx == 0 or candle.higher(self.merged_candles[idx - 1])) and candle.higher(self.merged_candles[idx + 1]):
                fractals.append(Fractal(idx, position, True, True))
            elif (idx == 0 or candle.lower(self.merged_candles[idx - 1])) and candle.lower(self.merged_candles[idx + 1]):
                fractals.append(Fractal(idx, position, False, True))
            else:
                insert = False

            if insert:
                valid_ones = [frac for frac in fractals if frac.is_valid]
                if len(valid_ones) > 1:
                    last = valid_ones.pop(-1)
                    fractals[-1].is_valid = False
                    second = valid_ones[-1]
                    if last.is_top == second.is_top:
                        if (last.is_top and self.merged_candles[last.index].high > self.merged_candles[second.index].high
                                or not last.is_top and self.merged_candles[last.index].low < self.merged_candles[second.index].low):
                            for rev_idx in range(len(fractals), 0, -1):
                                if fractals[rev_idx - 1].position == second.position:
                                    fractals[rev_idx - 1].is_valid = False
                                    break
                            fractals[-1].is_valid = True
                    elif last.index - second.index > 2 and last.position - second.position > 3:
                        fractals[-1].is_valid = True
                    elif last.position - second.position <= 3 and len(valid_ones) > 2:
                        if (last.is_top and self.merged_candles[last.index].high > self.merged_candles[valid_ones[-2].index].high
                                and self.merged_candles[second.index].low > self.merged_candles[valid_ones[-3].index].low
                            or not last.is_top and self.merged_candles[last.index].low < self.merged_candles[valid_ones[-2].index].low
                                and self.merged_candles[second.index].high < self.merged_candles[valid_ones[-3].index].high):
                            for rev_idx in range(len(fractals), 0, -1):
                                if fractals[rev_idx - 1].position == second.position:
                                    fractals[rev_idx - 1].is_valid = False
                                elif fractals[rev_idx - 1].position == valid_ones[-2].position:
                                    fractals[rev_idx - 1].is_valid = False
                                    break
                            fractals[-1].is_valid = True
        self.fractals = fractals
        strokes = []
        valid_fracs = [frac for frac in self.fractals if frac.is_valid]
        for idx, frac in enumerate(valid_fracs[:-1]):
            stroke = Stroke(frac.index, valid_fracs[idx + 1].index, self.merged_candles)
            print(stroke.tostring())
            strokes.append(stroke)
        self.strokes = strokes

    def find_hubs(self):
        hubs = []
        index = 1
        while index < len(self.strokes) - 2:
            past_stroke = self.strokes[index - 1]
            current_stroke = self.strokes[index]
            next_stroke = self.strokes[index + 2]
            if past_stroke.high - past_stroke.low < current_stroke.high - current_stroke.low:
                index += 1
                continue

            hub_high = min(current_stroke.high, next_stroke.high)
            hub_low = max(current_stroke.low, next_stroke.low)
            if hub_low >= hub_high:
                index += 1
                continue

            hub = Hub()
            hub.hub_high = hub_high
            hub.hub_low = hub_low
            hub.highest_high = max(current_stroke.high, next_stroke.high)
            hub.lowest_low = min(current_stroke.low, next_stroke.low)
            hub.start_pos = current_stroke.start_pos + 1
            hub.end_pos = next_stroke.end_pos - 1
            hub.strokes.extend(self.strokes[index: index + 3])

            check_index = index + 3
            while check_index < len(self.strokes):
                check_stroke = self.strokes[check_index]
                if check_stroke.low > hub.hub_high or check_stroke.high < hub.hub_low:
                    break
                hub.highest_high = max(hub.highest_high, check_stroke.high)
                hub.lowest_low = min(hub.lowest_low, check_stroke.low)
                hub.end_pos = check_stroke.end_pos - 1
                hub.strokes.append(check_stroke)
                check_index += 1

            hubs.append(hub)
            index = check_index
        for hub in hubs:
            if len(hub.strokes) < 3:
                hub.is_valid = False

        self.hubs = hubs

    def determine_range(self, fractal):
        before_idx = max(0, fractal.index - 1)
        after_idx = min(len(self.merged_candles) - 1, fractal.index + 1)

        # first try, use the full fractal range
        # high = max(self.merged_candles[fractal.index].high,
        #            self.merged_candles[before_idx].high,
        #            self.merged_candles[after_idx].high)
        # low = min(self.merged_candles[fractal.index].low,
        #           self.merged_candles[before_idx].low,
        #           self.merged_candles[after_idx].low)

        # second try, use the higher of side solid parts and the high of the center, vice versa
        if fractal.is_top:
            high = self.merged_candles[fractal.index].high
            low = max(max(self.merged_candles[before_idx].open, self.merged_candles[before_idx].close),
                      max(self.merged_candles[after_idx].open, self.merged_candles[after_idx].close))
        else:
            low = self.merged_candles[fractal.index].low
            high = min(min(self.merged_candles[before_idx].open, self.merged_candles[before_idx].close),
                       min(self.merged_candles[after_idx].open, self.merged_candles[after_idx].close))
        return np.int32(np.arange(low*100, high*100+1, 1))

    def detect_price_boundary(self):
        candle_size = len(self.raw_candles)
        recent_fractals = [frac for frac in self.fractals if frac.position > candle_size - 500 and frac.is_valid]
        fractal_ranges = [self.determine_range(frac) for frac in recent_fractals]
        counters = [Counter(price_range) for price_range in fractal_ranges]
        counter = reduce(lambda x, y: x + y, counters)
        sorted_counter = sorted(counter.items(), key=lambda x: x[0])

        count_trend = False
        sum_price = 0
        count_price = 0
        current_count = 0
        high_freq_prices = []
        for price, count in sorted_counter:
            if count != current_count:
                if current_count != 0 and count < current_count and count_trend:
                    average_price = sum_price / count_price
                    high_freq_prices.append(average_price)
                    count_trend = False
                if count > current_count:
                    count_trend = True
                current_count = count
                sum_price = 0
                count_price = 0
            sum_price += price
            count_price += 1
        return [round(price) / 100 for price in high_freq_prices]
