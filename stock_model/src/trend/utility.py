import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime as dt


def higher(x, y):
    return x.high >= y.high and x.low >= y.low


def contain(x, y):
    return x.high >= y.high and x.low <= y.low


def lower(x, y):
    return x.high <= y.high and x.low <= y.low


def merge_klines(klines):
    merged = []
    raw_idx = 0
    while raw_idx < len(klines):
        if len(merged) < 2:
            merged.append(klines[raw_idx])
            raw_idx += 1
            continue
        raising = ((len(merged) < 3 and merged[-1].close > merged[-1].open) or
                   (len(merged) > 2 and (merged[-3].high + merged[-3].close * 2 + merged[-3].low) <= (merged[-2].high + merged[-2].close * 2 + merged[-2].low)))
        if not(contain(merged[-2], merged[-1]) or contain(merged[-1], merged[-2])):
            merged.append(klines[raw_idx])
            raw_idx += 1
        if contain(merged[-2], merged[-1]):
            kline = merged.pop()
            merged[-1] = merged[-1].merge(kline, raising, False)
        elif contain(merged[-1], merged[-2]):
            kline = merged.pop()
            merged[-1] = merged[-1].merge(kline, raising, True)

    return merged


def find_endpoints(klines):
    end = []  # item is (idx in merged list, is top end point, valid end point)
    last = 0
    for idx, kline in enumerate(klines[:-1]):
        insert = False
        if (idx == 0 or higher(kline, klines[idx - 1])) and higher(kline, klines[idx + 1]):
            end.append([idx, True, True])
            insert = True
        elif (idx == 0 or lower(kline, klines[idx - 1])) and lower(kline, klines[idx + 1]):
            end.append([idx, False, True])
            insert = True

        if insert and len(end) > 1:
            dist = ((idx - 1 == last and klines[last].right + kline.left > 2)
                    or (idx - 2 == last and klines[last].right + klines[last + 1].period() + kline.left > 2)
                    or (idx - 3 == last and klines[last].right + klines[last + 1].period() + klines[last+2].period() + kline.left > 2)
                    or idx - 3 > last)
            if end[-1][1] == end[-2][1]:
                if ((end[-1][1] and klines[end[-1][0]].high > klines[end[-2][0]].high) or
                        (not end[-1][1] and klines[end[-1][0]].low < klines[end[-2][0]].low)):
                    if dist:
                        end[-2][2] = False
                    else:
                        end.pop(-2)
                else:
                    end.pop()
                    insert = False
            elif not dist:
                end.pop()
                insert = False

        if insert:
            last = idx
    return end


def find_cycle(points, klines, mean_allowance=0.1, max_allowance=4, keep_invalid=False, use_merge=True):
    end_idx = dict([(p[0], (p[1], p[2])) for p in points if keep_invalid or p[2]])
    cumu = 0
    slices = []   # item is of (# or ticks, is Peak?)
    for idx, kline in enumerate(klines):
        if idx in end_idx:
            if cumu != 0:
                slices.append((cumu, False))
            if use_merge:
                slices.append((kline.period(), True))
            else:
                slices.append((kline.left, False))
                slices.append((1, True))
                slices.append((kline.right, False))
            cumu = 0
        else:
            cumu += kline.period()
    if len(slices) < 3:
        return []
    ticks = []
    for cnt, flag in slices:
        ticks.extend([flag] * cnt)
    start_point = slices[0][0] + slices[1][0]    # TODO maybe a different start point would result in a different cycle

    cycles = []   # modified items: length, avg, std
    for cycle in range(3, 120):
        length = []
        # print('cycle {}'.format(cycle))
        short = int(max(np.floor(cycle * (1 - mean_allowance)), cycle - max_allowance))
        long = int(min(np.ceil(cycle * (1 + mean_allowance)), cycle + max_allowance))
        found = False
        point = start_point
        before = point
        after = point + slices[2][0]
        while not found:
            if np.all(ticks[point:point+short]) and np.all(ticks[point:point+long]):
                break
                # always in peak phrase
            try:
                last_point = point
                if np.any(ticks[point+short: point+long]):
                    peaks = [short+i for i, p in enumerate(ticks[point+short: point+long]) if p]
                    min_distance = np.min([np.abs(cycle - d) for d in peaks])
                    point = point+cycle+min_distance if ticks[point+cycle+min_distance] else point+cycle-min_distance
                    for p in range(point-1, last_point+short-1, -1):
                        if ticks[p] and p > last_point:
                            before = p     # earliest peak point in full range
                    for p in range(point, last_point+long):
                        if ticks[p] and p > last_point:
                            after = p      # latest peak point in full range
                    length.append(point - last_point)
                elif np.any(ticks[before+short: point+short]):
                    delta = int((point - before) / 2.)
                    for p in range(last_point+short-1, before+short-1, -1):
                        if ticks[p] and p > last_point:
                            before = p
                    for p in range(before+short, last_point+short):
                        if ticks[p] and p > last_point:
                            point = p
                            after = p
                    if len(length) > 1:
                        length[-1] -= delta
                    length.append(point - last_point + delta)
                elif np.any(ticks[point+long: after+long]):
                    delta = int((after - point) / 2.)
                    for p in range(after+long-1, last_point+long-1, -1):
                        if ticks[p] and p > last_point:
                            before = p
                            point = p
                    for p in range(last_point+long, after+long):
                        if ticks[p] and p > last_point:
                            after = p
                    if len(length) > 1:
                        length[-1] += delta
                    length.append(point - last_point - delta)
                elif point < len(ticks):
                    break
                    # cannot find peak in given range
                if last_point == point:
                    break
                    # cannot move forward
            except IndexError as e:
                found = True
        if found and len(length) > 0:
            cycles.append((cycle, np.mean(length), np.std(length)))
    mean = []
    for cycle in cycles:
        mc = int(round(cycle[1]))
        if mc not in mean:
            mean.append(mc)
    return mean


def find_pair_cycle(cycles):
    # cycle_length = [x[0] for x in cycles]
    cycle_length = cycles
    pcycles = []
    passed = []
    cycle_set = set(cycle_length)
    for cycle in cycle_length:
        if cycle in passed:
            continue
        cand = [cycle]
        c = cycle
        while True:
            if c * 2 in cycle_set:
                c = c * 2
                cand.append(c)
            elif c * 2 + 1 in cycle_set:
                c = c * 2 + 1
                cand.append(c)
            elif c * 2 - 1 in cycle_set:
                c = c * 2 - 1
                cand.append(c)
            elif c * 2 + 2 in cycle_set:
                c = c * 2 + 2
                cand.append(c)
            elif c * 2 - 2 in cycle_set:
                c = c * 2 - 2
                cand.append(c)
            else:
                break
        if len(cand) > 1:
            pcycles.append(cand)
            passed.extend(cand)
    return pcycles


class Kline(object):
    def __init__(self, open, close, high, low, date):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.date = date

        self.left = 0
        self.right = 0

    def period(self):
        return 1 + self.left + self.right

    def merge(self, later, raising=True, keep_later=True):
        new_date = later.date if keep_later else self.date
        if raising:
            line = Kline(max(self.open, later.open), max(self.close, later.close),
                         max(self.high, later.high), max(self.low, later.low), new_date)
        else:
            line = Kline(min(self.open, later.open), min(self.close, later.close),
                         min(self.high, later.high), min(self.low, later.low), new_date)
        if keep_later:
            line.left = self.period() + later.left
            line.right = later.right
        else:
            line.left = self.left
            line.right = self.right + later.period()
        return line


class Stroke(object):
    def __init__(self, start_idx, end_idx, klines):
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.is_bull = klines[start_idx].high < klines[end_idx].high and klines[start_idx].low < klines[end_idx].low
        self.high = max(klines[start_idx].high, klines[end_idx].high)
        self.low = min(klines[start_idx].low, klines[end_idx].low)
        span = self.high - self.low
        self.low_extreme = self.low + span * 0.238 if self.is_bull else self.high - span * 0.238
        self.low_energy = self.low + span * 0.382 if self.is_bull else self.high - span * 0.382
        self.mid_energy = self.low + span * 0.5
        self.high_energy = self.low + span * 0.382 if not self.is_bull else self.high - span * 0.382
        self.minor_target = self.high + span * 0.382 if self.is_bull else self.low - span * 0.382
        self.first_target = self.high + span * 0.618 if self.is_bull else self.low - span * 0.618
        self.major_target = self.high + span if self.is_bull else self.low - span
        self.second_target = self.high + span * 1.618 if self.is_bull else self.low - span * 1.618

        within = [(self.low_energy, '0.382 of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down')),
                  (self.mid_energy,  '0.5   of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down')),
                  (self.high_energy, '0.618 of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down'))]
        outside = [(self.high if self.is_bull else self.low, 'end   of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down')),
                   (self.minor_target, '1.382 of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down')),
                   (self.first_target, '1.618 of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down')),
                   (self.major_target, '2.0 of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down')),
                   (self.second_target, '2.618 of\t{{}} {} to\t{{}}'.format('up' if self.is_bull else 'down'))]

        within = [w for w in within if w[0] > 1]
        outside = [o for o in outside if o[0] > 1]

        self.support = within if self.is_bull else outside
        self.pressure = outside if self.is_bull else within

        self.end_change = False
        self.valid = True

    def change_endpoint(self):
        if self.is_bull:
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

    def tostring(self):
        return '{} {} to {}'.format(self.start_idx, 'up' if self.is_bull else 'down', self.end_idx)


def find_strokes(points, klines):
    valid_points = [x for x in points if x[2]]
    strokes = [Stroke(valid_points[0][0], valid_points[1][0], klines)]

    for idx, point in enumerate(valid_points):
        if idx < 2:
            continue
        for stroke in strokes:
            if ((point[1] and ((klines[point[0]].close > stroke.low_extreme and not stroke.is_bull) or
                               (klines[point[0]].high > stroke.second_target and stroke.is_bull))) or
                    (not point[1] and ((klines[point[0]].close < stroke.low_extreme and stroke.is_bull) or
                                       (klines[point[0]].low < stroke.second_target and not stroke.is_bull)))):
                stroke.valid = False
                continue
            stroke.support = [p for p in stroke.support if p[0] < klines[point[0]].low]
            stroke.pressure = [p for p in stroke.pressure if p[0] > klines[point[0]].high]
            if (not stroke.end_change and
                    ((stroke.is_bull and klines[point[0]].close > stroke.high) or
                     (not stroke.is_bull and klines[point[0]].close < stroke.low))):
                stroke.end_change = True
                stroke.change_endpoint()

        strokes = [s for s in strokes if s.valid]

        early_strokes = [s for s in strokes if s.is_bull == point[1]]
        longest = {}
        for stroke in early_strokes:
            if stroke.start_idx in longest:
                longest[stroke.start_idx] = longest[stroke.start_idx] if ((klines[longest[stroke.start_idx]].high > klines[stroke.end_idx].high and point[1])
                                                                          or (klines[longest[stroke.start_idx]].low < klines[stroke.end_idx].low and not point[1])) else stroke.end_idx
            else:
                longest[stroke.start_idx] = stroke.end_idx
        starts = [start for start, end in longest.items()
                  if (klines[end].high < klines[point[0]].high and point[1])
                  or (klines[end].low > klines[point[0]].low and not point[1])]

        new_strokes = [Stroke(x, point[0], klines) for x in starts]
        new_strokes.append(Stroke(valid_points[idx - 1][0], point[0], klines))

        for stroke in new_strokes:
            if stroke.tostring() not in set([s.tostring() for s in strokes]):
                strokes.append(stroke)

    return strokes


def find_center_endpoints(klines, radius=3):
    ends = []  # index, is top point, points contained in top (radius by default)
    for idx, ochl in enumerate(klines):
        if (np.all([True] + [higher(klines[idx + x], klines[idx + x + 1]) for x in range(radius) if idx + x + 1 < len(klines)]) and
                np.all([True] + [higher(klines[idx - x], klines[idx - x - 1]) for x in range(radius) if idx - x - 1 >= 0])):
            ends.append((idx, True, radius))
        elif (np.all([True] + [lower(klines[idx + x], klines[idx + x + 1]) for x in range(radius) if idx + x + 1 < len(klines)]) and
              np.all([True] + [lower(klines[idx - x], klines[idx - x - 1]) for x in range(radius) if idx - x - 1 >= 0])):
            ends.append((idx, False, radius))
    return ends


def find_center_counts(ochl_list, ends, radius=3):
    end_idx = [x[0] for x in ends]
    return None

