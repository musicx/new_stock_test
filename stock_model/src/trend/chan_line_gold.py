import numpy as np
import pandas as pd
import QUANTAXIS as qa
import datetime as dt


def higher(x, y):
    return x[2] >= y[2] and x[3] >= y[3]


def contain(x, y):
    return x[2] >= y[2] and x[3] <= y[3]


def lower(x, y):
    return x[2] <= y[2] and x[3] <= y[3]


def merge_ochl(ochl_list, times):
    merged = []
    merged_cnt = []
    merged_last_times = []
    merge_idx = 0

    for raw_idx, ochl in enumerate(ochl_list):
        if raw_idx == 0:
            merged.append(ochl_list[raw_idx])
            merged_last_times.append(times[raw_idx])
            merged_cnt.append(1)
            continue
        if contain(merged[merge_idx], ochl) or contain(ochl, merged[merge_idx]):
            if merge_idx == 0 or merged[merge_idx-1][2] <= merged[merge_idx][2]:
                merged[merge_idx][0] = max(merged[merge_idx][0], ochl[0])
                merged[merge_idx][1] = max(merged[merge_idx][1], ochl[1])
                merged[merge_idx][2] = max(merged[merge_idx][2], ochl[2])
                merged[merge_idx][3] = max(merged[merge_idx][3], ochl[3])
            else:
                merged[merge_idx][0] = min(merged[merge_idx][0], ochl[0])
                merged[merge_idx][1] = min(merged[merge_idx][1], ochl[1])
                merged[merge_idx][2] = min(merged[merge_idx][2], ochl[2])
                merged[merge_idx][3] = min(merged[merge_idx][3], ochl[3])
            merged_last_times[merge_idx] = times[raw_idx]
            merged_cnt[merge_idx] += 1
        else:
            merged.append(ochl)
            merged_last_times.append(times[raw_idx])
            merged_cnt.append(1)
            merge_idx += 1

    return merged, merged_last_times, merged_cnt


def find_endpoints(merged_ochl_list, merged_count):
    end = []  # item is (idx in merged list, is top end point, valid end point)
    last = 0
    for idx, ochl in enumerate(merged_ochl_list):
        if idx < 1:
            continue
        insert = False
        if (idx == 1 or higher(merged_ochl_list[idx - 1], merged_ochl_list[idx - 2])) and higher(merged_ochl_list[idx - 1], ochl):
            end.append([idx - 1, True, True])
            insert = True
        elif (idx == 1 or lower(merged_ochl_list[idx - 1], merged_ochl_list[idx - 2])) and lower(merged_ochl_list[idx - 1], ochl):
            end.append([idx - 1, False, True])
            insert = True

        if insert and len(end) > 1 and end[-1][1] == end[-2][1]:
            if ((end[-1][1] and merged_ochl_list[end[-1][0]][2] > merged_ochl_list[end[-2][0]][2]) or
                    (not end[-1][1] and merged_ochl_list[end[-1][0]][3] < merged_ochl_list[end[-2][0]][3])):
                if idx - 2 == last and merged_count[last] + merged_count[last + 1] + merged_count[idx] > 3 or idx - 2 > last:
                    end[-2][2] = False
                else:
                    end.pop(-2)
            else:
                end.pop()
                insert = False
        elif insert and idx > 1 and not (idx - 2 == last and merged_count[last] + merged_count[last + 1] + merged_count[idx] > 3
                                         or idx - 2 > last):
            end.pop()
            insert = False

        if insert:
            last = idx
    return end


def find_cycle(points, periods, keep_invalid=False, mean_allowance=0.1, stroke_allowance=0):
    end_idx = dict([(p[0], (p[1], p[2])) for p in points if keep_invalid or p[2]])
    cumu = 0
    slices = []   # item is of (# or ticks, is Peak?)
    for idx, cnt in enumerate(periods):
        if idx in end_idx:
            # print('last point to current point: {}'.format(cumu))
            # print(('high' if end_idx[idx][0] else 'low') + ' ' + ('valid' if end_idx[idx][1] else 'unvalid') + ' period: ' + '{}'.format(cnt))
            if cumu != 0:
                slices.append((cumu, False))
            slices.append((cnt, True))
            cumu = 0
        else:
            cumu += cnt
    if len(slices) < 3:
        return []
    ticks = []
    for cnt, flag in slices:
        ticks.extend([flag] * cnt)
    start_point = slices[0][0] + slices[1][0]

    cycles = []
    for cycle in range(3, 120):
        # print('cycle {}'.format(cycle))
        short = int(np.floor(cycle * (1 - mean_allowance)))
        long = int(np.ceil(cycle * (1 + mean_allowance)))
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
                elif np.any(ticks[before+short: point+short]):
                    for p in range(last_point+short-1, before+short-1, -1):
                        if ticks[p] and p > last_point:
                            before = p
                    for p in range(before+short, last_point+short):
                        if ticks[p] and p > last_point:
                            point = p
                            after = p
                elif np.any(ticks[point+long: after+long]):
                    for p in range(after+long-1, last_point+long-1, -1):
                        if ticks[p] and p > last_point:
                            before = p
                            point = p
                    for p in range(last_point+long, after+long):
                        if ticks[p] and p > last_point:
                            after = p
                elif point < len(ticks):
                    break
                    # cannot find peak in given range
                if last_point == point:
                    break
                    # cannot move forward
            except IndexError as e:
                found = True
        if found:
            cycles.append(cycle)
    return cycles


def find_pair_cycle(cycles):
    pcycles = []
    passed = []
    cycle_set = set(cycles)
    for cycle in cycles:
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
            else:
                break
        if len(cand) > 1:
            pcycles.append(cand)
            passed.extend(cand)
    return pcycles


class Stroke(object):
    def __init__(self, start_idx, end_idx, ochl):
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.is_bull = ochl[start_idx][2] < ochl[end_idx][2] and ochl[start_idx][3] < ochl[end_idx][3]
        self.high = max(ochl[start_idx][2], ochl[end_idx][2])
        self.low = min(ochl[start_idx][3], ochl[end_idx][3])
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


def find_strokes(points, merged_ochl_list):
    valid_points = [x for x in points if x[2]]
    strokes = [Stroke(valid_points[0][0], valid_points[1][0], merged_ochl_list)]

    for idx, point in enumerate(valid_points):
        if idx < 2:
            continue
        for stroke in strokes:
            if ((point[1] and ((merged_ochl_list[point[0]][1] > stroke.low_extreme and not stroke.is_bull) or
                               (merged_ochl_list[point[0]][2] > stroke.second_target and stroke.is_bull))) or
                    (not point[1] and ((merged_ochl_list[point[0]][1] < stroke.low_extreme and stroke.is_bull) or
                                       (merged_ochl_list[point[0]][3] < stroke.second_target and not stroke.is_bull)))):
                stroke.valid = False
                continue
            stroke.support = [p for p in stroke.support if p[0] < merged_ochl_list[point[0]][3]]
            stroke.pressure = [p for p in stroke.pressure if p[0] > merged_ochl_list[point[0]][2]]
            if (not stroke.end_change and
                    ((stroke.is_bull and merged_ochl_list[point[0]][1] > stroke.high) or
                     (not stroke.is_bull and merged_ochl_list[point[0]][1] < stroke.low))):
                stroke.end_change = True
                stroke.change_endpoint()

        strokes = [s for s in strokes if s.valid]

        early_strokes = [s for s in strokes if s.is_bull == point[1]]
        longest = {}
        for stroke in early_strokes:
            if stroke.start_idx in longest:
                longest[stroke.start_idx] = longest[stroke.start_idx] if ((merged_ochl_list[longest[stroke.start_idx]][2] > merged_ochl_list[stroke.end_idx][2] and point[1])
                                                                          or (merged_ochl_list[longest[stroke.start_idx]][3] < merged_ochl_list[stroke.end_idx][3] and not point[1])) else stroke.end_idx
            else:
                longest[stroke.start_idx] = stroke.end_idx
        starts = [start for start, end in longest.items()
                  if (merged_ochl_list[end][2] < merged_ochl_list[point[0]][2] and point[1])
                    or (merged_ochl_list[end][3] > merged_ochl_list[point[0]][3] and not point[1])]

        new_strokes = [Stroke(x, point[0], merged_ochl_list) for x in starts]
        new_strokes.append(Stroke(valid_points[idx - 1][0], point[0], merged_ochl_list))

        for stroke in new_strokes:
            if stroke.tostring() not in set([s.tostring() for s in strokes]):
                strokes.append(stroke)

    return strokes


if __name__ == '__main__':
    today = dt.datetime.today()
    # today = dt.datetime(2018, 7, 6)
    start_date = today - dt.timedelta(days=999)

    # stocks = ['000528', '002049', '300529', '300607', '600518', '600588', '603877']
    # stocks = ['300638', '600516']
    stocks = ['603619']

    with open('../data/ps_analysis.txt', 'w') as f:
        for stock in stocks:
            can = qa.QA_fetch_stock_day_adv(stock, start=start_date.strftime('%Y-%m-%d'),
                                            end=today.strftime('%Y-%m-%d')).to_qfq()
            raw = can.data.loc[:, ['open', 'close', 'high', 'low']].values
            dates = [x.strftime('%Y-%m-%d') for x in can.data.reset_index().date.tolist()]

            f.write('stock code: {}\n'.format(stock))

            merged, merged_dates, merged_cnt = merge_ochl(raw, dates)
            ends = find_endpoints(merged, merged_cnt)

            for point in ends:
                f.write('{}: {} {} end\n'.format(merged_dates[point[0]],
                                                 'valid' if point[2] else 'invalid',
                                                 'top' if point[1] else 'bottom'))

            cycles = find_cycle(ends, merged_cnt)
            print("found cycles: {}".format(','.join(map(str, cycles))))

            pcycles = find_pair_cycle(cycles)
            print("found paired cycles: {}".format(', '.join(map(str, pcycles))))

            strokes = find_strokes(ends + [[len(merged) - 1, not ends[-1][1], True]], merged)

            support = sorted([(item[0], item[1].format(merged_dates[stroke.start_idx], merged_dates[stroke.end_idx]))
                              for stroke in strokes for item in stroke.support if stroke.end_idx != len(merged) - 1], key=lambda x: x[0], reverse=True)
            pressure = sorted([(item[0], item[1].format(merged_dates[stroke.start_idx], merged_dates[stroke.end_idx]))
                               for stroke in strokes for item in stroke.pressure if stroke.end_idx != len(merged) - 1], key=lambda x: x[0])

            f.write('current price: {}, {}\n'.format(dates[-1], raw[-1]))

            f.write('support\n')
            for sup in support:
                f.write('{:0.2f},\t{:0.4f},\t{}\n'.format(sup[0], sup[0] / raw[-1][1], sup[1]))

            f.write('pressure\n')
            for pre in pressure:
                f.write('{:0.2f},\t{:0.4f},\t{}\n'.format(pre[0], pre[0] / raw[-1][1], pre[1]))

            f.write('-------------\n\n')
            f.flush()

